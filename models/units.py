import math

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from scipy import ndimage as sndi
from scipy import signal


def custom_sigmoid(x, dim=1, eps=1e-6):
    y = torch.softmax(x, dim=dim)
    y = y / torch.max(y, dim=dim, keepdim=True)[0].clamp_min(eps)
    return y


def coarse_center(img, img_ppi=500):
    # seg = sndi.gaussian_filter(img, sigma=19 * img_ppi / 500)
    # seg = sndi.grey_opening(seg, size=round(5 * img_ppi / 500))
    img = np.rint(img).astype(np.uint8)
    ksize1 = int(19 * img_ppi / 500)
    ksize2 = int(5 * img_ppi / 500)
    seg = cv2.GaussianBlur(img, ksize=(ksize1, ksize1), sigmaX=0, borderType=cv2.BORDER_REPLICATE)
    seg = cv2.morphologyEx(seg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize2, ksize2)))
    seg = seg.astype(np.float32)

    grid = np.stack(np.meshgrid(*[np.arange(x) for x in img.shape[:2]], indexing="ij")).reshape(2, -1)
    img_c = (seg.reshape(1, -1) * grid).sum(1) / seg.sum().clip(1e-6, None)
    return img_c


def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel, padding=padding), nn.ReLU(inplace=True))


def convbnrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class ImageGradient(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_x = [[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]]
        kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0)
        kernel_y = [[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]]
        kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0)
        self.weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
        self.weight_y = nn.Parameter(data=kernel_y, requires_grad=False)

    def forward(self, x):
        grad_x = F.conv2d(x, self.weight_x, padding=1)
        grad_y = F.conv2d(x, self.weight_y, padding=1)
        return grad_x, grad_y


class FingerprintOrientation(nn.Module):
    def __init__(self, win_size=16, stride=1):
        super().__init__()
        self.win_size = max(3, win_size // 2 * 2 + 1)
        self.stride = stride

        self.conv_grad = ImageGradient()
        self.conv_gaussian = ImageGaussian(self.win_size, self.win_size / 3.0)
        mean_kernel = torch.ones([self.win_size, self.win_size], dtype=torch.float32)[None, None] / self.win_size ** 2
        self.weight_avg = nn.Parameter(data=mean_kernel, requires_grad=False)

    def forward(self, x):
        assert x.size(1) == 1

        Gx, Gy = self.conv_grad(255 * x)
        Gxx = self.conv_gaussian(Gx ** 2)
        Gyy = self.conv_gaussian(Gy ** 2)
        Gxy = self.conv_gaussian(-Gx * Gy)
        sin2 = F.conv2d(2 * Gxy, self.weight_avg, padding=self.win_size // 2)
        cos2 = F.conv2d(Gxx - Gyy, self.weight_avg, padding=self.win_size // 2)
        ori = torch.atan2(sin2, cos2)[:, :, :: self.stride, :: self.stride]

        return torch.cat((torch.sin(ori), torch.cos(ori)), dim=1)


class FingerprintCompose(nn.Module):
    def __init__(self, win_size=8, do_norm=False, m0=0, var0=1.0, eps=1e-6):
        super().__init__()
        self.win_size = max(3, win_size // 2 * 2 + 1)

        self.norm = NormalizeModule(m0=m0, var0=var0, eps=eps) if do_norm else nn.Identity()
        self.conv_grad = ImageGradient()
        self.conv_gaussian = ImageGaussian(self.win_size, self.win_size / 3.0)
        mean_kernel = torch.ones([self.win_size, self.win_size], dtype=torch.float32)[None, None] / self.win_size ** 2
        self.weight_avg = nn.Parameter(data=mean_kernel, requires_grad=False)

    def forward(self, x):
        assert x.size(1) == 1

        Gx, Gy = self.conv_grad(x)
        Gxx = self.conv_gaussian(Gx ** 2)
        Gyy = self.conv_gaussian(Gy ** 2)
        Gxy = self.conv_gaussian(-Gx * Gy)
        sin2 = F.conv2d(2 * Gxy, self.weight_avg, padding=self.win_size // 2)
        cos2 = F.conv2d(Gxx - Gyy, self.weight_avg, padding=self.win_size // 2)

        x = torch.cat((x, sin2, cos2), dim=1)

        x = self.norm(x)

        return x


class FingerprintRepeat(nn.Module):
    def __init__(self, num_out=3) -> None:
        super().__init__()
        self.num_out = num_out

    def forward(self, x):
        assert x.size(1) == 1

        return x.repeat(1, self.num_out, 1, 1)


class DoubleConv(nn.Module):
    def __init__(self, in_chn, out_chn, do_bn=True, do_res=False):
        super().__init__()
        self.conv = (
            nn.Sequential(
                nn.Conv2d(in_chn, out_chn, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_chn),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(out_chn, out_chn, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_chn),
                nn.LeakyReLU(inplace=True),
            )
            if do_bn
            else nn.Sequential(
                nn.Conv2d(in_chn, out_chn, 3, padding=1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(out_chn, out_chn, 3, padding=1),
                nn.LeakyReLU(inplace=True),
            )
        )

        self.do_res = do_res
        if self.do_res:
            if out_chn < in_chn:
                self.original = nn.Conv2d(in_chn, out_chn, 1, padding=0)
            elif out_chn == in_chn:
                self.original = nn.Identity()
            else:
                self.original = ChannelPad(out_chn - in_chn)

    def forward(self, x):
        out = self.conv(x)
        if self.do_res:
            res = self.original(x)
            out = out + res
        return out


class DoubleConv_25D(nn.Module):
    def __init__(self, in_chn, out_chn, do_bn=True, do_res=False):
        super().__init__()
        self.conv = (
            nn.Sequential(
                nn.Conv3d(in_chn, out_chn, (3, 3, 1), padding=(1, 1, 0), bias=False),
                nn.BatchNorm3d(out_chn),
                nn.LeakyReLU(inplace=True),
                nn.Conv3d(out_chn, out_chn, (3, 3, 1), padding=(1, 1, 0), bias=False),
                nn.BatchNorm3d(out_chn),
                nn.LeakyReLU(inplace=True),
            )
            if do_bn
            else nn.Sequential(
                nn.Conv3d(in_chn, out_chn, (3, 3, 1), padding=(1, 1, 0)),
                nn.LeakyReLU(inplace=True),
                nn.Conv3d(out_chn, out_chn, (3, 3, 1), padding=(1, 1, 0)),
                nn.LeakyReLU(inplace=True),
            )
        )

        self.do_res = do_res
        if self.do_res:
            if out_chn < in_chn:
                self.original = nn.Conv3d(in_chn, out_chn, 1, padding=0)
            elif out_chn == in_chn:
                self.original = nn.Identity()
            else:
                self.original = ChannelPad(out_chn - in_chn)

    def forward(self, x):
        out = self.conv(x)
        if self.do_res:
            res = self.original(x)
            out = out + res
        return out


class ChannelPad(nn.Module):
    def __init__(self, after_C, before_C=0, value=0) -> None:
        super().__init__()
        self.before_C = before_C
        self.after_C = after_C
        self.value = value

    def forward(self, x):
        prev_0 = [0] * (x.ndim - 2) * 2
        out = F.pad(x, (*prev_0, self.before_C, self.after_C), value=self.value)
        return out


def gaussian_fn(win_size, std):
    n = torch.arange(0, win_size) - (win_size - 1) / 2.0
    sig2 = 2 * std ** 2
    w = torch.exp(-(n ** 2) / sig2) / (np.sqrt(2 * np.pi) * std)
    return w


def gkern(win_size, std):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = gaussian_fn(win_size, std)
    gkern2d = torch.outer(gkern1d, gkern1d)
    return gkern2d


class ImageGaussian(nn.Module):
    def __init__(self, win_size, std):
        super().__init__()
        self.win_size = max(3, win_size // 2 * 2 + 1)

        n = np.arange(0, win_size) - (win_size - 1) / 2.0
        gkern1d = np.exp(-(n ** 2) / (2 * std ** 2)) / (np.sqrt(2 * np.pi) * std)
        gkern2d = np.outer(gkern1d, gkern1d)
        gkern2d = torch.FloatTensor(gkern2d).unsqueeze(0).unsqueeze(0)
        self.gkern2d = nn.Parameter(data=gkern2d, requires_grad=False)

    def forward(self, x):
        x_gaussian = F.conv2d(x, self.gkern2d, padding=self.win_size // 2)
        return x_gaussian


class NormalizeInput(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)

    def forward(self, input):
        input = (input - input.mean(dim=(-1, -2), keepdim=True)) / input.std(dim=(-1, -2), keepdim=True)
        input = input * self.std.type_as(input) + self.mean.type_as(input)
        return input


class Gray2RGB(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, gray):
        assert gray.size(1) == 1
        return gray.repeat(1, 3, 1, 1)


class CrossAttention(nn.Module):
    def __init__(self, k_chn) -> None:
        super().__init__()
        self.C = k_chn
        self.W = nn.Linear(k_chn, k_chn, bias=False)

    def forward(self, a, b):
        a_shape = a.shape[2:]
        b_shape = b.shape[2:]

        a_flat = a.flatten(2)  # (B,C,WH)
        b_flat = b.flatten(2)
        S = torch.bmm(self.W(a_flat.transpose(1, 2)), b_flat)

        a_new = torch.bmm(a_flat, torch.softmax(S, 1)).reshape(-1, self.C, *a_shape)
        b_new = torch.bmm(b_flat, torch.softmax(S.transpose(1, 2), 1)).reshape(-1, self.C, *b_shape)

        return a_new, b_new


class DownSample(nn.Module):
    def __init__(self, scale_factor=2) -> None:
        super().__init__()
        self.scale_factor = 1.0 / scale_factor

    def forward(self, input, mode="nearest", align_corners=False):
        return F.interpolate(input, scale_factor=self.scale_factor, mode=mode, align_corners=align_corners)


class AlexNet(nn.Module):
    # ?????
    def __init__(self, num_out=3, num_layers=[64, 192, 384, 256, 256]):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
        )


def positionalencoding2d(channels, height, width):
    """
    :param channels: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: channels*height*width position matrix
    """
    if channels % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with " "odd dimension (got dim={:d})".format(channels))
    pe = torch.zeros(channels, height, width)
    # Each dimension use half of channels
    channels = int(channels / 2)
    div_term = 10000.0 ** (-torch.arange(0, channels, 2).float() / channels)
    pos_w = torch.arange(0.0, width).unsqueeze(1)
    pos_h = torch.arange(0.0, height).unsqueeze(1)
    pe[0:channels:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:channels:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[channels::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[channels + 1 :: 2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe


class Encoder(nn.Module):
    def __init__(self, num_layers, do_bn, in_chn=3, do_res=False, do_pos=False):
        super().__init__()
        self.n_layer = len(num_layers)
        self.do_pos = do_pos

        self.layer0 = DoubleConv(in_chn, num_layers[0], do_bn)
        if do_pos:
            in_channel = 2 * num_layers[0]
        else:
            in_channel = num_layers[0]
        for ii, out_channel in enumerate(num_layers[1:]):
            setattr(self, f"pool{ii}", nn.MaxPool2d(2, 2))
            setattr(self, f"layer{ii+1}", DoubleConv(in_channel, out_channel, do_bn=do_bn, do_res=do_res))
            in_channel = out_channel

    def forward(self, input):
        y = self.layer0(input)
        if self.do_pos:
            B, C, H, W = y.shape
            pos_enc = positionalencoding2d(C, H, W)
            y = torch.cat((y, pos_enc.type_as(y).repeat(B, 1, 1, 1)), dim=1)
        out = [y]
        for ii in range(self.n_layer - 1):
            y = getattr(self, f"pool{ii}")(y)
            y = getattr(self, f"layer{ii+1}")(y)
            out.append(y)

        return out


class Decoder(nn.Module):
    def __init__(self, in_channel, num_layers, out_channel, expansion=1, do_bn=True) -> None:
        super().__init__()
        self.n_layer = len(num_layers)

        for ii, cur_channel in enumerate(num_layers):
            setattr(self, f"layer{ii}", DoubleConv(in_channel * expansion, cur_channel * expansion, do_bn))
            setattr(self, f"upsample{ii}", nn.ConvTranspose2d(cur_channel * expansion, cur_channel * expansion, 2, 2))
            in_channel = cur_channel

        self.out = nn.Sequential(
            nn.Conv2d(in_channel * expansion, in_channel * expansion, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channel * expansion),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channel * expansion, out_channel, 1),
        )

    def forward(self, input):
        y = input
        out = []
        for ii in range(self.n_layer):
            y = getattr(self, f"layer{ii}")(y)
            out.append(y)
            y = getattr(self, f"upsample{ii}")(y)
        y = self.out(y)
        out.append(y)

        return out


class DecoderSkip(nn.Module):
    def __init__(self, in_channel, num_layers, out_channel, expansion=1, do_bn=True) -> None:
        super().__init__()
        self.n_layer = len(num_layers)

        for ii, cur_channel in enumerate(num_layers):
            setattr(self, f"upsample{ii}", nn.ConvTranspose2d(in_channel * expansion, cur_channel * expansion, 2, 2))
            setattr(self, f"layer{ii}", DoubleConv((cur_channel + cur_channel) * expansion, cur_channel * expansion, do_bn))
            in_channel = cur_channel

        self.out = nn.Sequential(nn.Conv2d(in_channel * expansion, out_channel, 1))

    def forward(self, inputs):
        y = inputs[0]
        for ii in range(self.n_layer):
            y = getattr(self, f"upsample{ii}")(y)
            y = getattr(self, f"layer{ii}")(torch.cat((inputs[ii + 1], y), dim=1))
        y = self.out(y)

        return y


class DecoderSkip2(nn.Module):
    def __init__(self, in_channel, num_layers, expansion=1, do_bn=True) -> None:
        super().__init__()
        self.n_layer = len(num_layers)

        for ii, cur_channel in enumerate(num_layers):
            setattr(self, f"upsample{ii}", nn.ConvTranspose2d(in_channel * expansion, cur_channel * expansion, 2, 2))
            setattr(self, f"layer{ii}", DoubleConv((cur_channel + cur_channel) * expansion, cur_channel * expansion, do_bn))
            in_channel = cur_channel

    def forward(self, inputs):
        y = inputs[0]
        for ii in range(self.n_layer):
            y = getattr(self, f"upsample{ii}")(y)
            y = getattr(self, f"layer{ii}")(torch.cat((inputs[ii + 1], y), dim=1))

        return y


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, dim), nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class BasicConv(nn.Module):
    def __init__(
        self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False
    ):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-6, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialGate, self).__init__()
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  # broadcasting
        return scale

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout)) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class FastCartoonTexture(nn.Module):
    def __init__(self, sigma=2.5, eps=1e-6) -> None:
        super().__init__()
        self.sigma = sigma
        self.eps = eps
        self.cmin = 0.3
        self.cmax = 0.7
        self.lim = 20

        self.img_grad = ImageGradient()

    def lowpass_filtering(self, img, L):
        img_fft = torch.fft.fftshift(torch.fft.fft2(img), dim=(-2, -1)) * L

        img_rec = torch.fft.ifft2(torch.fft.fftshift(img_fft, dim=(-2, -1)))
        img_rec = torch.real(img_rec)

        return img_rec

    def gradient_norm(self, img):
        Gx, Gy = self.img_grad(img)
        return torch.sqrt(Gx ** 2 + Gy ** 2) + self.eps

    def forward(self, input):
        H, W = input.size(-2), input.size(-1)
        grid_y, grid_x = torch.meshgrid(torch.linspace(-0.5, 0.5, H), torch.linspace(-0.5, 0.5, W), indexing="ij")
        grid_radius = torch.sqrt(grid_x ** 2 + grid_y ** 2) + self.eps

        L = (1.0 / (1 + (2 * np.pi * grid_radius * self.sigma) ** 4)).type_as(input)[None, None]

        grad_img1 = self.gradient_norm(input)
        grad_img1 = self.lowpass_filtering(grad_img1, L)

        img_low = self.lowpass_filtering(input, L)
        grad_img2 = self.gradient_norm(img_low)
        grad_img2 = self.lowpass_filtering(grad_img2, L)

        diff = grad_img1 - grad_img2
        flag = torch.abs(grad_img1)
        diff = torch.where(flag > 1, diff / flag.clamp_min(self.eps), torch.zeros_like(diff))

        weight = (diff - self.cmin) / (self.cmax - self.cmin)
        weight = torch.clamp(weight, 0, 1)

        cartoon = weight * img_low + (1 - weight) * input
        texture = (input - cartoon + self.lim) * 255 / (2 * self.lim)
        texture = torch.clamp(texture, 0, 255)
        return texture


class NormalizeModule(nn.Module):
    def __init__(self, m0=0.0, var0=1.0, eps=1e-6):
        super(NormalizeModule, self).__init__()
        self.m0 = m0
        self.var0 = var0
        self.eps = eps

    def forward(self, x):
        x_m = x.mean(dim=(1, 2, 3), keepdim=True)
        x_var = x.var(dim=(1, 2, 3), keepdim=True)
        y = (self.var0 * (x - x_m) ** 2 / x_var.clamp_min(self.eps)).sqrt()
        y = torch.where(x > x_m, self.m0 + y, self.m0 - y)
        return y

class ConvBnPRelu(nn.Module):
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_chn, out_chn, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_chn, eps=0.001, momentum=0.99)
        self.relu = nn.PReLU(out_chn, init=0)

    def forward(self, input):
        y = self.conv(input)
        y = self.bn(y)
        y = self.relu(y)
        return y

def gabor_bank(enh_ksize=25, ori_stride=2, sigma=4.5, Lambda=8, psi=0, gamma=0.5):
    grid_theta, grid_y, grid_x = torch.meshgrid(
        torch.arange(-90, 90, ori_stride, dtype=torch.float32),
        torch.arange(-(enh_ksize // 2), enh_ksize // 2 + 1, dtype=torch.float32),
        torch.arange(-(enh_ksize // 2), enh_ksize // 2 + 1, dtype=torch.float32),
        indexing="ij",
    )
    cos_theta = torch.cos(torch.deg2rad(-grid_theta))
    sin_theta = torch.sin(torch.deg2rad(-grid_theta))

    x_theta = grid_y * sin_theta + grid_x * cos_theta
    y_theta = grid_y * cos_theta - grid_x * sin_theta
    # gabor filters
    sigma_x = sigma
    sigma_y = float(sigma) / gamma
    exp_fn = torch.exp(-0.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2))
    gb_cos = exp_fn * torch.cos(2 * np.pi * x_theta / Lambda + psi)
    gb_sin = exp_fn * torch.sin(2 * np.pi * x_theta / Lambda + psi)

    return gb_cos[:, None], gb_sin[:, None]


def cycle_gaussian_weight(ang_stride=2, to_tensor=True):
    gaussian_pdf = signal.windows.gaussian(181, 3)
    coord = np.arange(ang_stride / 2, 180, ang_stride)
    delta = np.abs(coord.reshape(1, -1, 1, 1) - coord.reshape(-1, 1, 1, 1))
    delta = np.minimum(delta, 180 - delta) + 90
    if to_tensor:
        return torch.tensor(gaussian_pdf[delta.astype(int)]).float()
    else:
        return gaussian_pdf[delta.astype(int)].astype(np.float32)


def orientation_highest_peak(x, ang_stride=2):
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    filter_weight = cycle_gaussian_weight(ang_stride=ang_stride).type_as(x)
    return F.conv2d(x, filter_weight, stride=1, padding=0)


def select_max_orientation(x):
    x = x / torch.max(x, dim=1, keepdim=True).values.clamp_min(1e-6)
    x = torch.where(x > 0.999, x, torch.zeros_like(x))
    x = x / x.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return x


def padding_input(input):
    B, C, H, W = input.size()
    H_up = (H // 8 + 1) * 8
    W_up = (W // 8 + 1) * 8
    l_pad = (H_up - H) // 2
    r_pad = H_up - H - l_pad
    u_pad = (W_up - W) // 2
    b_pad = W_up - W - u_pad
    return F.pad(input, (l_pad, r_pad, u_pad, b_pad))


def gather_minumap(minumap, tar_shape=(128, 128)):
    B, C, H, W = minumap.size()
    grid_o1 = np.arange(6) * np.deg2rad(60)
    grid_o2 = np.arange(C) * np.deg2rad(360.0 / C)
    weight = np.abs(grid_o1[:, None] - grid_o2[None])
    weight = np.minimum(weight, 2 * np.pi - weight)
    weight = np.exp(-weight / (2 * 1 ** 2))
    weight = torch.tensor(weight[..., None, None]).type_as(minumap)
    minumap = F.conv2d(minumap, weight)
    minumap = F.interpolate(minumap, size=tar_shape, mode="bilinear", align_corners=True)
    return minumap


def create_minumap(minu, minu_mask, middel_shape, crop_shape, map_shape=(128, 128), pose=None, sigma=1):
    scale = middel_shape[0] * 1.0 / crop_shape[0]

    if pose is not None:
        theta = pose[:, 2] * 60
        cos_theta = torch.deg2rad(theta).cos()
        sin_theta = torch.deg2rad(theta).sin()
        R = torch.stack(
            (
                torch.stack((scale * cos_theta, scale * sin_theta), dim=1),
                torch.stack((scale * -sin_theta, scale * cos_theta), dim=1),
            ),
            dim=1,
        )
        t = pose[:, :2]
        minu_align = torch.zeros_like(minu)
        minu_align[..., :2] = torch.bmm(minu[..., :2] - t[:, None], R)
        minu_align[..., 2] = minu[..., 2] - theta[:, None]
    else:
        minu_align = torch.zeros_like(minu)
        minu_align[..., :2] = minu[..., :2] * scale
        minu_align[..., 2:] = minu[..., 2:]

    grid_o, grid_y, grid_x = torch.meshgrid(
        torch.arange(6) * 60.0, torch.linspace(-1, 1, map_shape[0]), torch.linspace(-1, 1, map_shape[1]), indexing="ij"
    )
    grid_o = grid_o.type_as(minu)[None, None]
    grid_x = grid_x.type_as(minu)[None, None]
    grid_y = grid_y.type_as(minu)[None, None]
    
    Cs = torch.exp(
        -((grid_x - minu_align[..., 0, None, None, None]) ** 2 + (grid_y - minu_align[..., 1, None, None, None]) ** 2)
        / (2 * (sigma * scale / map_shape[0] * 2) ** 2)
    )
    Co = torch.exp(-normlization_angle(grid_o - minu_align[..., 2, None, None, None]) / (2 * sigma ** 2))
    minu_map = torch.sum(Cs * Co * minu_mask[..., None, None, None], dim=1)
    return minu_map


def calculate_mask_local_patch(mask, local_thresh=0.5):
    H, W = mask.shape[-2:]
    ksize = ((H * 2 + 3 - 1) // (3 + 1), (W * 2 + 3 - 1) // (3 + 1))
    stride = ((ksize[0] + 1) // 2, (ksize[1] + 1) // 2)
    local_patch = F.avg_pool2d(mask, kernel_size=ksize, stride=stride, padding=0).flatten(1)
    return 1 * (local_patch.detach() > local_thresh)


def normlization_angle(delta):
    delta = torch.abs(delta) % 360
    return torch.deg2rad(torch.minimum(delta, 360 - delta))


def project_feature(project, feature):
    return torch.sigmoid(project) * feature


class ReGroupConv2D(nn.Module):
    def __init__(self, in_chns, out_chns, groups):
        super().__init__()
        self.in_chns = in_chns
        self.out_chns = out_chns
        self.groups = groups
        self.conv = nn.Conv2d(in_chns * groups, out_chns * groups, kernel_size=1, groups=groups)

    def forward(self, x):
        B, _, H, W = x.shape
        assert (H * W) == self.groups

        x = x.permute(0, 2, 3, 1).reshape(B, -1, 1, 1)
        x = self.conv(x)
        x = x.view(B, H, W, self.out_chns).permute(0, 3, 1, 2)
        return x


class LimitBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.layer = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=stride, stride=stride, padding=0),
            norm_layer(planes),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=1),
            norm_layer(planes),
        )
        self.relu = nn.LeakyReLU(inplace=True)

        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.layer(x)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out

class PositionEncoding2D(nn.Module):
    def __init__(self, in_size, ndim): #, translation=False
        super().__init__()
        # self.translation = translation
        n_encode = ndim // 2
        self.in_size = in_size
        coordinate = torch.meshgrid(torch.arange(in_size[0]), torch.arange(in_size[1]), indexing="ij")
        div_term = torch.exp(torch.arange(0, n_encode, 2).float() * (-math.log(10000.0) / n_encode)).view(-1, 1, 1) # d_model/4个
        # paper 更改正确
        pe = torch.cat(
            (
                torch.sin(coordinate[0].unsqueeze(0) * div_term),
                torch.cos(coordinate[0].unsqueeze(0) * div_term),
                torch.sin(coordinate[1].unsqueeze(0) * div_term),
                torch.cos(coordinate[1].unsqueeze(0) * div_term),
            ),
            dim=0,
        )
        self.div_term = div_term # B, d_model, 1, 1
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        # x, shift = x
        # assert ((shift is None) and (not self.translation)) or ((shift is not None) and self.translation)
        # B = x.size(0)
        # if self.translation:
        #     # recalculate the position encoding
        #     coordinate = torch.meshgrid(torch.arange(x.size(-2)), torch.arange(x.size(-1)), indexing="ij")
        #     i_coord = coordinate[0].unsqueeze(0).repeat(B, 1, 1).reshape(B, -1) - shift[:,1:2] * (self.in_size[0] // 2)
        #     j_coord = coordinate[1].unsqueeze(0).repeat(B, 1, 1).reshape(B, -1) - shift[:,0:1] * (self.in_size[1] // 2)
        #     i_coord = i_coord.reshape(B, self.in_size[0], self.in_size[1])
        #     j_coord = j_coord.reshape(B, self.in_size[0], self.in_size[1]) # B, H, W
        #     i_coord = i_coord.unsqueeze(1)
        #     j_coord = j_coord.unsqueeze(1)
        #     pe = torch.cat(
        #         (
        #             torch.sin(i_coord * self.div_term.unsqueeze(0)),  # B, d_model, 1, 1
        #             torch.cos(i_coord * self.div_term.unsqueeze(0)),
        #             torch.sin(j_coord * self.div_term.unsqueeze(0)),
        #             torch.cos(j_coord * self.div_term.unsqueeze(0)),
        #         ),
        #         dim=1,
        #     )
            # self.pe = pe.float().to(x.device)
            
        return x + self.pe


# __all__ = ['AdaptiveRotatedConv2d']


def _get_rotation_matrix(thetas):
    bs, g = thetas.shape
    device = thetas.device
    thetas = thetas.reshape(-1)  # [bs, g] --> [bs x g]
    
    x = torch.cos(thetas)
    y = torch.sin(thetas)
    x = x.unsqueeze(0).unsqueeze(0)  # shape = [1, 1, bs * g]
    y = y.unsqueeze(0).unsqueeze(0)
    a = x - y
    b = x * y
    c = x + y

    rot_mat_positive = torch.cat((
        torch.cat((a, 1-a, torch.zeros(1, 7, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 1, bs*g, device=device), x-b, b, torch.zeros(1, 1, bs*g, device=device), 1-c+b, y-b, torch.zeros(1, 3, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 2, bs*g, device=device), a, torch.zeros(1, 2, bs*g, device=device), 1-a, torch.zeros(1, 3, bs*g, device=device)), dim=1),
        torch.cat((b, y-b, torch.zeros(1,1 , bs*g, device=device), x-b, 1-c+b, torch.zeros(1, 4, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 4, bs*g, device=device), torch.ones(1, 1, bs*g, device=device), torch.zeros(1, 4, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 4, bs*g, device=device), 1-c+b, x-b, torch.zeros(1, 1, bs*g, device=device), y-b, b), dim=1),
        torch.cat((torch.zeros(1, 3, bs*g, device=device), 1-a, torch.zeros(1, 2, bs*g, device=device), a, torch.zeros(1, 2, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 3, bs*g, device=device), y-b, 1-c+b, torch.zeros(1, 1, bs*g, device=device), b, x-b, torch.zeros(1, 1, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 7, bs*g, device=device), 1-a, a), dim=1)
    ), dim=0)  # shape = [k^2, k^2, bs*g]

    rot_mat_negative = torch.cat((
        torch.cat((c, torch.zeros(1, 2, bs*g, device=device), 1-c, torch.zeros(1, 5, bs*g, device=device)), dim=1),
        torch.cat((-b, x+b, torch.zeros(1, 1, bs*g, device=device), b-y, 1-a-b, torch.zeros(1, 4, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 1, bs*g, device=device), 1-c, c, torch.zeros(1, 6, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 3, bs*g, device=device), x+b, 1-a-b, torch.zeros(1, 1, bs*g, device=device), -b, b-y, torch.zeros(1, 1, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 4, bs*g, device=device), torch.ones(1, 1, bs*g, device=device), torch.zeros(1, 4, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 1, bs*g, device=device), b-y, -b, torch.zeros(1, 1, bs*g, device=device), 1-a-b, x+b, torch.zeros(1, 3, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 6, bs*g, device=device), c, 1-c, torch.zeros(1, 1, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 4, bs*g, device=device), 1-a-b, b-y, torch.zeros(1, 1, bs*g, device=device), x+b, -b), dim=1),
        torch.cat((torch.zeros(1, 5, bs*g, device=device), 1-c, torch.zeros(1, 2, bs*g, device=device), c), dim=1)
    ), dim=0)  # shape = [k^2, k^2, bs*g]

    mask = (thetas >= 0).unsqueeze(0).unsqueeze(0)
    mask = mask.float()                                                   # shape = [1, 1, bs*g]
    rot_mat = mask * rot_mat_positive + (1 - mask) * rot_mat_negative     # shape = [k*k, k*k, bs*g]
    rot_mat = rot_mat.permute(2, 0, 1)                                    # shape = [bs*g, k*k, k*k]
    rot_mat = rot_mat.reshape(bs, g, rot_mat.shape[1], rot_mat.shape[2])  # shape = [bs, g, k*k, k*k] # 3 x 3 rotation matrix
    return rot_mat


def batch_rotate_multiweight(weights, lambdas, thetas):
    """
    Let
        batch_size = b
        kernel_number = n
        kernel_size = 3
    Args:
        weights: tensor, shape = [kernel_number, Cout, Cin, k, k]
        thetas: tensor of thetas,  shape = [batch_size, kernel_number]
    Return:
        weights_out: tensor, shape = [batch_size x Cout, Cin // groups, k, k]
    """
    assert(thetas.shape == lambdas.shape)
    assert(lambdas.shape[1] == weights.shape[0])

    b = thetas.shape[0]
    n = thetas.shape[1]
    k = weights.shape[-1]
    _, Cout, Cin, _, _ = weights.shape

    # Stage 1:
    # input: thetas: [b, n]
    #        lambdas: [b, n]
    # output: rotation_matrix: [b, n, 9, 9] (with gate) --> [b*9, n*9]

    #       Sub_Stage 1.1:
    #       input: [b, n] kernel
    #       output: [b, n, 9, 9] rotation matrix
    rotation_matrix = _get_rotation_matrix(thetas)

    #       Sub_Stage 1.2:
    #       input: [b, n, 9, 9] rotation matrix
    #              [b, n] lambdas
    #          --> [b, n, 1, 1] lambdas
    #          --> [b, n, 1, 1] lambdas dot [b, n, 9, 9] rotation matrix
    #          --> [b, n, 9, 9] rotation matrix with gate (done)
    #       output: [b, n, 9, 9] rotation matrix with gate
    lambdas = lambdas.unsqueeze(2).unsqueeze(3)
    rotation_matrix = torch.mul(rotation_matrix, lambdas) # gate on the rotation matrix

    #       Sub_Stage 1.3: Reshape
    #       input: [b, n, 9, 9] rotation matrix with gate
    #       output: [b*9, n*9] rotation matrix with gate
    rotation_matrix = rotation_matrix.permute(0, 2, 1, 3)
    rotation_matrix = rotation_matrix.reshape(b*9, n*9)

    # Stage 2: Reshape 
    # input: weights: [n, Cout, Cin, 3, 3]
    #             --> [n, 3, 3, Cout, Cin]
    #             --> [n*9, Cout*Cin] done
    # output: weights: [n*9, Cout*Cin]
    weights = weights.permute(0, 3, 4, 1, 2) 
    weights = weights.contiguous().view(n*9, Cout*Cin)


    # Stage 3: torch.mm
    # [b*9, n*9] x [n*9, Cout*Cin]
    # --> [b*9, Cout*Cin]
    weights = torch.mm(rotation_matrix, weights)

    # Stage 4: Reshape Back
    # input: [b*9, Cout*Cin]
    #    --> [b, 3, 3, Cout, Cin]
    #    --> [b, Cout, Cin, 3, 3]
    #    --> [b * Cout, Cin, 3, 3] done
    # output: [b * Cout, Cin, 3, 3]
    weights = weights.contiguous().view(b, 3, 3, Cout, Cin)
    weights = weights.permute(0, 3, 4, 1, 2)
    weights = weights.reshape(b * Cout, Cin, 3, 3)

    return weights


class AdaptiveRotatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=1, dilation=1, groups=1, bias=False,
                 kernel_number=1, rounting_func=None, rotate_func=batch_rotate_multiweight):
        super().__init__()

        self.kernel_number = kernel_number
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        self.rounting_func = rounting_func
        self.rotate_func = rotate_func

        self.weight = nn.Parameter(
            torch.Tensor(
                kernel_number, # kernel_number for different kernels
                out_channels,
                in_channels // groups,
                kernel_size,
                kernel_size,
            )
        )
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        # get alphas, angles
        # # [bs, Cin, h, w] --> [bs, n_theta], [bs, n_theta]
        alphas, angles = self.rounting_func(x)

        # rotate weight
        # # [Cout, Cin, k, k] --> [bs * Cout, Cin, k, k]
        rotated_weight = self.rotate_func(self.weight, alphas, angles)

        # reshape images
        bs, Cin, h, w = x.shape
        x = x.reshape(1, bs * Cin, h, w)  # [1, bs * Cin, h, w]
        
        # adaptive conv over images using group conv
        out = F.conv2d(input=x, weight=rotated_weight, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=(self.groups * bs))
        
        # reshape back
        out = out.reshape(bs, self.out_channels, *out.shape[2:])
        return out

    def extra_repr(self):

        s = ('{in_channels}, {out_channels}, kernel_number={kernel_number}'
             ', kernel_size={kernel_size}, stride={stride}, bias={bias}')
             
        if self.padding != (0,) * len([self.padding]):
            s += ', padding={padding}'
        if self.dilation != (1,) * len([self.dilation]):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        return s.format(**self.__dict__)
    

if __name__ == '__main__':
    PE =  PositionEncoding2D((16, 16), 128, translation=True)
    x = torch.rand(2, 128, 16, 16)
    shift = torch.rand(2, 2)
    y = PE(x, shift)

