import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
    
class PositionEncoding2D(nn.Module):
    def __init__(self, in_size, ndim): #, translation=False
        super().__init__()
        # self.translation = translation
        n_encode = ndim // 2
        self.in_size = in_size
        coordinate = torch.meshgrid(torch.arange(in_size[0]), torch.arange(in_size[1]), indexing="ij")
        div_term = torch.exp(torch.arange(0, n_encode, 2).float() * (-math.log(10000.0) / n_encode)).view(-1, 1, 1) # d_model/4个
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
        return x + self.pe