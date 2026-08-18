
import torch, torch.nn as nn, torch.nn.functional as F
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from model.mamba_2 import OptimizedMamba2
from model.vertical_mamba import WindowedVisionMamba2Block

class LayerNorm2d(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, [self.weight.shape[0]], self.weight, self.bias, self.eps)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        return self.weight[:,None,None] * (x - u) / torch.sqrt(s + self.eps) + self.bias[:,None,None]

class MambaLayer2D(nn.Module):
    """使用已验证的 WVM2B（8x8 窗口双向 SSM）替代全序列 Mamba"""
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.mamba = WindowedVisionMamba2Block(d_model=dim, window_size=8, d_state=16, expand=1)
    def forward(self, x):
        B, C, H, W = x.shape
        # 确保 H,W 是 8 的倍数
        pad_h = (8 - H % 8) % 8
        pad_w = (8 - W % 8) % 8
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        # WVM2B expects [B, H*W, C] sequence format
        _, _, Hp, Wp = x.shape
        seq = x.reshape(B, C, Hp*Wp).transpose(-1, -2).contiguous()
        seq_out = self.mamba(seq, Hp, Wp)
        x_out = seq_out.transpose(-1, -2).reshape(B, C, Hp, Wp)
        if pad_h or pad_w:
            x_out = x_out[:, :, :H, :W]
        return x_out

class MambaEncoder2D(nn.Module):
    def __init__(self, in_chans=1, depths=[2,2,2,2], dims=[64,128,256,512]):
        super().__init__()
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(nn.Conv2d(in_chans, dims[0], kernel_size=7, stride=2, padding=3),
                            LayerNorm2d(dims[0], eps=1e-6, data_format="channels_first"))
        self.downsample_layers.append(stem)
        for i in range(3):
            self.downsample_layers.append(nn.Sequential(
                LayerNorm2d(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2)))
        self.stages = nn.ModuleList()
        for i in range(4):
            self.stages.append(nn.Sequential(*[MambaLayer2D(dim=dims[i]) for _ in range(depths[i])]))
        self.mlps = nn.ModuleList()
        for i in range(4):
            norm = LayerNorm2d(dims[i], eps=1e-6, data_format="channels_first")
            self.add_module(f'norm{i}', norm)
            self.mlps.append(nn.Sequential(nn.Conv2d(dims[i], 4*dims[i], 1), nn.GELU(), nn.Conv2d(4*dims[i], dims[i], 1)))
    def forward(self, x):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            norm = getattr(self, f'norm{i}')
            outs.append(self.mlps[i](norm(x)))
        return tuple(outs)

class SegMamba2D(nn.Module):
    def __init__(self, in_chans=1, out_chans=1, depths=[2,2,2,2], feat_size=[64,128,256,512], hidden_size=16, norm_name="batch", res_block=True):
        super().__init__()
        self.feat_size = feat_size
        self.mamba_encoder = MambaEncoder2D(in_chans, depths=depths, dims=feat_size)
        s = 2  # spatial_dims
        self.encoder1 = UnetrBasicBlock(spatial_dims=s, in_channels=in_chans, out_channels=feat_size[0], kernel_size=3, stride=1, norm_name=norm_name, res_block=res_block)
        self.encoder2 = UnetrBasicBlock(spatial_dims=s, in_channels=feat_size[0], out_channels=feat_size[1], kernel_size=3, stride=1, norm_name=norm_name, res_block=res_block)
        self.encoder3 = UnetrBasicBlock(spatial_dims=s, in_channels=feat_size[1], out_channels=feat_size[2], kernel_size=3, stride=1, norm_name=norm_name, res_block=res_block)
        self.encoder4 = UnetrBasicBlock(spatial_dims=s, in_channels=feat_size[2], out_channels=feat_size[3], kernel_size=3, stride=1, norm_name=norm_name, res_block=res_block)
        self.encoder5 = UnetrBasicBlock(spatial_dims=s, in_channels=feat_size[3], out_channels=hidden_size, kernel_size=3, stride=1, norm_name=norm_name, res_block=res_block)
        self.decoder5 = UnetrUpBlock(spatial_dims=s, in_channels=hidden_size, out_channels=feat_size[3], kernel_size=3, upsample_kernel_size=2, norm_name=norm_name, res_block=res_block)
        self.decoder4 = UnetrUpBlock(spatial_dims=s, in_channels=feat_size[3], out_channels=feat_size[2], kernel_size=3, upsample_kernel_size=2, norm_name=norm_name, res_block=res_block)
        self.decoder3 = UnetrUpBlock(spatial_dims=s, in_channels=feat_size[2], out_channels=feat_size[1], kernel_size=3, upsample_kernel_size=2, norm_name=norm_name, res_block=res_block)
        self.decoder2 = UnetrUpBlock(spatial_dims=s, in_channels=feat_size[1], out_channels=feat_size[0], kernel_size=3, upsample_kernel_size=2, norm_name=norm_name, res_block=res_block)
        self.decoder1 = UnetrBasicBlock(spatial_dims=s, in_channels=feat_size[0], out_channels=feat_size[0], kernel_size=3, stride=1, norm_name=norm_name, res_block=res_block)
        self.out = UnetOutBlock(spatial_dims=s, in_channels=feat_size[0], out_channels=out_chans)
    def forward(self, x):
        outs = self.mamba_encoder(x)
        enc1 = self.encoder1(x)
        enc2, enc3, enc4 = self.encoder2(outs[0]), self.encoder3(outs[1]), self.encoder4(outs[2])
        enc_hidden = self.encoder5(outs[3])
        dec3 = self.decoder5(enc_hidden, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        return self.out(self.decoder1(dec0))
