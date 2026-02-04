import numpy as np
import torch
import torch.nn as nn
import kornia.contrib as kc
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer
# import trilinear


class ResidualDownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch)
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1, stride=stride) if (in_ch != out_ch or stride != 1) else nn.Identity()

    def forward(self, x):
        return F.silu(self.conv(x) + self.skip(x))

class MetricEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 64, 3, padding=1),  # [bs,64,512,512]
            nn.GroupNorm(8, 64),
            nn.SiLU(),

            ResidualDownBlock(64, 128, stride=2),
            ResidualDownBlock(128, 256, stride=2),
            ResidualDownBlock(256, 512, stride=2),

            nn.Conv2d(512, 1024, 1),
            nn.AdaptiveAvgPool2d((32, 32))
        )

    def forward(self, x):
        x = self.encoder(x)  # (bs, 1024, 32, 32)
        x = x.view(x.size(0), 1024, -1)  # (bs, 1024(dim), 1024)
        # x = x.transpose(1, 2)  # (bs, 1024, 1024(dim))
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch // 2, 1),
            nn.ReLU(),
            nn.Conv2d(out_ch // 2, out_ch // 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_ch // 2, out_ch, 1)
        )
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class MetricNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 64, 3, stride=2, padding=1),  # bs, 64, 256, 256
            ResidualBlock(64, 128),
            nn.MaxPool2d(2),  # bs, 128, 128, 128

            ResidualBlock(128, 256),
            nn.MaxPool2d(2),  # bs, 256, 64, 64

            ResidualBlock(256, 512),
            nn.MaxPool2d(2),  # bs, 512, 32, 32

            ResidualBlock(512, 1024),  # bs, 1024, 32, 32
        )

    def forward(self, x):
        x = self.encoder(x)  # (bs, 1024, 32, 32)
        x = x.view(x.size(0), 1024, -1)  # (bs, 1024(dim), 1024)
        # x = x.transpose(1, 2)  # (bs, 1024, 1024(dim))
        return x


class PredNoiseAndCoeff_Transformer(nn.Module):
    def __init__(self, conv_out_weight_init=None, bias_out_weight_init=None):
        super().__init__()
        # noise
        self.conv_out = nn.Conv2d(320, 4, kernel_size=3, padding=1)

        # ds
        self.splat_layers = nn.Sequential(
            nn.Conv2d(320, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        # Transformer-Conv
        self.transformer_conv = nn.Sequential(
            nn.TransformerEncoder(
                encoder_layer=nn.TransformerEncoderLayer(
                    d_model=256,
                    nhead=8,
                    dim_feedforward=1024,
                    activation='gelu',
                    batch_first=True
                ),
                num_layers=6
            ),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.coeffs_out = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1, groups=8),
            nn.GELU(),
            nn.Conv2d(512, 12 * 8, 1),
            nn.ReLU()
        )

        if conv_out_weight_init is not None and bias_out_weight_init is not None:
            self._init_conv_out(conv_out_weight_init, bias_out_weight_init)
            print("Successfully initialized conv_out weights.")

    def _init_conv_out(self, weight_init, bias_init):
        assert weight_init.shape == (4, 320, 3, 3), f"Expected weight_init shape (4, 320, 3, 3), but got {weight_init.shape}"
        assert bias_init.shape == (4,), f"Expected bias_init shape (4,), but got {bias_init.shape}"

        with torch.no_grad():
            self.conv_out.weight.copy_(weight_init)
            self.conv_out.bias.copy_(bias_init)

    def forward(self, x):  # bs, 320, 64, 64
        noise_pred = self.conv_out(x)  # bs, 4, 64, 64

        trans_out = self.splat_layers(x)  # bs, 256, 16, 16

        trans_out = trans_out.flatten(2).permute(0, 2, 1)  # bs, 16*16, 256

        trans_out = self.transformer_conv[0](trans_out)  # bs, 256, 256
        trans_out = trans_out.permute(0, 2, 1).view(-1, 256, 16, 16)  # bs, 256, 16, 16
        trans_out = self.transformer_conv[1:](trans_out).squeeze(2)  # bs, 256, 16, 16

        coeffs = self.coeffs_out(trans_out).view(-1, 12, 8, 16, 16)  # bs, 12, 8, 16, 16

        return noise_pred, coeffs


class PredNoiseAndCoeff_HDR(nn.Module):
    def __init__(self, conv_out_weight_init=None, bias_out_weight_init=None):
        super(PredNoiseAndCoeff_HDR, self).__init__()
        self.conv_out = nn.Conv2d(320, 4, kernel_size=3, padding=1)

        # ------------------------- coeffs -----------------------
        self.splat = nn.Sequential(
            ResidualDownBlock(320, 256, stride=2),  # bs, 256, 32, 32
            ResidualDownBlock(256, 256),  # bs, 256, 32, 32
        )

        # global
        self.global_conv = nn.Sequential(
            ResidualDownBlock(256, 256, stride=2),  # bs, 256, 16, 16
            nn.AdaptiveAvgPool2d(3),   # bs, 256, 3, 3
            nn.Conv2d(256, 128, 1),  # bs, 128, 3, 3
        )
        self.global_fc = nn.Sequential(
            nn.Flatten(),  # bs, 128*3*3=1152
            nn.Linear(1152, 256),
            TransformerEncoderLayer(d_model=256, nhead=4),
            nn.Linear(256, 128),  # bs, 128
        )

        # local
        self.local_conv = nn.Sequential(
            ResidualDownBlock(256, 128, stride=2),  # bs, 128, 16, 16
            ResidualDownBlock(128, 128),  # bs, 128, 16, 16
        )
        self.local_se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # bs, 256, 1, 1
            nn.Conv2d(128, 8, 1),  # bs, 8, 1, 1
            nn.SiLU(),
            nn.Conv2d(8, 128, 1),  # bs, 128, 1, 1
            nn.Sigmoid()
        )

        # fusion
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(128 * 2, 128, 1),
            nn.Sigmoid()
        )

        # pred
        self.pred = nn.Conv2d(128, 96, kernel_size=3, padding=1, groups=8)  # bs, 96, 16, 16

        # initial
        if conv_out_weight_init is not None and bias_out_weight_init is not None:
            self._init_conv_out(conv_out_weight_init, bias_out_weight_init)
            print("Successfully initialized conv_out weights.")

    def _init_conv_out(self, weight_init, bias_init):
        assert weight_init.shape == (4, 320, 3, 3), f"Expected weight_init shape (4, 320, 3, 3), but got {weight_init.shape}"
        assert bias_init.shape == (4,), f"Expected bias_init shape (4,), but got {bias_init.shape}"

        with torch.no_grad():
            self.conv_out.weight.copy_(weight_init)
            self.conv_out.bias.copy_(bias_init)

    def forward(self, x):
        noise_pred = self.conv_out(x)  # bs, 4, 64, 64

        # Splat
        splat = self.splat(x)  # bs, 256, 32, 32

        # global
        global_feat = self.global_conv(splat)   # bs, 128, 2, 2
        global_feat = self.global_fc(global_feat)  # bs, 128

        # local
        local_feat = self.local_conv(splat)  # bs, 128, 16, 16
        attn = self.local_se(local_feat)  # bs, 128, 16, 16
        local_feat = local_feat * attn  # bs, 128, 16, 16

        # fusion
        global_feat = global_feat.view(-1, 128, 1, 1)  # bs, 128, 1, 1
        global_feat = global_feat.expand_as(local_feat)  # bs, 128, 16, 16
        fusion = torch.cat([local_feat, global_feat], dim=1)  # bs, 256, 16, 16
        gate = self.fusion_gate(fusion)  # bs, 128, 16, 16
        fusion = local_feat * gate + global_feat * (1 - gate)  # bs, 128, 16, 16

        # pred
        coeffs = self.pred(fusion)  # bs, 96, 16, 16
        coeffs = coeffs.view(-1, 12, 8, 16, 16)  # bs, 12, 8, 16, 16

        return noise_pred, coeffs


def generate_multi_scale_masks(mask, sigma=5.0):
    """
    多尺度衰减掩码生成
    参数:
        mask (torch.Tensor): (bs, 32, 32) 浮点张量
        sigma (float): 衰减参数
    返回:
        torch.Tensor: (bs, 8-64, 1024) 的合并掩码
    """
    bs, H, W = mask.shape
    s = 64

    grid = torch.stack(torch.meshgrid(
        torch.linspace(0, H - 1, s),
        torch.linspace(0, W - 1, s),
        indexing='ij',
    ), dim=-1).round().int()  # (s, s, 2)

    coords = grid.reshape(1, s * s, 2).expand(bs, -1, -1)  # (bs, s*s, 2)

    attn_mask = batch_attention_mask(mask, coords, sigma)  # (bs, s*s, 32, 32)

    attn_mask = attn_mask.view(bs, s * s, -1)  # (bs, s*s, 1024)

    return attn_mask


def batch_attention_mask(mask, coords, sigma):
    """
    批量处理版本的距离衰减掩码
    参数:
        mask (torch.Tensor): (bs, 32, 32)
        coords (torch.Tensor): (bs, N, 2) 整数坐标
        sigma (float): 衰减参数
    """
    bs, N, _ = coords.shape

    target_vals = mask[torch.arange(bs)[:, None], coords[..., 0], coords[..., 1]]
    target_vals = target_vals.view(bs, N, 1, 1)

    region_mask = (mask.unsqueeze(1) == target_vals).clone().detach().to(torch.float32)

    return torch.where(
        region_mask.to(dtype=torch.bool),
        0.0,
        -10000.0
    )

    # distance = kc.distance_transform(region_mask.view(-1, 1, 32, 32))
    # distance = distance.view(bs, N, 32, 32)
    #
    # # decay = torch.exp(-distance.pow(2) / (2 * sigma ** 2))
    # # return region_mask + (1 - region_mask) * decay
    # decay = -distance.pow(2) / (2 * sigma ** 2)
    # return torch.where(region_mask.to(dtype=torch.bool), 0.0, decay)


def attn_interpolate_to_dim(input_tensor, target_dim, heads):
    """
    将形状为 (bs, 1, 4096, 1024) 的张量插值到 (bs, heads, target_dim, 1024)
    参数：
    input_tensor (torch.Tensor): 原始张量，形状为 (bs, 4096, 1024)
    target_dim (int): 目标维度
    heads (int): 注意力头数
    返回：
    torch.Tensor: 插值后的张量，形状为 (bs, heads, target_dim, 1024)
    """
    interpolated_tensor = F.interpolate(input_tensor, size=(target_dim, 1024), mode='bilinear', align_corners=True)
    interpolated_tensor = interpolated_tensor.expand(-1, heads, -1, -1)
    return interpolated_tensor


class PredNoiseAndCoeff(nn.Module):
    def __init__(self, conv_out_weight_init=None, bias_out_weight_init=None):
        super(PredNoiseAndCoeff, self).__init__()
        self.conv_out = nn.Conv2d(320, 4, kernel_size=3, stride=1, padding=1)  # [bs, 4, 64, 64]

        # self.coeffs_branch = nn.Sequential(
        #     nn.Conv2d(320, 128, kernel_size=3, stride=1, padding=1),  # [bs, 128, 64, 64]
        #     nn.ReLU(),
        #     nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),  # [bs, 128, 32, 32]
        #     nn.ReLU(),
        #     nn.Conv2d(128, 12 * 8, kernel_size=3, stride=2, padding=1),  # [bs, 96, 16, 16]
        # )
        self.coeffs_branch = nn.Sequential(
            nn.Conv2d(320, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 96, 3, 2, 1, output_padding=1),
        )

        if conv_out_weight_init is not None and bias_out_weight_init is not None:
            self._init_conv_out(conv_out_weight_init, bias_out_weight_init)
            print("Successfully initialized conv_out weights.")

    def _init_conv_out(self, weight_init, bias_init):
        assert weight_init.shape == (4, 320, 3, 3), f"Expected weight_init shape (4, 320, 3, 3), but got {weight_init.shape}"
        assert bias_init.shape == (4,), f"Expected bias_init shape (4,), but got {bias_init.shape}"

        with torch.no_grad():
            self.conv_out.weight.copy_(weight_init)
            self.conv_out.bias.copy_(bias_init)

            # self.conv_out.weight.requires_grad_(False)
            # self.conv_out.bias.requires_grad_(False)

    def forward(self, x):
        noise_pred = self.conv_out(x)  # [bs, 4, 64, 64]
        coeffs = self.coeffs_branch(x)  # [bs, 96, 128, 128]
        coeffs = coeffs.view(x.size(0), 12, 8, 128, 128)  # Reshape to [bs, 12, 8, 128, 128]

        return noise_pred, coeffs

class PredNoiseAnd3DLUT(nn.Module):
    def __init__(self, lut_dim=33, K=5, conv_out_weight_init=None, bias_out_weight_init=None):
        super(PredNoiseAnd3DLUT, self).__init__()
        self.lut_dim = lut_dim
        self.K = K

        self.conv_out = nn.Conv2d(320, 4, kernel_size=3, stride=1, padding=1)  # [bs, 4, 64, 64]

        # -- LUT prediction branch: predict 5 LUTs
        self.lut_branch = nn.Sequential(
            nn.Conv2d(320, 256, 3, 1, 1),  # [bs, 256, 64, 64]
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # Global average pooling to get feature vector [bs, 256, 1, 1]
            nn.Conv2d(256, 3 * lut_dim ** 3 * K, 1),  # Output parameters for K LUTs [bs, 3*lut_size^3*K, 1, 1]
        )

        # -- Routing network: predict per-pixel weights for each LUT
        self.routing_net = nn.Sequential(
            nn.Conv2d(320, K, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(K, K, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(K, K, kernel_size=4, stride=2, padding=1),  # 2x 上采样
            nn.Conv2d(K, K, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(K, K, kernel_size=4, stride=2, padding=1),  # 2x 上采样
            nn.Conv2d(K, K, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(K, K, kernel_size=4, stride=2, padding=1),  # 2x 上采样
        )

        self.trilinear = TrilinearInterpolation()

        if conv_out_weight_init is not None and bias_out_weight_init is not None:
            self._init_conv_out(conv_out_weight_init, bias_out_weight_init)
            print("Successfully initialized conv_out weights.")

    def _init_conv_out(self, weight_init, bias_init):
        assert weight_init.shape == (4, 320, 3, 3), f"Expected weight_init shape (4, 320, 3, 3), but got {weight_init.shape}"
        assert bias_init.shape == (4,), f"Expected bias_init shape (4,), but got {bias_init.shape}"

        with torch.no_grad():
            self.conv_out.weight.copy_(weight_init)
            self.conv_out.bias.copy_(bias_init)

            # self.conv_out.weight.requires_grad_(False)
            # self.conv_out.bias.requires_grad_(False)

    def forward(self, feat_map, img):
        noise_pred = self.conv_out(feat_map)  # [bs, 4, 64, 64]

        img_norm = (img * 0.5) + 0.5

        # -- LUT parameters prediction
        lut_params = self.lut_branch(feat_map)  # [bs, 3*lut_size^3*K, 1, 1]
        lut_params = lut_params.view(-1, self.K, 3, self.lut_dim, self.lut_dim, self.lut_dim)  # [bs, K, 3, lut_size, lut_size, lut_size]
        B = feat_map.shape[0]

        transformed_images = []
        for k in range(self.K):
            lut_k = lut_params[:, k]  # [B, 3, D, D, D]
            outs = []
            for i in range(B):
                single_lut = lut_k[i]  # [3, D, D, D]
                single_img = img_norm[i].unsqueeze(0)  # [1, 3, H, W]
                out_i = self.trilinear(single_lut, single_img)  # [1, 3, H, W]
                outs.append(out_i)
            out = torch.cat(outs, dim=0)
            transformed_images.append(out.unsqueeze(1))  # [B, 1, 3, H, W]
        lut_stack = torch.cat(transformed_images, dim=1)  # [B, K, 3, H, W]

        # -- Pixel-level weights prediction (for each LUT)
        pixel_weights = self.routing_net(feat_map)  # [B, K, 512, 512]
        pixel_weights = F.softmax(pixel_weights, dim=1)

        # -- Weighted sum per pixel
        R_exp = pixel_weights.unsqueeze(2)  # [B, K, 1, 512, 512]
        lut_image = (lut_stack * R_exp).sum(dim=1)  # [B, 3, 512, 512]

        return noise_pred, lut_image

class PredNoiseAnd3DLUTWR(nn.Module):
    def __init__(self, lut_dim=33, K=5, lambda_smooth=5e-4, lambda_monotonicity=10,
                 conv_out_weight_init=None, bias_out_weight_init=None):
        super(PredNoiseAnd3DLUTWR, self).__init__()
        self.lut_dim = lut_dim
        self.K = K
        self.lambda_smooth = lambda_smooth
        self.lambda_monotonicity = lambda_monotonicity

        self.conv_out = nn.Conv2d(320, 4, kernel_size=3, stride=1, padding=1)  # [bs, 4, 64, 64]

        # -- LUT prediction branch: predict 5 LUTs
        self.lut_branch = nn.Sequential(
            nn.Conv2d(320, 256, 3, 1, 1),  # [bs, 256, 64, 64]
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # Global average pooling to get feature vector [bs, 256, 1, 1]
            nn.Conv2d(256, 3 * lut_dim ** 3 * K, 1),  # Output parameters for K LUTs [bs, 3*lut_size^3*K, 1, 1]
        )

        # -- Routing network: predict per-pixel weights for each LUT
        self.routing_net = nn.Sequential(
            nn.Conv2d(320, K, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(K, K, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(K, K, kernel_size=4, stride=2, padding=1),  # 2x 上采样
            nn.Conv2d(K, K, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(K, K, kernel_size=4, stride=2, padding=1),  # 2x 上采样
            nn.Conv2d(K, K, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(K, K, kernel_size=4, stride=2, padding=1),  # 2x 上采样
        )

        self.trilinear = TrilinearInterpolation()

        self.tv3 = TV_3D(dim=lut_dim)

        if conv_out_weight_init is not None and bias_out_weight_init is not None:
            self._init_conv_out(conv_out_weight_init, bias_out_weight_init)
            print("Successfully initialized conv_out weights.")

    def _init_conv_out(self, weight_init, bias_init):
        assert weight_init.shape == (4, 320, 3, 3), f"Expected weight_init shape (4, 320, 3, 3), but got {weight_init.shape}"
        assert bias_init.shape == (4,), f"Expected bias_init shape (4,), but got {bias_init.shape}"

        with torch.no_grad():
            self.conv_out.weight.copy_(weight_init)
            self.conv_out.bias.copy_(bias_init)

            # self.conv_out.weight.requires_grad_(False)
            # self.conv_out.bias.requires_grad_(False)

    def forward(self, feat_map, img):
        noise_pred = self.conv_out(feat_map)  # [bs, 4, 64, 64]

        img_norm = (img * 0.5) + 0.5

        # -- LUT parameters prediction
        lut_params = self.lut_branch(feat_map)  # [bs, 3*lut_size^3*K, 1, 1]
        lut_params = lut_params.view(-1, self.K, 3, self.lut_dim, self.lut_dim, self.lut_dim)  # [bs, K, 3, lut_size, lut_size, lut_size]
        B = feat_map.shape[0]

        transformed_images = []
        for k in range(self.K):
            lut_k = lut_params[:, k]  # [B, 3, D, D, D]
            outs = []
            for i in range(B):
                single_lut = lut_k[i]  # [3, D, D, D]
                single_img = img_norm[i].unsqueeze(0)  # [1, 3, H, W]
                out_i = self.trilinear(single_lut, single_img)  # [1, 3, H, W]
                outs.append(out_i)
            out = torch.cat(outs, dim=0)
            transformed_images.append(out.unsqueeze(1))  # [B, 1, 3, H, W]
        lut_stack = torch.cat(transformed_images, dim=1)  # [B, K, 3, H, W]

        # -- Pixel-level weights prediction (for each LUT)
        pixel_weights = self.routing_net(feat_map)  # [B, K, 512, 512]
        pixel_weights = F.softmax(pixel_weights, dim=1)

        # -- Weighted sum per pixel
        R_exp = pixel_weights.unsqueeze(2)  # [B, K, 1, 512, 512]
        lut_image = (lut_stack * R_exp).sum(dim=1)  # [B, 3, 512, 512]

        #  Regularization loss
        weights_norm = torch.mean(pixel_weights ** 2)
        tv_cons, mn_cons = 0.0, 0.0
        for i in range(B):
            for k in range(self.K):
                single_lut = lut_params[i, k]  # [3, D, D, D]
                tv, mn = self.tv3(single_lut)  # 每次输入都是单个 LUT，即 4D 张量
                tv_cons += tv
                mn_cons += mn

        reg_loss = self.lambda_smooth * (weights_norm + tv_cons / B) + self.lambda_monotonicity * mn_cons / B

        return noise_pred, lut_image, reg_loss

class TrilinearInterpolation(nn.Module):
    """
    Pure-PyTorch 3D LUT trilinear interpolation using grid_sample.
    - lut: Tensor of shape [3, D, D, D], values represent color mapping in [0,1]
    - x:   Tensor of shape [B, 3, H, W], input image in [0,1]
    Returns:
    - out: Tensor of shape [B, 3, H, W], interpolated result
    """
    def __init__(self, align_corners=True):
        super().__init__()
        self.align_corners = align_corners

    def forward(self, lut, x):
        B, C, H, W = x.shape
        D = lut.shape[-1]
        # [B,3,D,D,D]
        lut_vol = lut.unsqueeze(0).expand(B, -1, -1, -1, -1)
        # build grid [B,1,H,W,3] with coords in [-1,1]
        grid = x.permute(0,2,3,1).unsqueeze(1) * 2.0 - 1.0
        # sample → [B,3,1,H,W]
        sampled = F.grid_sample(
            lut_vol, grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=self.align_corners
        )
        # squeeze depth → [B,3,H,W]
        return sampled.squeeze(2)

class PredNoiseAndRoutLUTs(nn.Module):
    """
    Region-aware LUT-based color adjustment model.
    Inputs:
        feat_map: tensor [B, 320, 64, 64] (features + score guidance)
        img:      tensor [B, 3, 512, 512] (original image)
    Outputs:
        out:      tensor [B, 3, 512, 512] (adjusted image)
    """
    def __init__(self, K=5, lut_dim=33, conv_out_weight_init=None, bias_out_weight_init=None):
        super(PredNoiseAndRoutLUTs, self).__init__()
        self.K = K
        self.lut_dim = lut_dim

        # -- Noise network
        self.conv_out = nn.Conv2d(320, 4, kernel_size=3, stride=1, padding=1)  # [bs, 4, 64, 64]

        # -- Routing network: predict per-pixel weights for each LUT
        self.routing_net = nn.Sequential(
            nn.Conv2d(320, K, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(K, K, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(K, K, kernel_size=4, stride=2, padding=1),  # 2x 上采样
            nn.Conv2d(K, K, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(K, K, kernel_size=4, stride=2, padding=1),  # 2x 上采样
            nn.Conv2d(K, K, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(K, K, kernel_size=4, stride=2, padding=1),  # 2x 上采样
        )

        # -- Learned 3D LUT basis
        self.luts = nn.ParameterList([
            nn.Parameter(torch.zeros(3, lut_dim, lut_dim, lut_dim))
            for _ in range(K)
        ])
        for i, lut in enumerate(self.luts):
            # identity initialization for first, zeros for others
            if i == 0:
                coords = torch.linspace(0, 1, lut_dim)
                grid = torch.stack(torch.meshgrid(coords, coords, coords, indexing='ij'), dim=-1)  # [D, D, D, 3]
                lut.data.copy_(grid.permute(3, 0, 1, 2))
            else:
                nn.init.zeros_(lut)

        # -- Trilinear interpolation module
        self.trilinear = TrilinearInterpolation()

        # -- Refinement network: upsample + conv to restore fine details
        self.refine = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, padding=1)
        )

        if conv_out_weight_init is not None and bias_out_weight_init is not None:
            self._init_conv_out(conv_out_weight_init, bias_out_weight_init)
            print("Successfully initialized conv_out weights.")

    def _init_conv_out(self, weight_init, bias_init):
        assert weight_init.shape == (4, 320, 3, 3), f"Expected weight_init shape (4, 320, 3, 3), but got {weight_init.shape}"
        assert bias_init.shape == (4,), f"Expected bias_init shape (4,), but got {bias_init.shape}"

        with torch.no_grad():
            self.conv_out.weight.copy_(weight_init)
            self.conv_out.bias.copy_(bias_init)

            # self.conv_out.weight.requires_grad_(False)
            # self.conv_out.bias.requires_grad_(False)

    def forward(self, feat_map, img):
        # feat_map: [B, 320, 64, 64], img: [B, 3, 512, 512]
        noise_pred = self.conv_out(feat_map)  # [bs, 4, 64, 64]

        # 1) Predict routing weights
        R = self.routing_net(feat_map)  # [B, K, 512, 512]
        R = F.softmax(R, dim=1)

        # 2) Downsample image to LUT resolution
        img_norm = (img * 0.5) + 0.5

        # 3) Apply each LUT via trilinear interpolation
        lut_outputs = []
        for lut_k in self.luts:
            out_k = self.trilinear(lut_k, img_norm)
            lut_outputs.append(out_k)
        lut_stack = torch.stack(lut_outputs, dim=1)  # [B, K, 3, 512, 512]

        # 4) Weighted sum per pixel
        R_exp = R.unsqueeze(2)  # [B, K, 1, 512, 512]
        lut_image = (lut_stack * R_exp).sum(dim=1)  # [B, 3, 512, 512]

        # 5) Refinement
        lut_image = self.refine(lut_image)
        return noise_pred, lut_image


class TV_3D(nn.Module):
    def __init__(self, dim=33):
        super(TV_3D, self).__init__()

        weight_r = torch.ones(3, dim, dim, dim - 1, dtype=torch.float)
        weight_r[:, :, :, (0, dim - 2)] *= 2.0
        weight_g = torch.ones(3, dim, dim - 1, dim, dtype=torch.float)
        weight_g[:, :, (0, dim - 2), :] *= 2.0
        weight_b = torch.ones(3, dim - 1, dim, dim, dtype=torch.float)
        weight_b[:, (0, dim - 2), :, :] *= 2.0
        self.relu = torch.nn.ReLU()

        # 注册为 buffer
        self.register_buffer('weight_r', weight_r)
        self.register_buffer('weight_g', weight_g)
        self.register_buffer('weight_b', weight_b)

    def forward(self, LUT):
        dif_r = LUT[:, :, :, :-1] - LUT[:, :, :, 1:]
        dif_g = LUT[:, :, :-1, :] - LUT[:, :, 1:, :]
        dif_b = LUT[:, :-1, :, :] - LUT[:, 1:, :, :]
        tv = torch.mean(torch.mul((dif_r ** 2), self.weight_r)) + torch.mean(
            torch.mul((dif_g ** 2), self.weight_g)) + torch.mean(torch.mul((dif_b ** 2), self.weight_b))

        mn = torch.mean(self.relu(dif_r)) + torch.mean(self.relu(dif_g)) + torch.mean(self.relu(dif_b))

        return tv, mn

class PredNoiseAndRoutLUTsWR(nn.Module):
    """
    Region-aware LUT-based color adjustment model.
    Inputs:
        feat_map: tensor [B, 320, 64, 64] (features + score guidance)
        img:      tensor [B, 3, 512, 512] (original image)
    Outputs:
        out:      tensor [B, 3, 512, 512] (adjusted image)
    """
    def __init__(self, K=9, lut_dim=33, lambda_smooth=1e-3, lambda_monotonicity=10,
                 conv_out_weight_init=None, bias_out_weight_init=None):
        super(PredNoiseAndRoutLUTsWR, self).__init__()
        self.K = K
        self.lut_dim = lut_dim
        self.lambda_smooth = lambda_smooth
        self.lambda_monotonicity = lambda_monotonicity

        # -- Noise network
        self.conv_out = nn.Conv2d(320, 4, kernel_size=3, stride=1, padding=1)  # [bs, 4, 64, 64]

        # -- Routing network: predict per-pixel weights for each LUT
        self.routing_net = nn.Sequential(
            nn.Conv2d(320, K, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(K, K, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(K, K, kernel_size=4, stride=2, padding=1),  # 2x 上采样
            nn.Conv2d(K, K, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(K, K, kernel_size=4, stride=2, padding=1),  # 2x 上采样
            nn.Conv2d(K, K, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(K, K, kernel_size=4, stride=2, padding=1),  # 2x 上采样
        )

        # -- Learned 3D LUT basis
        self.luts = nn.ParameterList([
            nn.Parameter(torch.zeros(3, lut_dim, lut_dim, lut_dim))
            for _ in range(K)
        ])
        for i, lut in enumerate(self.luts):
            # identity initialization for first, zeros for others
            if i == 0:
                coords = torch.linspace(0, 1, lut_dim)
                grid = torch.stack(torch.meshgrid(coords, coords, coords, indexing='ij'), dim=-1)  # [D, D, D, 3]
                lut.data.copy_(grid.permute(3, 0, 1, 2))
            else:
                nn.init.zeros_(lut)

        # -- Trilinear interpolation module
        self.trilinear = TrilinearInterpolation()

        # -- Refinement network
        self.refine = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, padding=1)
        )

        self.tv3 = TV_3D(dim=lut_dim)

        if conv_out_weight_init is not None and bias_out_weight_init is not None:
            self._init_conv_out(conv_out_weight_init, bias_out_weight_init)
            print("Successfully initialized conv_out weights.")

    def _init_conv_out(self, weight_init, bias_init):
        assert weight_init.shape == (4, 320, 3, 3), f"Expected weight_init shape (4, 320, 3, 3), but got {weight_init.shape}"
        assert bias_init.shape == (4,), f"Expected bias_init shape (4,), but got {bias_init.shape}"

        with torch.no_grad():
            self.conv_out.weight.copy_(weight_init)
            self.conv_out.bias.copy_(bias_init)

            # self.conv_out.weight.requires_grad_(False)
            # self.conv_out.bias.requires_grad_(False)

    def forward(self, feat_map, img):
        # feat_map: [B, 320, 64, 64], img: [B, 3, 512, 512]
        noise_pred = self.conv_out(feat_map)  # [bs, 4, 64, 64]

        # 1) Predict routing weights
        R = self.routing_net(feat_map)  # [B, K, 512, 512]
        R = F.softmax(R, dim=1)

        dh = R[:, :, :, :-1] - R[:, :, :, 1:]
        dw = R[:, :, :-1, :] - R[:, :, 1:, :]
        tv_R = dh.abs().mean() + dw.abs().mean()

        # 2) Downsample image to LUT resolution
        img_norm = (img * 0.5) + 0.5

        # 3) Apply each LUT via trilinear interpolation
        lut_outputs = []
        for lut_k in self.luts:
            out_k = self.trilinear(lut_k, img_norm)
            lut_outputs.append(out_k)
        lut_stack = torch.stack(lut_outputs, dim=1)  # [B, K, 3, 512, 512]

        # 4) Weighted sum per pixel
        R_exp = R.unsqueeze(2)  # [B, K, 1, 512, 512]
        lut_image = (lut_stack * R_exp).sum(dim=1)  # [B, 3, 512, 512]

        # 5) Refinement
        lut_image = self.refine(lut_image)

        # 6) Regularization loss
        weights_norm = torch.mean(R ** 2)
        tv_cons, mn_cons = 0.0, 0.0
        for lut in self.luts:
            tv, mn = self.tv3(lut)
            tv_cons += tv
            mn_cons += mn

        reg_loss = self.lambda_smooth * (weights_norm + tv_cons + tv_R) + self.lambda_monotonicity * mn_cons

        return noise_pred, lut_image, reg_loss

class UnetConvout(nn.Module):
    def __init__(self, conv_out_weight_init=None, bias_out_weight_init=None):
        super(UnetConvout, self).__init__()
        self.conv_out = nn.Conv2d(320, 4, kernel_size=3, stride=1, padding=1)  # [bs, 4, 64, 64]

        if conv_out_weight_init is not None and bias_out_weight_init is not None:
            self._init_conv_out(conv_out_weight_init, bias_out_weight_init)
            print("Successfully initialized conv_out weights.")

    def _init_conv_out(self, weight_init, bias_init):
        assert weight_init.shape == (4, 320, 3, 3), f"Expected weight_init shape (4, 320, 3, 3), but got {weight_init.shape}"
        assert bias_init.shape == (4,), f"Expected bias_init shape (4,), but got {bias_init.shape}"

        with torch.no_grad():
            self.conv_out.weight.copy_(weight_init)
            self.conv_out.bias.copy_(bias_init)

            # self.conv_out.weight.requires_grad_(False)
            # self.conv_out.bias.requires_grad_(False)

    def forward(self, x):
        noise_pred = self.conv_out(x)  # [bs, 4, 64, 64]

        return noise_pred

if __name__ == '__main__':
    x = torch.randn(2, 320, 64, 64)
    img = torch.randn(2, 3, 512, 512)
    pred = PredNoiseAnd3DLUTWR()
    noise_pred, coeffs = pred(x, img)
    print("Noise shape:", noise_pred.shape)
    print("Coeff shape:", coeffs.shape)
