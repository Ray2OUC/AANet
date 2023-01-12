import torch.nn as nn
import math
import copy
import torch.nn.functional as F
import torch
from einops.einops import rearrange

# Cross-Scale SimAM
class HieraSimAM_Block(nn.Module):
    def __init__(self, lambda_=1e-4):
        super(HieraSimAM_Block, self).__init__()
        self.lambda_ = lambda_
        self.activation = nn.Sigmoid()
    def forward(self, x):
        n, c, h, w = x.size()
        n = w * h
        # Locality Aggregation
        mean_4 = F.adaptive_avg_pool2d(x, 8)
        mean_3 = F.adaptive_avg_pool2d(mean_4, 4)
        mean_2 = F.adaptive_avg_pool2d(mean_3, 2)
        mean_1 = F.adaptive_avg_pool2d(mean_2, 1)
        mean_1 = F.interpolate(mean_1, size=(h, w), mode='nearest')
        mean_2 = F.interpolate(mean_2, size=(h, w), mode='nearest')
        mean_3 = F.interpolate(mean_3, size=(h, w), mode='nearest')
        mean_4 = F.interpolate(mean_4, size=(h, w), mode='nearest')
        # Cross-Scale Interaction
        max_mean = torch.stack([mean_1, mean_2, mean_3, mean_4], dim=1)
        max_mean, _ = torch.max(max_mean, dim=1)
        # Global Saliency Statistic
        d = (x - max_mean).pow(2)
        # Locality Variance Statistics    8*8-cells
        v = d.sum(dim=[2, 3], keepdim=True) / (n-64)
        # E_inv groups all importance of X
        E_inv = d / (4 * (v + self.lambda_)) + 0.5
        # Attentive Feature Recalibration
        return x * self.activation(E_inv)

#Sin-Cos Positional encoding
class PositionEncodingSine(nn.Module):
    """
    This is a sinusoidal position encoding that generalized to 2-dimensional images
    """

    def __init__(self, d_model, max_shape=(256, 256)):
        """
        Args:
            max_shape (tuple): for 1/8 featmap, the max length of 256 corresponds to 2048 pixels
        """
        super().__init__()

        pe = torch.zeros((d_model, *max_shape))
        y_position = torch.ones(max_shape).cumsum(0).float().unsqueeze(0)
        x_position = torch.ones(max_shape).cumsum(1).float().unsqueeze(0)
        div_term = torch.exp(torch.arange(0, d_model//2, 2).float() * (-math.log(10000.0) / d_model//2))
        div_term = div_term[:, None, None]  # [C//4, 1, 1]
        pe[0::4, :, :] = torch.sin(x_position * div_term)
        pe[1::4, :, :] = torch.cos(x_position * div_term)
        pe[2::4, :, :] = torch.sin(y_position * div_term)
        pe[3::4, :, :] = torch.cos(y_position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)  # [1, C, H, W]

    def forward(self, x):
        """
        Args:
            x: [N, C, H, W]
        """
        return x + self.pe[:, :, :x.size(2), :x.size(3)]

# We borrow the linear transformer defined in LoFTR(CVPR-2021) for capturing global dependencies
def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1

class LinearAttention(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.feature_map = elu_feature_map
        self.eps = eps

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """ Multi-Head linear attention proposed in "Transformers are RNNs"
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        # set padded position to zero
        if q_mask is not None:
            Q = Q * q_mask[:, :, None, None]
        if kv_mask is not None:
            K = K * kv_mask[:, :, None, None]
            values = values * kv_mask[:, :, None, None]

        v_length = values.size(1)
        values = values / v_length  # prevent fp16 overflow
        KV = torch.einsum("nshd,nshv->nhdv", K, values)  # (S,D)' @ S,V
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
        queried_values = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length

        return queried_values.contiguous()

class LoFTREncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 attention='linear'):
        super(LoFTREncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = LinearAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model*2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model*2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source, x_mask=None, source_mask=None):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        bs = x.size(0)
        query, key, value = x, source, source

        # multi-head attention
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        message = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)  # [N, L, (H, D)]
        message = self.merge(message.view(bs, -1, self.nhead*self.dim))  # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        return x + message

class FeatureTransformer_2(nn.Module):
    def __init__(self, d_model=256, n_head=8, n_layers=4):
        super(FeatureTransformer_2, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.n_layers = n_layers
        self.encoder1 = LoFTREncoderLayer(d_model, n_head)
        self.encoder2 = LoFTREncoderLayer(d_model, n_head)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat0, mask0=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
        """
        assert self.d_model == feat0.size(2), "the feature number of src and transformer must be equal"

        #for i in range(self.n_layers):
        feat0 = self.encoder1(feat0, feat0, mask0, mask0)
        feat0 = self.encoder2(feat0, feat0, mask0, mask0)

        return feat0

#AANet--descriptor network
class AANet(nn.Module):
    def __init__(self, input_chan=3):
        super(AANet, self).__init__()

        # 1x scale
        self.backbone_0 = nn.Sequential(
            nn.Conv2d(input_chan, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.InstanceNorm2d(32, affine=False),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.LeakyReLU(0.1, True),
            nn.InstanceNorm2d(64, affine=False),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.InstanceNorm2d(64, affine=False)
        )

        # 0.5x scale
        self.backbone_1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.InstanceNorm2d(128, affine=False),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.LeakyReLU(0.1, True),
            nn.InstanceNorm2d(64, affine=False),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.InstanceNorm2d(64, affine=False)
        )

        # 0.25x scale
        self.backbone_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.InstanceNorm2d(128, affine=False),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.LeakyReLU(0.1, True),
            nn.InstanceNorm2d(64, affine=False),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.InstanceNorm2d(64, affine=False)
        )

        # 0.5x scale
        self.backbone_3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.InstanceNorm2d(128, affine=False),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.LeakyReLU(0.2, True),
            nn.InstanceNorm2d(128, affine=False),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        )

        self.attentional_skip1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            HieraSimAM_Block(),
            nn.PReLU(),
            nn.InstanceNorm2d(128, affine=False),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        )

        self.attentional_skip2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            HieraSimAM_Block(),
            nn.PReLU(),
            nn.InstanceNorm2d(128, affine=False),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        )

        self.pos_encoding = PositionEncodingSine(128)
        self.transformer = FeatureTransformer_2(128, 8, 4)  # position encoding before use
        self.attentional_skip3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        )
        self.Merge_layer = nn.Sequential(
            HieraSimAM_Block(),
            nn.PReLU(),
            nn.InstanceNorm2d(192, affine=False),
            nn.Conv2d(192, 128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(128, affine=False),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        )
        #self.recon_layer = reconNet(output_chan=3)

    def _forw_impl(self, x):
        x = self.backbone_0(x)
        x1 = self.backbone_1(x)
        x2 = self.backbone_2(x1)
        x3 = self.backbone_3(x2)

        x1_skip = self.attentional_skip1(x1)

        x2_skip = self.attentional_skip2(x2)

        [n, c, h, w] = x3.size()
        x3_skip = rearrange(self.pos_encoding(x3), 'n c h w -> n (h w) c')
        del x1, x2, x3
        x3_skip = self.transformer(x3_skip)  # N L C
        x3_skip = x3_skip.view(n, h, w, c)  # N H W C
        x3_skip = x3_skip.permute(0, 3, 1, 2) # N C H W
        x3_skip = self.attentional_skip3(x3_skip)

        [n, c, h, w] = x2_skip.size()
        x3_skip = F.interpolate(x3_skip, size=[h, w], mode='bilinear', align_corners=True)
        x_fuse = torch.cat((x1_skip, x2_skip, x3_skip), dim=1)
        del x1_skip, x2_skip, x3_skip
        x_fuse = self.Merge_layer(x_fuse)
        #x_rec = self.recon_layer(x3, x_fuse)
        return x_fuse #, x_rec

    def forward(self, x):
        return self._forw_impl(x)

# Regression network
class reconNet(nn.Module):
    def __init__(self, output_chan=3):
        super(reconNet, self).__init__()

        self.Upsample_32 = nn.Sequential(
            nn.InstanceNorm2d(128, affine=False),
            nn.ReLU(True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.InstanceNorm2d(64, affine=False),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(64, affine=False),
            nn.ReLU(True)
        )
        self.Upsample_210 = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.InstanceNorm2d(64, affine=False),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(64, affine=False),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.InstanceNorm2d(64, affine=False),
            nn.ReLU(True),
            nn.Conv2d(64, output_chan, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )


    def forward(self, x3, x_des):
        x = self.Upsample_32(x3)
        x = torch.cat((x, x_des), dim=1)
        x = self.Upsample_210(x)
        return x

