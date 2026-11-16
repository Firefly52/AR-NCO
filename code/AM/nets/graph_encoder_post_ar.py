import torch
import numpy as np
from torch import nn
import math


class SkipConnection(nn.Module):
    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, input):
        return input + self.module(input)


class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim,
            val_dim=None,
            key_dim=None
    ):
        super(MultiHeadAttention, self).__init__()

        if val_dim is None:
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        self.W_out = nn.Parameter(torch.Tensor(n_heads, val_dim, embed_dim))

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None):
        if h is None:
            h = q

        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)
        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.input_dim

        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)

        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        K = torch.matmul(hflat, self.W_key).view(shp)
        V = torch.matmul(hflat, self.W_val).view(shp)

        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))

        if mask is not None:
            mask = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
            compatibility[mask] = -np.inf

        attn = torch.softmax(compatibility, dim=-1)

        if mask is not None:
            attnc = attn.clone()
            attnc[mask] = 0
            attn = attnc

        heads = torch.matmul(attn, V)

        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, n_query, self.embed_dim)

        return out


class Normalization(nn.Module):
    def __init__(self, embed_dim, normalization='batch'):
        super(Normalization, self).__init__()

        normalizer_class = {
            'batch': nn.BatchNorm1d,
            'instance': nn.InstanceNorm1d
        }.get(normalization, None)

        self.normalizer = normalizer_class(embed_dim, affine=True)

    def init_parameters(self):
        for name, param in self.named_parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input):
        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            assert self.normalizer is None
            return input


class MultiHeadAttentionLayer(nn.Sequential):
    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden=512,
            normalization='batch',
    ):
        super(MultiHeadAttentionLayer, self).__init__(
            SkipConnection(
                MultiHeadAttention(
                    n_heads,
                    input_dim=embed_dim,
                    embed_dim=embed_dim
                )
            ),
            Normalization(embed_dim, normalization),
            SkipConnection(
                nn.Sequential(
                    nn.Linear(embed_dim, feed_forward_hidden),
                    nn.ReLU(),
                    nn.Linear(feed_forward_hidden, embed_dim)
                ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim)
            ),
            Normalization(embed_dim, normalization)
        )


# ===================== 你要的【全新改写版】GraphAttentionEncoder =====================
class GraphAttentionEncoder(nn.Module):
    def __init__(
            self,
            n_heads,
            embed_dim,
            n_layers,
            node_dim=None,
            normalization='batch',
            feed_forward_hidden=512
    ):
        super(GraphAttentionEncoder, self).__init__()

        self.init_embed = nn.Linear(node_dim, embed_dim) if node_dim is not None else None
        self.embed_dim = embed_dim
        self.n_layers = n_layers

        # 堆叠多层注意力层
        self.layers = nn.ModuleList([
            MultiHeadAttentionLayer(n_heads, embed_dim, feed_forward_hidden, normalization)
            for _ in range(n_layers)
        ])

        # ===================== 层间自注意力模块 =====================
        # 可学习查询向量 Q (1, 1, embed_dim)
        self.layer_query = nn.Parameter(torch.randn(1, 1, embed_dim))

        # 从每层输出得到 K, V
        self.layer_proj_k = nn.Linear(embed_dim, embed_dim)
        self.layer_proj_v = nn.Linear(embed_dim, embed_dim)

        # 缩放因子
        self.norm_factor = 1 / math.sqrt(embed_dim)

        # 最后的归一化
        self.final_norm = Normalization(embed_dim, normalization)

    def forward(self, x, mask=None):
        assert mask is None, "mask not supported!"

        # 1. 初始嵌入
        if self.init_embed is not None:
            h = self.init_embed(x.view(-1, x.size(-1))).view(*x.size()[:2], -1)
        else:
            h = x

        # 2. 【保存初始嵌入和每一层输出】到列表，让 h0 也参与层融合
        layer_outputs = [h]
        current_h = h

        for layer in self.layers:
            current_h = layer(current_h)
            layer_outputs.append(current_h)  # 保存每一层

        # shape: [n_layers + 1, batch_size, graph_size, embed_dim]
        layer_outputs = torch.stack(layer_outputs, dim=0)

        # ===================== 层间自注意力 =====================
        batch_size = x.size(0)
        graph_size = x.size(1)

        # Q: 可学习查询 → [batch_size, 1, embed_dim]
        Q = self.layer_query.repeat(batch_size, 1, 1)

        # K, V 从所有层输出投影
        # layer_outputs: [L, B, N, D] → [B, N, L, D]
        layer_outputs = layer_outputs.permute(1, 2, 0, 3).contiguous()

        K = self.layer_proj_k(layer_outputs)  # [B, N, L, D]
        V = self.layer_proj_v(layer_outputs)  # [B, N, L, D]

        # 注意力计算（在层维度做 attention）
        attn_score = torch.matmul(Q.unsqueeze(1), K.transpose(-2, -1))  # [B, N, 1, L]
        attn_score = self.norm_factor * attn_score
        attn_weight = torch.softmax(attn_score, dim=-1)  # [B, N, 1, L]

        # 加权求和得到融合特征
        fused_feature = torch.matmul(attn_weight, V).squeeze(2)  # [B, N, D]

        # ===================== 归一化 =====================
        # h0 已经作为历史项参与融合，不再单独作为残差加回去
        fused_feature = self.final_norm(fused_feature)

        # 图级嵌入（平均）
        graph_embedding = fused_feature.mean(dim=1)

        # 返回：
        # 1. 所有层输出列表
        # 2. 融合后的节点特征（给解码器用）
        # 3. 融合后的图嵌入
        return layer_outputs, fused_feature, graph_embedding
