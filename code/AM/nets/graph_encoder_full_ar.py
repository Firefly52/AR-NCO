import math

import torch
import torch.nn.functional as F
from torch import nn


def reshape_by_heads(qkv, head_num):
    batch_size, graph_size, _ = qkv.size()
    return qkv.view(batch_size, graph_size, head_num, -1).permute(0, 2, 1, 3)


def multi_head_attention(q, k, v):
    score = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(q.size(-1))
    attn = torch.softmax(score, dim=-1)
    out = torch.matmul(attn, v)
    batch_size, head_num, graph_size, qkv_dim = out.size()
    return out.permute(0, 2, 1, 3).contiguous().view(batch_size, graph_size, head_num * qkv_dim)


class RMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-6):
        super(RMSNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class FullAttnRes(nn.Module):

    def __init__(self, dim, max_history_len):
        super(FullAttnRes, self).__init__()
        self.w_l = nn.Parameter(torch.randn(max_history_len, dim) * 0.02)
        self.score_norm = RMSNorm(dim)

    def init_state(self, h0):
        max_len = self.w_l.size(0)
        h0_norm = self.score_norm(h0)
        scores0 = torch.einsum('ld,bnd->lbn', self.w_l, h0_norm)
        m = scores0
        denom = torch.ones_like(scores0)
        numer = h0.unsqueeze(0).expand(max_len, -1, -1, -1).contiguous()
        history_len = 1
        return m, denom, numer, history_len

    def step(self, state, new_h):
        m, denom, numer, history_len = state
        max_len = self.w_l.size(0)
        next_len = history_len + 1
        if next_len > max_len:
            raise ValueError("history_len={} exceeds max_history_len={}".format(next_len, max_len))

        new_h_norm = self.score_norm(new_h)
        scores = torch.einsum('ld,bnd->lbn', self.w_l, new_h_norm)
        valid = (torch.arange(1, max_len + 1, device=new_h.device) >= next_len)[:, None, None]

        m_new = torch.where(valid, torch.maximum(m, scores), m)
        exp_m = torch.exp(m - m_new)
        exp_s = torch.exp(scores - m_new)

        denom_new = torch.where(valid, denom * exp_m + exp_s, denom)
        numer_new = torch.where(
            valid.unsqueeze(-1),
            numer * exp_m.unsqueeze(-1) + exp_s.unsqueeze(-1) * new_h.unsqueeze(0),
            numer,
        )

        history_len = next_len
        out = numer_new[history_len - 1] / (denom_new[history_len - 1].unsqueeze(-1) + 1e-12)
        return (m_new, denom_new, numer_new, history_len), out


class EncoderLayer(nn.Module):

    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden=512,
            qkv_dim=None
    ):
        super(EncoderLayer, self).__init__()

        if qkv_dim is None:
            assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"
            qkv_dim = embed_dim // n_heads

        self.head_num = n_heads
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.Wq = nn.Linear(embed_dim, n_heads * qkv_dim, bias=False)
        self.Wk = nn.Linear(embed_dim, n_heads * qkv_dim, bias=False)
        self.Wv = nn.Linear(embed_dim, n_heads * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(n_heads * qkv_dim, embed_dim)

        if feed_forward_hidden > 0:
            self.ff1 = nn.Linear(embed_dim, feed_forward_hidden)
            self.ff2 = nn.Linear(feed_forward_hidden, embed_dim)
        else:
            self.ff1 = None
            self.ff2 = nn.Linear(embed_dim, embed_dim)

    def forward_mha(self, x):
        x = self.norm1(x)
        q = reshape_by_heads(self.Wq(x), self.head_num)
        k = reshape_by_heads(self.Wk(x), self.head_num)
        v = reshape_by_heads(self.Wv(x), self.head_num)
        return self.multi_head_combine(multi_head_attention(q, k, v))

    def forward_ffn(self, x):
        x = self.norm2(x)
        if self.ff1 is None:
            return self.ff2(x)
        return self.ff2(F.relu(self.ff1(x)))

    def forward(self, x):
        attn_out = self.forward_mha(x)
        ffn_out = self.forward_ffn(attn_out)
        return attn_out, ffn_out


class GraphAttentionEncoder(nn.Module):

    def __init__(
            self,
            n_heads,
            embed_dim,
            n_layers,
            node_dim=None,
            normalization='batch',
            feed_forward_hidden=512,
            qkv_dim=None
    ):
        super(GraphAttentionEncoder, self).__init__()

        self.init_embed = nn.Linear(node_dim, embed_dim) if node_dim is not None else None
        self.embed_dim = embed_dim
        self.n_layers = n_layers

        self.layers = nn.ModuleList([
            EncoderLayer(n_heads, embed_dim, feed_forward_hidden, qkv_dim)
            for _ in range(n_layers)
        ])

        self.max_history_len = 1 + 2 * n_layers
        self.attn_res = FullAttnRes(embed_dim, self.max_history_len)
        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        assert mask is None, "mask not supported!"

        if self.init_embed is not None:
            x = self.init_embed(x.view(-1, x.size(-1))).view(*x.size()[:2], -1)

        state = self.attn_res.init_state(x)
        agg = x
        history_outputs = [x]

        for layer in self.layers:
            mha_out = layer.forward_mha(x)
            history_outputs.append(mha_out)
            state, agg = self.attn_res.step(state, mha_out)

            ffn_out = layer.forward_ffn(agg)
            history_outputs.append(ffn_out)
            state, agg = self.attn_res.step(state, ffn_out)
            x = agg

        embeddings = self.final_norm(agg)
        graph_embedding = embeddings.mean(dim=1)
        return torch.stack(history_outputs, dim=0), embeddings, graph_embedding
