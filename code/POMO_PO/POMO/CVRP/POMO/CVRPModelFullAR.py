import torch
import torch.nn as nn

from CVRPModel import (
    CVRP_Decoder,
    Feed_Forward_Module,
    _get_encoding,
    multi_head_attention,
    reshape_by_heads,
)


class CVRPModel(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params

        self.encoder = CVRP_Encoder(**model_params)
        self.decoder = CVRP_Decoder(**model_params)
        self.encoded_nodes = None

    def pre_forward(self, reset_state):
        depot_xy = reset_state.depot_xy
        node_xy = reset_state.node_xy
        node_demand = reset_state.node_demand
        node_xy_demand = torch.cat((node_xy, node_demand[:, :, None]), dim=2)

        self.encoded_nodes = self.encoder(depot_xy, node_xy_demand)
        self.decoder.set_kv(self.encoded_nodes)

    def forward(self, state):
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)
        device = state.BATCH_IDX.device

        if state.selected_count == 0:
            selected = torch.zeros(size=(batch_size, pomo_size), dtype=torch.long, device=device)
            prob = torch.ones(size=(batch_size, pomo_size), device=device)

        elif state.selected_count == 1:
            selected = torch.arange(start=1, end=pomo_size + 1, device=device)[None, :].expand(batch_size, pomo_size)
            prob = torch.ones(size=(batch_size, pomo_size), device=device)

        else:
            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
            probs = self.decoder(encoded_last_node, state.load, ninf_mask=state.ninf_mask)
            if self.training or self.model_params['eval_type'] == 'softmax':
                while True:
                    with torch.no_grad():
                        selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1).squeeze(dim=1).reshape(batch_size, pomo_size)
                    prob = probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(batch_size, pomo_size)
                    if (prob != 0).all():
                        break
            else:
                selected = probs.argmax(dim=2)
                prob = None

        return selected, prob

    def route_forward(self, route_info):
        route = route_info.route
        load = route_info.load
        ninf_mask = route_info.ninf_mask

        batch_size = route.size(0)
        pomo_size = route.size(1)
        seq_len = route.size(2)
        problem_size = ninf_mask.size(2)

        encoded_route_node = _get_encoding(self.encoded_nodes, route)
        load = load.view(batch_size, -1)
        probs = self.decoder(encoded_route_node, load=load, ninf_mask=ninf_mask)
        probs = probs.view(batch_size, pomo_size, seq_len, problem_size)
        probs = probs[:, :, 1:-1, :]
        node_index_to_pick = route[:, :, 2:, None]
        prob = probs.gather(3, node_index_to_pick).squeeze(3)
        prob = torch.cat([torch.ones(batch_size, pomo_size, 2, device=prob.device), prob], dim=2)
        return prob


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class FullAttnRes(nn.Module):
    def __init__(self, dim, max_history_len):
        super().__init__()
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
            raise ValueError('history_len={} exceeds max_history_len={}'.format(next_len, max_len))

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


class CVRP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = model_params['embedding_dim']
        self.encoder_layer_num = model_params['encoder_layer_num']

        self.embedding_depot = nn.Linear(2, embedding_dim)
        self.embedding_node = nn.Linear(3, embedding_dim)
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(self.encoder_layer_num)])
        self.max_history_len = 1 + 2 * self.encoder_layer_num
        self.attn_res = FullAttnRes(embedding_dim, self.max_history_len)
        self.final_norm = nn.LayerNorm(embedding_dim)
    def forward(self, depot_xy, node_xy_demand):
        embedded_depot = self.embedding_depot(depot_xy)
        embedded_node = self.embedding_node(node_xy_demand)
        x = torch.cat((embedded_depot, embedded_node), dim=1)

        state = self.attn_res.init_state(x)
        agg = x

        for layer in self.layers:
            mha_out = layer.forward_mha(x)
            state, agg = self.attn_res.step(state, mha_out)

            ffn_out = layer.forward_ffn(agg)
            state, agg = self.attn_res.step(state, ffn_out)
            x = agg

        return self.final_norm(agg)


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        head_num = model_params['head_num']
        qkv_dim = model_params['qkv_dim']
        self.head_num = head_num

        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
        self.feed_forward = Feed_Forward_Module(**model_params)

    def forward_mha(self, x):
        x = self.norm1(x)
        q = reshape_by_heads(self.Wq(x), head_num=self.head_num)
        k = reshape_by_heads(self.Wk(x), head_num=self.head_num)
        v = reshape_by_heads(self.Wv(x), head_num=self.head_num)
        return self.multi_head_combine(multi_head_attention(q, k, v))

    def forward_ffn(self, x):
        x = self.norm2(x)
        return self.feed_forward(x)

    def forward(self, x):
        mha_out = self.forward_mha(x)
        ffn_out = self.forward_ffn(mha_out)
        return mha_out, ffn_out
