import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TSPModel(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.encoder = TSP_Encoder(**model_params)
        self.decoder = TSP_Decoder(**model_params)
        self.encoded_nodes = None
        self.entropy = torch.zeros(1)

    def pre_forward(self, reset_state):
        self.encoded_nodes = self.encoder(reset_state.problems)
        self.decoder.set_kv(self.encoded_nodes)

    def forward(self, state):
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)
        problem_size = state.ninf_mask.size(2)
        device = state.BATCH_IDX.device

        if state.current_node is None:
            selected = torch.arange(pomo_size, device=device)[None, :].expand(batch_size, pomo_size)
            selected = selected % problem_size
            prob = torch.ones(size=(batch_size, pomo_size), device=device)
            encoded_first_node = _get_encoding(self.encoded_nodes, selected)
            self.decoder.set_q1(encoded_first_node)

        elif not self.training and state.selected_count < 3 and pomo_size > problem_size:
            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
            probs = self.decoder(encoded_last_node, ninf_mask=state.ninf_mask)
            candidate_number = pomo_size // problem_size
            candidates = torch.topk(probs, candidate_number, dim=2)[1]
            index = torch.arange(candidate_number, device=device)[None, None, :].expand(batch_size, problem_size, candidate_number).transpose(1, 2).reshape(batch_size, pomo_size)
            selected = candidates[state.BATCH_IDX, state.POMO_IDX, index]
            prob = probs[state.BATCH_IDX, state.POMO_IDX, selected]

        else:
            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
            probs = self.decoder(encoded_last_node, ninf_mask=state.ninf_mask)
            self.store_entropy(probs)

            if self.training or self.model_params['eval_type'] == 'softmax':
                while True:
                    selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1).squeeze(dim=1).reshape(batch_size, pomo_size)
                    prob = probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(batch_size, pomo_size)
                    if (prob != 0).all():
                        break
            else:
                selected = probs.argmax(dim=2)
                prob = probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(batch_size, pomo_size)

        return selected, prob

    def route_forward(self, route_info):
        route = route_info.route
        batch_size = route.size(0)
        pomo_size = route.size(1)
        problem_size = route.size(2)
        start = route[:, :, 0]
        encoded_first_node = _get_encoding(self.encoded_nodes, start)
        self.decoder.set_q1(encoded_first_node, problem_size - 1)
        encoded_route_node = _get_encoding(self.encoded_nodes, route[:, :, :-1])
        probs = self.decoder(encoded_route_node, ninf_mask=route_info.ninf_mask)
        probs = probs.view(batch_size, pomo_size, problem_size - 1, problem_size)
        node_index_to_pick = route[:, :, 1:, None]
        prob = probs.gather(3, node_index_to_pick).squeeze(3)
        prob = torch.cat([torch.ones(batch_size, pomo_size, 1, device=prob.device), prob], dim=2)
        return prob

    @torch.no_grad()
    def store_entropy(self, probs):
        self.entropy = self.entropy.to(probs.device)
        self.entropy = self.entropy - (probs.detach() * torch.log(probs + 1e-10)).sum(dim=2).mean()


def _get_encoding(encoded_nodes, node_index_to_pick):
    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    index_shape = node_index_to_pick.shape
    embedding_dim = encoded_nodes.size(2)
    gathering_index = node_index_to_pick[..., None].expand(*index_shape, embedding_dim).contiguous()
    if len(index_shape) == 3:
        gathering_index = gathering_index.view(batch_size, -1, embedding_dim)
    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
    return picked_nodes


########################################
# 论文原版：Full Attention Residual
########################################

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class FullAttnRes(nn.Module):
    def __init__(self, dim, max_history_len, query_init_std=0.02):
        super().__init__()
        # 每个历史长度对应一个可学习查询向量
        self.w_l = nn.Parameter(torch.randn(max_history_len, dim) * query_init_std)
        # 注意力打分使用 RMSNorm(v): score = W^T · RMS(v)
        self.score_norm = RMSNorm(dim)
        self.max_history_len = max_history_len

    def init_state(self, h0: torch.Tensor, record_attention: bool = False):
        """初始化逐层聚合状态（不缓存历史列表）。

        这里的“逐层聚合”指在线累积：对每个未来长度 L 维护 softmax 的分母/分子，
        从而在不保存所有历史 h_i 的情况下，得到与“全历史注意力（k_i=v_i）”一致的聚合输出。

        Returns:
            state = (m, denom, numer, history_len)
            - m:     [MaxL, B, N]    每个 L 的在线最大值（log-sum-exp 稳定项）
            - denom: [MaxL, B, N]    sum_i exp(score_i - m)
            - numer: [MaxL, B, N, D] sum_i exp(score_i - m) * h_i
            - history_len: int       当前历史长度（已包含 h0）
        """
        max_len = self.w_l.size(0)
        h0_norm = self.score_norm(h0)
        scores0 = torch.einsum('ld,bnd->lbn', self.w_l, h0_norm)  # [MaxL, B, N]
        m = scores0
        denom = torch.ones_like(scores0)
        numer = h0.unsqueeze(0).expand(max_len, -1, -1, -1).contiguous()
        history_len = 1

        if record_attention:
            recorded_scores = [scores0[-1].detach().cpu()]  # score at L=13 (index -1)
            state = (m, denom, numer, history_len)
            return state, recorded_scores
        return m, denom, numer, history_len

    def step(self, state, new_h: torch.Tensor, record_attention: bool = False):
        """追加一条新历史并返回当前长度对应的聚合输出。

        new_h: [B, N, D]
        """
        m, denom, numer, history_len = state
        max_len = self.w_l.size(0)
        next_len = history_len + 1
        if next_len > max_len:
            raise ValueError(f"history_len={next_len} exceeds max_history_len={max_len}")

        new_h_norm = self.score_norm(new_h)
        scores = torch.einsum('ld,bnd->lbn', self.w_l, new_h_norm)  # [MaxL, B, N]
        # 只对 L >= next_len 的聚合器更新（因为更短的长度不包含 new_h）
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

        new_state = (m_new, denom_new, numer_new, history_len)
        if record_attention:
            return new_state, out, scores[-1].detach().cpu()  # score at L=13 (index -1)
        return new_state, out


########################################
# 编码器：100% 论文 Full AttnRes
########################################

class TSP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = model_params['embedding_dim']
        self.encoder_layer_num = model_params['encoder_layer_num']

        self.embedding = nn.Linear(2, embedding_dim)
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(self.encoder_layer_num)])
        # h0 + 每层两次历史追加（MHA后与FFN后）
        self.max_history_len = 1 + 2 * self.encoder_layer_num
        query_init_std = model_params.get('query_init_std', 0.02)
        self.attn_res = FullAttnRes(embedding_dim, self.max_history_len, query_init_std)
        self.final_norm = nn.LayerNorm(embedding_dim)
    def forward(self, data):
        record = self.model_params.get('record_attention', False)

        x = self.embedding(data)  # h0

        if record:
            state, recorded_scores = self.attn_res.init_state(x, record_attention=True)
        else:
            state = self.attn_res.init_state(x)

        agg = x

        # 每层两次聚合：h_l -> MHA -> 聚合(new_h) -> FFN -> 聚合(next_h)
        for layer in self.layers:
            mha_out = layer.forward_mha(x)
            if record:
                state, agg, new_score = self.attn_res.step(state, mha_out, record_attention=True)
                recorded_scores.append(new_score)
            else:
                state, agg = self.attn_res.step(state, mha_out)
            new_h = agg

            ffn_out = layer.forward_ffn(new_h)
            if record:
                state, agg, new_score = self.attn_res.step(state, ffn_out, record_attention=True)
                recorded_scores.append(new_score)
            else:
                state, agg = self.attn_res.step(state, ffn_out)
            x = agg

        if record:
            # stacked: [13, B, N] → softmax → [B, N, 13]
            stacked = torch.stack(recorded_scores, dim=0)  # [13, B, N]
            weights = F.softmax(stacked, dim=0)             # [13, B, N]
            self._recorded_weights = weights.permute(1, 2, 0).cpu()  # [B, N, 13]

        return self.final_norm(agg)


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        dim = model_params['embedding_dim']
        head_num = model_params['head_num']
        qkv_dim = model_params['qkv_dim']
        self.head_num = head_num

        # PreNorm: LayerNorm 放在每个子层计算之前
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # MHA
        self.Wq = nn.Linear(dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, dim)

        # FFN
        self.ff1 = nn.Linear(dim, model_params['ff_hidden_dim'])
        self.ff2 = nn.Linear(model_params['ff_hidden_dim'], dim)

    def forward_mha(self, x):
        x = self.norm1(x)
        q = reshape_by_heads(self.Wq(x), head_num=self.head_num)
        k = reshape_by_heads(self.Wk(x), head_num=self.head_num)
        v = reshape_by_heads(self.Wv(x), head_num=self.head_num)
        attn_out = self.multi_head_combine(multi_head_attention(q, k, v))
        return attn_out

    def forward_ffn(self, x):
        x = self.norm2(x)
        ffn_out = self.ff2(F.relu(self.ff1(x)))
        return ffn_out

    def forward(self, x):
        # 保留兼容接口：默认按顺序执行 MHA 后接 FFN
        attn_out = self.forward_mha(x)
        ffn_out = self.forward_ffn(attn_out)
        return attn_out, ffn_out


########################################
# 解码器：完全不变
########################################

class TSP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        decoder_layer_num = self.model_params.pop('decoder_layer_num', 1)
        self.layers = nn.ModuleList([DecoderLayer(mode='feature', **model_params) for _ in range(decoder_layer_num)])
        self.layers.append(DecoderLayer(mode='logit', **model_params))

    def set_kv(self, encoded_nodes):
        for layer in self.layers:
            layer.set_kv(encoded_nodes)

    def set_q1(self, encoded_q1, expand=None):
        self.layers[0].set_q1(encoded_q1, expand)

    def forward(self, encoded_last_node, ninf_mask):
        out = encoded_last_node
        for layer in self.layers:
            out = layer(out, ninf_mask)
        return out


class DecoderLayer(nn.Module):
    def __init__(self, mode='feature', **model_params):
        super().__init__()
        self.model_params = model_params
        self.mode = mode
        self.first = False
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        if mode == 'feature':
            self.Wq_first = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
            self.Wq_last = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
            self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
            self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
            self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
            self.addAndNormalization1 = Add_And_Normalization_Module(**model_params)
            self.feedForward = Feed_Forward_Module(**model_params)
            self.addAndNormalization2 = Add_And_Normalization_Module(**model_params)
        elif mode == 'logit':
            self.Wq_first = nn.Linear(embedding_dim, embedding_dim, bias=False)
            self.Wq_last = nn.Linear(embedding_dim, embedding_dim, bias=False)
            self.Wlogit_k = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.k = None
        self.v = None
        self.logitk = None
        self.q_first = None

    def set_kv(self, encoded_nodes):
        head_num = self.model_params['head_num']
        if self.mode == 'feature':
            self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num=head_num)
            self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num=head_num)
        elif self.mode == 'logit':
            self.logitk = self.Wlogit_k(encoded_nodes)

    def set_q1(self, encoded_q1, expand=None):
        batch_size = encoded_q1.shape[0]
        embedding_dim = encoded_q1.shape[2]
        self.first = True
        if expand is not None:
            encoded_q1 = encoded_q1.unsqueeze(2).expand(-1, -1, expand, -1).contiguous().view(batch_size, -1, embedding_dim)
        if self.mode == 'feature':
            self.q_first = reshape_by_heads(self.Wq_first(encoded_q1), head_num=self.model_params['head_num'])
        else:
            self.q_first = self.Wq_first(encoded_q1)

    def forward(self, input, ninf_mask=None):
        if self.mode == 'feature':
            head_num = self.model_params['head_num']
            q = reshape_by_heads(self.Wq_last(input), head_num=head_num)
            if self.first:
                q = q + self.q_first
            out_concat = multi_head_attention(q, self.k, self.v, rank3_ninf_mask=ninf_mask)
            multi_head_out = self.multi_head_combine(out_concat)
            out1 = self.addAndNormalization1(input, multi_head_out)
            out2 = self.feedForward(out1)
            out3 = self.addAndNormalization2(out1, out2)
            return out3

        elif self.mode == 'logit':
            q = self.Wq_last(input)
            if self.first:
                q = q + self.q_first
            score = torch.matmul(q, self.logitk.transpose(1, 2))
            sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
            logit_clipping = self.model_params['logit_clipping']
            score_scaled = score / sqrt_embedding_dim
            score_clipped = logit_clipping * torch.tanh(score_scaled)
            score_masked = score_clipped + ninf_mask
            probs = F.softmax(score_masked, dim=2)
            return probs


########################################
# 工具函数
########################################

def reshape_by_heads(qkv, head_num):
    batch_s = qkv.size(0)
    n = qkv.size(1)
    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    q_transposed = q_reshaped.transpose(1, 2)
    return q_transposed


def multi_head_attention(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):
    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)
    input_s = k.size(2)
    score = torch.matmul(q, k.transpose(2, 3))
    score_scaled = score / math.sqrt(key_dim)
    if rank2_ninf_mask is not None:
        score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_s, head_num, n, input_s)
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s)
    weights = nn.Softmax(dim=3)(score_scaled)
    out = torch.matmul(weights, v)
    out_transposed = out.transpose(1, 2)
    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
    return out_concat


class Add_And_Normalization_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)

    def forward(self, input1, input2):
        added = input1 + input2
        transposed = added.transpose(1, 2)
        normalized = self.norm(transposed)
        back_trans = normalized.transpose(1, 2)
        return back_trans


class Feed_Forward_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']
        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        return self.W2(F.relu(self.W1(input1)))