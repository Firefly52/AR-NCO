from nets.attention_model_post_ar import AttentionModel as AttentionModelPostAR
from nets.graph_encoder_full_ar import GraphAttentionEncoder as FullARGraphAttentionEncoder


class AttentionModel(AttentionModelPostAR):

    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 problem,
                 n_encode_layers=2,
                 tanh_clipping=10.,
                 mask_inner=True,
                 mask_logits=True,
                 normalization='batch',
                 n_heads=8,
                 checkpoint_encoder=False,
                 shrink_size=None):
        super(AttentionModel, self).__init__(
            embedding_dim,
            hidden_dim,
            problem,
            n_encode_layers=n_encode_layers,
            tanh_clipping=tanh_clipping,
            mask_inner=mask_inner,
            mask_logits=mask_logits,
            normalization=normalization,
            n_heads=n_heads,
            checkpoint_encoder=checkpoint_encoder,
            shrink_size=shrink_size
        )

        self.embedder = FullARGraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=self.n_encode_layers,
            normalization=normalization
        )
