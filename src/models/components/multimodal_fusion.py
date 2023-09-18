import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, repeat

from functools import wraps


##################################################
####### Multimodal Bottleneck Transformer ########
##################################################

class MBTEncoderBlock(nn.Module):
    def __init__(self, latent_dim, mlp_dim, num_heads, dropout_rate=0.1, attention_dropout_rate=0.1,
                 droplayer_p=0.0):
        super(MBTEncoderBlock, self).__init__()
        
        self.dropout_rate = dropout_rate
        self.droplayer_p = droplayer_p
        
        self.mha = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            dropout=attention_dropout_rate
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_dim, latent_dim),
            nn.Dropout(dropout_rate)
        )

    def get_drop_pattern(self, x, deterministic):
        if not deterministic and self.droplayer_p:
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            return torch.bernoulli(
                self.droplayer_p * torch.ones(shape)).to(x.device).float()
        else:
            return torch.zeros_like(x)

    def forward(self, inputs, deterministic=True):
        # Attention block.
        x = F.layer_norm(inputs, normalized_shape=inputs.shape[1:])
        x = self.mha(x, x, x)[0]
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        drop_pattern = self.get_drop_pattern(x, deterministic)
        x = x * (1.0 - drop_pattern) + inputs

        # MLP block.
        y = F.layer_norm(x, normalized_shape=x.shape[1:])
        y = self.mlp(y)
        
        drop_pattern = self.get_drop_pattern(x, deterministic)
        y = y * (1.0 - drop_pattern) + x

        return y
    
    
    
class MultimodalBottleneckTransformer(nn.Module):
    def __init__(
        self, 
        latent_dim, 
        mlp_dim, 
        num_layers, 
        num_heads, 
        dropout_rate=0.1,
        attention_dropout_rate=0.1, 
        stochastic_droplayer_rate=0.0,
        modality_fusion=('nodes', 'edges', 'text'), 
        fusion_layer=0,
        use_bottleneck=True,
        share_encoder=False):
        super(MultimodalBottleneckTransformer, self).__init__()
        
        self.latent_dim = latent_dim
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads

        self.num_layers = num_layers
        self.modality_fusion = modality_fusion
        self.fusion_layer = fusion_layer
        self.use_bottleneck = use_bottleneck
        self.share_encoder = share_encoder

        self.encoders = nn.ModuleDict()

        for lyr in range(num_layers):
            droplayer_p = (lyr / max(num_layers - 1, 1)) * stochastic_droplayer_rate
            encoder_block = MBTEncoderBlock(
                latent_dim=latent_dim,
                mlp_dim=mlp_dim,
                num_heads=num_heads,
                dropout_rate=dropout_rate,
                attention_dropout_rate=attention_dropout_rate,
                droplayer_p=droplayer_p
            )

            self.encoders[f'encoder_{lyr}'] = encoder_block

            if not self.share_encoder:
                for modality in self.modality_fusion:
                    self.encoders[f'encoder_{lyr}_{modality}'] = MBTEncoderBlock(
                        latent_dim=latent_dim,
                        mlp_dim=mlp_dim,
                        num_heads=num_heads,
                        dropout_rate=dropout_rate,
                        attention_dropout_rate=attention_dropout_rate,
                        droplayer_p=droplayer_p
                    )

        # Other initialization code if needed
        # ...
        
    def add_positional_embed(self, x, feat_name):
        """Adds positional embedding."""
        assert x.dim() == 3  # (batch, len, emb)
        pos_emb = nn.Parameter(torch.randn(1, x.shape[1], x.shape[2]), requires_grad=True)
        return x + pos_emb

    def get_context(self, target_modality, modality_fusion, x):
        context = []
        for modality in modality_fusion:
            if modality != target_modality and modality in modality_fusion:
                context.append(x[modality])
        return context

    def combine_context(self, x, other_modalities):
        num_tokens = x.shape[1]
        other_modalities.append(x)
        x_combined = torch.cat(other_modalities, dim=1)
        return x_combined, num_tokens

    def forward(self, x: dict, bottleneck: torch.Tensor, train: bool):
        # Add positional embeddings
        for modality in self.modality_fusion:
            if modality == 'text':
                x[modality] = self.add_positional_embed(x[modality], 'posembed_input' + name) #!
            else:
                name = '_' + modality

        x_combined = None

        for lyr in range(self.num_layers):
            encoders = {}

            encoders['rgb'] = self.encoders[f'encoder_{lyr}']

            for modality in self.modality_fusion:
                if modality != 'rgb':
                    if self.share_encoder:
                        encoders[modality] = encoders['rgb']
                    else:
                        encoders[modality] = self.encoders[f'encoder_{lyr}_{modality}']

            if (lyr < self.fusion_layer or len(self.modality_fusion)) == 1:
                for modality in self.modality_fusion:
                    x[modality] = encoders[modality](x[modality], deterministic=not train)
            else:
                if self.use_bottleneck:
                    bottle = []
                    for modality in self.modality_fusion:
                        t_mod = x[modality].shape[1]
                        in_mod = torch.cat([x[modality], bottleneck], dim=1)
                        out_mod = encoders[modality](in_mod, deterministic=not train)
                        x[modality] = out_mod[:, :t_mod]
                        bottle.append(out_mod[:, t_mod:])
                    bottleneck = torch.mean(torch.stack(bottle, dim=-1), dim=-1)
                else:
                    if not self.share_encoder and len(self.modality_fusion) > 1:
                        x_new = {}
                        for modality in self.modality_fusion:
                            other_modalities = self.get_context(modality, self.modality_fusion, x)
                            combined_mods, t = self.combine_context(x[modality], other_modalities)
                            combined_mods = encoders[modality](combined_mods, deterministic=not train)
                            x_new[modality] = combined_mods[:, -t:]
                        x = x_new

                    elif self.share_encoder and len(self.modality_fusion) > 1:
                        if x_combined is None:
                            x_combined = []
                            for modality in self.modality_fusion:
                                x_combined.append(x[modality])
                            x_combined = torch.cat(x_combined, dim=1)
                        x_combined = encoders['rgb'](x_combined, deterministic=not train)

        if x_combined is not None:
            x_out = x_combined
        else:
            x_out = []
            for modality in self.modality_fusion:
                x_out.append(x[modality])
            x_out = torch.cat(x_out, dim=1)

        encoded = nn.LayerNorm(normalized_shape=x_out.shape[1:])(x_out)

        return encoded
    


##################################################
################# Perceiver ######################
##################################################


# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cache_fn(f):
    cache = None
    @wraps(f)
    def cached_fn(*args, _cache = True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache
    return cached_fn

# structured dropout, more effective than traditional attention dropouts

def dropout_seq(seq, mask, dropout):
    b, n, *_, device = *seq.shape, seq.device
    logits = torch.randn(b, n, device = device)

    if exists(mask):
        logits = logits.masked_fill(~mask, -torch.finfo(logits.dtype).max)

    keep_prob = 1. - dropout
    num_keep = max(1,  int(keep_prob * n))
    keep_indices = logits.topk(num_keep, dim = 1).indices

    batch_indices = torch.arange(b, device = device)
    batch_indices = rearrange(batch_indices, 'b -> b 1')

    seq = seq[batch_indices, keep_indices]

    if exists(mask):
        seq_counts = mask.sum(dim = -1)
        seq_keep_counts = torch.ceil(seq_counts * keep_prob).int()
        keep_mask = torch.arange(num_keep, device = device) < rearrange(seq_keep_counts, 'b -> b 1')

        mask = mask[batch_indices, keep_indices] & keep_mask

    return seq, mask

# helper classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)

        return self.fn(x, **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context = None, mask = None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim = -1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)

# main class

class PerceiverEncoder(nn.Module):
    def __init__(
        self,
        *,
        depth,
        dim,
        queries_dim,
        logits_dim = None,
        num_latents = 512,
        latent_dim = 512,
        cross_heads = 1,
        latent_heads = 8,
        cross_dim_head = 64,
        latent_dim_head = 64,
        weight_tie_layers = False,
        decoder_ff = False,
        seq_dropout_prob = 0.
    ):
        super().__init__()
        self.seq_dropout_prob = seq_dropout_prob

        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        self.cross_attend_blocks = nn.ModuleList([
            PreNorm(latent_dim, Attention(latent_dim, dim, heads = cross_heads, dim_head = cross_dim_head), context_dim = dim),
            PreNorm(latent_dim, FeedForward(latent_dim))
        ])

        get_latent_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, heads = latent_heads, dim_head = latent_dim_head))
        get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim))
        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        cache_args = {'_cache': weight_tie_layers}

        for i in range(depth):
            self.layers.append(nn.ModuleList([
                get_latent_attn(**cache_args),
                get_latent_ff(**cache_args)
            ]))

        self.decoder_cross_attn = PreNorm(queries_dim, Attention(queries_dim, latent_dim, heads = cross_heads, dim_head = cross_dim_head), context_dim = latent_dim)
        self.decoder_ff = PreNorm(queries_dim, FeedForward(queries_dim)) if decoder_ff else None

        self.to_logits = nn.Linear(queries_dim, logits_dim) if exists(logits_dim) else nn.Identity()

    def forward(
        self,
        data,
        mask = None,
        queries = None
    ):
        b, *_, device = *data.shape, data.device

        x = repeat(self.latents, 'n d -> b n d', b = b)

        cross_attn, cross_ff = self.cross_attend_blocks

        # structured dropout (as done in perceiver AR https://arxiv.org/abs/2202.07765)

        if self.training and self.seq_dropout_prob > 0.:
            data, mask = dropout_seq(data, mask, self.seq_dropout_prob)

        # cross attention only happens once for Perceiver IO

        x = cross_attn(x, context = data, mask = mask) + x
        x = cross_ff(x) + x

        # layers

        for self_attn, self_ff in self.layers:
            x = self_attn(x) + x
            x = self_ff(x) + x

        if not exists(queries):
            return x

        # make sure queries contains batch dimension

        if queries.ndim == 2:
            queries = repeat(queries, 'n d -> b n d', b = b)

        # cross attend from decoder queries to latents
        
        latents = self.decoder_cross_attn(queries, context = x)

        # optional decoder feedforward

        if exists(self.decoder_ff):
            latents = latents + self.decoder_ff(latents)

        # final linear out

        return self.to_logits(latents)
