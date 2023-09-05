import torch
import torch.nn as nn
import torch.nn.functional as F


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
    
    
    
class MBT(nn.Module):
    def __init__(self, latent_dim, mlp_dim, num_layers, num_heads, dropout_rate=0.1,
                 attention_dropout_rate=0.1, stochastic_droplayer_rate=0.0,
                 modality_fusion=('kg', 'text'), fusion_layer=0,
                 use_bottleneck=False,
                 share_encoder=False):
        super(MBT, self).__init__()
        
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
                    if modality != 'rgb':
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
            if modality == 'rgb':
                name = ''
            else:
                name = '_' + modality
            x[modality] = self.add_positional_embed(x[modality], 'posembed_input' + name) #!

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
    
