import torch
from torch import nn

from src.models.components.graph_transformer import GraphTransformer
from src.models.components.multimodal_fusion import (
    MultimodalBottleneckTransformer, 
    PerceiverEncoder
    )
from src.models.components.encoding_layers import (
    PositionalEncoding, 
    LookupTableModalityEmbedding, 
    SinusoidalModalityEmbedding
    )


class TextOnlyBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self):
        pass


class KGAugmentedLM(nn.Module):
    def __init__(
        self, 
        latent_dim: int = 128, 
        num_unimodal_encoder_layers: int = 2,
        num_unimodal_encoder_heads: int = 2, 
        fusion_model: str = 'bottleneck',
        bottleneck_width: int = 4,
        num_fusion_layers: int = 2, 
        num_fusion_heads: int = 2,
        graph_vocab_size: int = 31090, 
        num_edge_types: int = 522,
        text_vocab_size: int = 30522, 
        modality_encoding: str = 'sinusoidal',):
        super().__init__()
        
        # EMBEDDINGS:
        self.node_emb = nn.Embedding(graph_vocab_size, latent_dim)
        self.text_emb = nn.Embedding(text_vocab_size, latent_dim)
        
        # POSITIONAL ENCODING:
        self.pos_enc = PositionalEncoding(latent_dim)
        
        # ENCODER: knowledge graph (uni-modal)
        self.graph_transformer_encoder = GraphTransformer(
            latent_dim=latent_dim, 
            num_layers=num_unimodal_encoder_layers, 
            num_edge_types=num_edge_types, 
            heads=num_unimodal_encoder_heads, 
            accept_adjacency_matrix=True)
        
        # ENCODER: text context (uni-modal)
        self.encoder_layer_in = nn.TransformerEncoderLayer(
            d_model=latent_dim, 
            nhead=num_unimodal_encoder_heads)
        self.text_transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer_in, 
            num_layers=num_unimodal_encoder_layers)
        
        # EMBEDDING: modality specific embedding
        if modality_encoding == 'sinusoidal':
            self.modality_embedding = SinusoidalModalityEmbedding(latent_dim, 3)
        elif modality_encoding == 'lookup':
            self.modality_embedding = LookupTableModalityEmbedding(latent_dim, 3)
        
        # ENCODER: multimodal fusion
        if fusion_model == 'bottleneck':
            self.fusion = MultimodalBottleneckTransformer(
                latent_dim=latent_dim, 
                num_layers=num_fusion_layers, 
                num_heads=num_fusion_heads, 
                fusion_layer=0,
                bottleneck_width=bottleneck_width,
                use_bottleneck=True)
            self.fusion_type = 'bottleneck'
        elif fusion_model == 'perceiver':
            self.fusion = PerceiverEncoder(
                depth=0,
                dim=0,
                queries_dim=0,
                logits_dim = None,
                num_latents = 512,
                latent_dim = 512,
                cross_heads = 1,
                latent_heads = 8,
                cross_dim_head = 64,
                latent_dim_head = 64,)
            self.fusion_type = 'perceiver'
        else:
            raise ValueError(f'Unknown fusion model: {fusion_model}')
        
        # DECODER: graph masked entity prediction (uni-modal)
        self.encoder_layer_out = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_unimodal_encoder_heads,)
        self.graph_transformer_decoder = nn.TransformerEncoder(
            self.encoder_layer_out, 
            num_layers=num_unimodal_encoder_layers)
        self.node_predictor = nn.Linear(latent_dim, graph_vocab_size)
        
        # DECODER: text generation (uni-modal)
        self.text_decoder_layer = nn.TransformerDecoderLayer(
            d_model=latent_dim, 
            nhead=num_unimodal_encoder_heads)
        self.text_transformer_decoder = nn.TransformerDecoder(
            self.text_decoder_layer, 
            num_layers=num_unimodal_encoder_layers)
        self.text_fc_out = nn.Linear(latent_dim, text_vocab_size)
                
        
    def forward(
        self,
        src_ids, 
        src_attention_mask=None, 
        trg_ids=None, 
        trg_attention_mask=None,
        node_ids=None, 
        adj_mat=None, 
        node_mask=None,
        mode='train',
        max_length=128):

        if mode in ['train', 'val', 'test']:
            return self.train_val_test_step(
                src_ids, 
                src_attention_mask, 
                trg_ids, 
                trg_attention_mask,
                node_ids, 
                adj_mat, 
                node_mask,)
        elif mode == 'predict':
            return self.predict_step(
                src_ids, 
                src_attention_mask, 
                node_ids, 
                adj_mat, 
                node_mask,
                max_length=max_length)
        
    
    def train_val_test_step(
        self, 
        src_ids, 
        src_attention_mask, 
        trg_ids, 
        trg_attention_mask,
        node_ids, 
        adj_mat, 
        node_mask,):
        
        # Shift trg_ids to create decoder input and target
        trg_input = trg_ids[:, :-1]
        text_target = trg_ids[:, 1:]
        
        # Adjust trg_attention_mask to match the dimensions of trg_input
        trg_attention_mask = trg_attention_mask[:, :-1] if trg_attention_mask.size(1) > 1 else None
        
        # embeddings
        node_emb = self.node_emb(node_ids)
        src_emb = self.pos_enc(self.text_emb(src_ids)).permute(1, 0, 2)
        trg_emb = self.pos_enc(self.text_emb(trg_input)).permute(1, 0, 2)
                
        # unimodal encoding
        node_emb, _ = self.graph_transformer_encoder(nodes=node_emb, adj_mat=adj_mat, mask=node_mask)
        src_emb = self.text_transformer_encoder(src=src_emb, src_key_padding_mask=src_attention_mask).permute(1, 0, 2)
        
        # multimodal fusion
        if self.fusion_type == 'bottleneck':
            fusion_emb = self.fusion({'nodes': node_emb, 'text': src_emb}, True)
        elif self.fusion_type == 'perceiver':
            fusion_emb = self.fusion(torch.cat([node_emb, src_emb], dim=1))
                
        # Predict node and edge features
        graph_dec_in = fusion_emb[:, :node_emb.shape[1]].permute(1, 0, 2)
        graph_dec_out = self.graph_transformer_decoder(graph_dec_in, src_key_padding_mask=node_mask)
        node_logits = self.node_predictor(graph_dec_out.permute(1, 0, 2))
        
        # Prepare causal mask for text generation
        seq_len = trg_input.size(1)  # trg_input has shape [batch_size, seq_len]
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len)

        # Predict text continuation
        decoder_out = self.text_transformer_decoder(
            tgt=trg_emb, 
            memory=fusion_emb.permute(1, 0, 2), 
            tgt_mask=tgt_mask,
            tgt_is_causal=True,
            tgt_key_padding_mask=trg_attention_mask)
        text_logits = self.text_fc_out(decoder_out.permute(1, 0, 2))
        
        return node_logits, text_logits, text_target
    
    
    def predict_step(
            self, 
            src_ids, 
            src_attention_mask, 
            node_ids, 
            adj_mat, 
            node_mask,
            max_length):
        
        # Initial token for autoregressive generation (e.g., start token)
        batch_size = src_ids.size(0)
        start_token_id = 1 
        end_token_id = 2
        
        # Initialize trg_ids for the batch
        trg_ids = torch.full((batch_size, 1), start_token_id, device=src_ids.device)

        # Initialize trg_attention_mask for the start token
        trg_attention_mask = torch.ones((1, 1), device=src_ids.device).bool()
        
        # Initialize a mask to track active sequences (not yet ended)
        active_sequences = torch.ones(batch_size, dtype=torch.bool, device=src_ids.device)
        
        # embeddings
        node_emb = self.node_emb(node_ids)
        src_emb = self.pos_enc(self.text_emb(src_ids)).permute(1, 0, 2)
        
        # unimodal encoding
        node_emb, _ = self.graph_transformer_encoder(nodes=node_emb, adj_mat=adj_mat, mask=node_mask)
        src_emb = self.text_transformer_encoder(src=src_emb, src_key_padding_mask=src_attention_mask).permute(1, 0, 2)
        
        # multimodal fusion
        if self.fusion_type == 'bottleneck':
            fusion_emb = self.fusion({'nodes': node_emb, 'text': src_emb}, False)
        elif self.fusion_type == 'perceiver':
            fusion_emb = self.fusion(torch.cat([node_emb, src_emb], dim=1))
                    
        # Autoregressive text generation
        generated_text_ids = []
        for _ in range(max_length):
            trg_emb = self.pos_enc(self.text_emb(trg_ids))
            seq_len = trg_ids.size(1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(sz=seq_len)
            
            # Update trg_key_padding_mask based on active_sequences
            trg_attention_mask = ~active_sequences.unsqueeze(1).expand(-1, seq_len)
            
            decoder_out = self.text_transformer_decoder(
                tgt=trg_emb.permute(1, 0, 2), 
                memory=fusion_emb.permute(1, 0, 2), 
                tgt_mask=tgt_mask,
                tgt_is_causal=True, 
                tgt_key_padding_mask=trg_attention_mask
            )
            
            # Extract the predicted token (typically, the token with the highest probability)
            next_token_id = decoder_out.argmax(dim=2)[-1, :].unsqueeze(1)
            trg_ids = torch.cat([trg_ids, next_token_id], dim=1)
            
            # Update the trg_attention_mask to include the new token
            trg_attention_mask = torch.cat([trg_attention_mask, torch.ones((batch_size, 1),dtype=torch.bool)], dim=1)
            
            # Update active sequences (mark as False if end token is generated)
            active_sequences &= (next_token_id.squeeze(-1) != end_token_id)

            # Optionally break if all sequences are inactive
            if not active_sequences.any():
                break
            
            generated_text_ids.append(next_token_id)
            

        return torch.cat(generated_text_ids, dim=1)
