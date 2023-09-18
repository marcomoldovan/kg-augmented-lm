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
        latent_dim, 
        num_unimodal_encoder_layers, 
        num_unimodal_encoder_heads, 
        fusion_model,
        num_fusion_layers, 
        num_fusion_heads,
        text_vocab_size, 
        graph_vocab_size, 
        num_edge_types,
        modality_encoding='sinusoidal',):
        super().__init__()
        
        # EMBEDDINGS:
        self.node_emb = nn.Embedding(graph_vocab_size, latent_dim)
        self.edge_emb = nn.Embedding(num_edge_types, latent_dim)
        self.text_emb = nn.Embedding(text_vocab_size, latent_dim)
        
        # POSITIONAL ENCODING:
        self.pos_enc = PositionalEncoding(latent_dim)
        
        # ENCODER: knowledge graph (uni-modal)
        self.graph_transformer_encoder = GraphTransformer(
            dim=latent_dim, 
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
        self.edge_predictor = nn.Linear(latent_dim, num_edge_types)
        
        # DECODER: text generation (uni-modal)
        self.text_decoder_layer = nn.TransformerDecoderLayer(
            d_model=latent_dim, 
            nhead=num_unimodal_encoder_heads)
        self.text_transformer_decoder = nn.TransformerDecoder(
            self.text_decoder_layer, 
            num_layers=num_unimodal_encoder_layers)
        self.text_fc_out = nn.Linear(latent_dim, latent_dim)
                
        
    def forward(
        self,
        query_ids,
        src_ids, 
        src_attention_mask, 
        trg_ids, 
        trg_attention_mask,
        node_ids, 
        adj_mat, 
        node_mask,
        train=True):

        if train:
            return self.train_step(
                src_ids, 
                src_attention_mask, 
                trg_ids, 
                trg_attention_mask,
                node_ids, 
                adj_mat, 
                node_mask,)
        else:
            return self.predict_step(
                src_ids, 
                src_attention_mask, 
                trg_ids, 
                trg_attention_mask,
                node_ids, 
                adj_mat, 
                node_mask,)
        
    
    def train_step(
        self, 
        src_ids, 
        src_attention_mask, 
        trg_ids, 
        trg_attention_mask,
        node_ids, 
        adj_mat, 
        node_mask,):
        #TODO permute batch and sequence dimension? ChatGPT does it in "Complex Multimodal Decoder Architecture"
        # embeddings
        node_emb = self.node_emb(node_ids)
        src_emb = self.pos_enc(self.text_emb(src_ids))
        trg_emb = self.pos_enc(self.text_emb(trg_ids))
        
        # unimodal encoding
        node_emb, edge_emb = self.graph_transformer_encoder(node_emb, adj_mat, node_mask)
        src_emb = self.text_transformer_encoder(src_emb, src_attention_mask)
        
        # multimodal fusion
        if self.fusion_type == 'bottleneck':
            fusion_emb = self.fusion({'nodes': node_emb, 'edges': edge_emb, 'text': src_emb})
        elif self.fusion_type == 'perceiver':
            fusion_emb = self.fusion(torch.cat([node_emb, edge_emb, src_emb], dim=1))
        
        fused_features = fusion_emb[:, :node_emb.shape[1]]
        
        # Predict node and edge features
        node_logits = self.node_predictor(fused_features)
        edge_logits = self.edge_predictor(fused_features)
        
        # Predict text continuation
        #TODO idea: force model to predict node id after every mention of a node. Increase factuality? <- this is a data problem, not a model problem.
        text_output = self.text_transformer_decoder(trg_emb, fused_features, trg_attention_mask)
        
        return node_logits, edge_logits, text_output
    
    
    def predict_step(
            self, 
            src_ids, 
            src_attention_mask, 
            node_ids, 
            adj_mat, 
            node_mask,
            max_length=100):
        
        # Initial token for autoregressive generation (e.g., start token)
        start_token_id = ...  # Define appropriately
        trg_ids = torch.tensor([[start_token_id]], device=src_ids.device)
        
        # embeddings
        node_emb = self.node_emb(node_ids)
        src_emb = self.pos_enc(self.text_emb(src_ids))
        
        # unimodal encoding
        node_emb, edge_emb = self.graph_transformer_encoder(node_emb, adj_mat, node_mask)
        src_emb = self.text_transformer_encoder(src_emb, src_attention_mask)
        
        # multimodal fusion
        if self.fusion_type == 'bottleneck':
            fusion_emb = self.fusion({'nodes': node_emb, 'edges': edge_emb, 'text': src_emb})
        elif self.fusion_type == 'perceiver':
            fusion_emb = self.fusion(torch.cat([node_emb, edge_emb, src_emb], dim=1))
            
        fused_features = fusion_emb[:, :node_emb.shape[1]]
        
        # Autoregressive text generation
        generated_text_ids = []
        for _ in range(max_length):
            trg_emb = self.pos_enc(self.text_emb(trg_ids))
            text_output = self.text_transformer_decoder(trg_emb, fused_features)
            
            # Extract the predicted token (typically, the token with the highest probability)
            next_token_id = text_output.argmax(dim=2)[:, -1].unsqueeze(1)
            trg_ids = torch.cat([trg_ids, next_token_id], dim=1)
            
            generated_text_ids.append(next_token_id)

        return torch.cat(generated_text_ids, dim=1)
