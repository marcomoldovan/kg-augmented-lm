import torch
from torch import nn

from src.models.components.graph_transformer import GraphTransformer
from src.models.components.multimodal_bottleneck_transformer import MBT



class KGAugmentedLM(nn.Module):
    def __init__(self, latent_dim, num_unimodal_encoder_layers, num_unimodal_encoder_heads, num_mbt_layers, num_mbt_heads, mbt_fusion_layer, mbt_use_bottleneck, vocab_size, num_concepts, num_edge_types):
        super().__init__()
        
        # EMBEDDINGS:
        self.node_emb = nn.Embedding(num_concepts, latent_dim)
        self.text_emb = nn.Embedding(vocab_size, latent_dim)
        
        # ENCODER: knowledge graph (uni-modal)
        self.graph_transformer_encoder = GraphTransformer(dim=latent_dim, num_layers=num_unimodal_encoder_layers, num_edge_types=num_edge_types, heads=num_unimodal_encoder_heads)
        
        # ENCODER: language modeling (uni-modal)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=latent_dim, nhead=num_unimodal_encoder_heads)
        self.text_transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_unimodal_encoder_layers)
        
        # ENCODER: graph + text fusion (multi-modal)
        self.mbt = MBT(latent_dim=latent_dim, num_layers=num_mbt_layers, num_heads=num_mbt_heads, fusion_layer=mbt_fusion_layer, use_bottleneck=mbt_use_bottleneck)
        
        # DECODER: knowledge graph link prediction (uni-modal)
        
        # DECODER: text generation (uni-modal)
        
        # TODO: attention mask
        # TODO: node mask
        # TODO: head mask
        # TODO: modality encoding
        
    def compute_adjacency_matrices(nodes, edge_indices, edge_types, num_edge_types):
        """
        Compute adjacency matrices for a batch of subgraphs based on edge information.

        Args:
            nodes (torch.Tensor): Tensor containing node embeddings for the subgraphs.
                                Shape: (batch_size, num_nodes, node_dim)
            edge_indices (torch.Tensor): Tensor containing edge indices for each batch element.
                                        Shape: (2, num_edges)
            edge_types (torch.Tensor): Tensor containing edge types for each edge.
                                    Shape: (num_edges)

        Returns:
            torch.Tensor: A tensor containing adjacency matrices for each batch element
                        and each edge type. Shape: (batch_size, num_nodes, num_nodes)
        """
        batch_size, num_nodes, node_dim = nodes.shape
        num_edges = edge_indices.shape[1]
        
        # Create an empty tensor for adjacency matrices
        adjacency_matrices = torch.zeros(batch_size, num_nodes, num_nodes)
        
        # Iterate over each batch and compute adjacency matrices
        for batch_idx in range(batch_size):
            for edge_idx in range(num_edges):
                head_node_idx, tail_node_idx = edge_indices[0, edge_idx], edge_indices[1, edge_idx]
                edge_type = edge_types[edge_idx]
                
                # Set the edge type as an entry in the adjacency matrix
                adjacency_matrices[batch_idx, head_node_idx, tail_node_idx] = edge_type
        
        return adjacency_matrices
        
        
    def forward(self, text_ids, concept_ids, edge_indices, edge_types):
        
        adj_mat = self.compute_adjacency_matrices(concept_ids, edge_indices, edge_types)
        
        node_emb = self.node_emb(concept_ids)
        text_emb = self.text_emb(text_ids)
        
        node_emb = self.graph_transformer_encoder(node_emb, adj_mat)
        text_emb = self.text_transformer_encoder(text_emb)
        
        fusion_emb = self.mbt({'kg': node_emb, 'text': text_emb})
        
        augmented_node_emb = fusion_emb[:, :node_emb.shape[1]]
        augmented_text_emb = fusion_emb[:, node_emb.shape[1]:]
        
        link_pred = ...
        text_gen = ...
        
        return link_pred, text_gen