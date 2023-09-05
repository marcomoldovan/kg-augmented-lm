import pytest
import torch

from src.models.components.graph_transformer import GraphTransformer
from src.models.components.multimodal_bottleneck_transformer import MBT
from src.models.components.kg_augmented_lm import KGAugmentedLM

def synthetic_input(batch_size: int, num_nodes: int, node_dim: int, num_edges: int, num_edge_types: int) -> torch.Tensor:
    input_ids = torch.randint(low=0, high=100, size=(batch_size, num_nodes))
    nodes = torch.rand(batch_size, num_nodes, node_dim)
    edge_indices = torch.randint(low=0, high=num_nodes, size=(2, num_edges))
    edge_types = torch.randint(low=0, high=num_edge_types, size=(num_edges,))
    input_dict = {"input_ids": input_ids, "nodes": nodes, "edge_indices": edge_indices, "edge_types": edge_types}
    return input_dict

@pytest.mark.parametrize("latent_dim", [128, 256])
def test_mnist_graph_transformer(latent_dim: int) -> None:
    inputs = synthetic_input(batch_size=2, num_nodes=10, node_dim=latent_dim, num_edges=10, num_edge_types=2)
    model = GraphTransformer(dim=latent_dim, num_layers=2, num_edge_types=2, heads=2)
    output = model(**inputs)
    
@pytest.mark.parametrize("latent_dim", [128, 256])
def test_mnist_multimodal_bottleneck_transformer(latent_dim: int) -> None:
    inputs = synthetic_input(batch_size=2, num_nodes=10, node_dim=latent_dim, num_edges=10, num_edge_types=2)
    model = MBT(latent_dim=latent_dim, num_layers=2, num_heads=2, fusion_layer=1, use_bottleneck=True)
    output = model(**inputs)
    
@pytest.mark.parametrize("latent_dim", [128, 256])
def test_mnist_kg_lm(latent_dim: int) -> None:
    inputs = synthetic_input(batch_size=2, num_nodes=10, node_dim=latent_dim, num_edges=10, num_edge_types=2)
    model = KGAugmentedLM(latent_dim=latent_dim, num_unimodal_encoder_layers=2, num_unimodal_encoder_heads=2, num_mbt_layers=2, num_mbt_heads=2, mbt_fusion_layer=1, mbt_use_bottleneck=True, vocab_size=100, num_concepts=100, num_edge_types=2)
    outputs = model(**inputs)
    