import pytest
import torch

from src.models.components.kg_augmented_lm import KGAugmentedLM
from src.data.wikigraphs_datamodule import WikigraphsDataModule

# pytest tests/test_models.py

@pytest.mark.slow
@pytest.mark.parametrize("latent_dim", [128, 256])
def test_mnist_kg_lm(latent_dim: int) -> None:
    dm = WikigraphsDataModule(batch_size=4)
    dm.prepare_data()
    dm.setup('fit')
    batch = next(iter(dm.train_dataloader()))
    
    model = KGAugmentedLM(
        latent_dim=latent_dim, 
        num_unimodal_encoder_layers=2, 
        num_unimodal_encoder_heads=2, 
        num_mbt_layers=2, 
        num_mbt_heads=2, 
        mbt_fusion_layer=1, 
        mbt_use_bottleneck=True, 
        vocab_size=100, 
        num_concepts=100, 
        num_edge_types=2)
    outputs = model(**batch)
    