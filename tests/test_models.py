import pytest
import torch

from src.data.wikigraphs_datamodule import WikigraphsDataModule
from src.models.components.kg_augmented_lm import KGAugmentedLM
from src.models.lit_module import KGAugmentedLMLitModule

# pytest tests/test_models.py

@pytest.mark.slow
@pytest.mark.parametrize("latent_dim", [128, 256])
def test_wikigraphs_kg_lm(latent_dim: int) -> None:
    dm = WikigraphsDataModule(batch_size=4)
    dm.prepare_data()
    dm.setup('fit')
    batch = next(iter(dm.train_dataloader()))
    
    model = KGAugmentedLM(
        latent_dim=128, 
        num_unimodal_encoder_layers=2, 
        num_unimodal_encoder_heads=2, 
        fusion_model='bottleneck',
        num_fusion_layers=2, 
        num_fusion_heads=2,
        text_vocab_size=dm.text_vocab_size,
        graph_vocab_size=dm.graph_vocab_size, 
        num_edge_types=dm.num_edge_types,
        modality_encoding='sinusoidal',)
    outputs = model(**batch)
    

# @pytest.mark.slow
# @pytest.mark.parametrize("latent_dim", [128, 256])
# def test_wikigraphs_kg_lm_lit_module_pretraining(latent_dim: int) -> None:
#     dm = WikigraphsDataModule(batch_size=4)
#     dm.prepare_data()
#     dm.setup('fit')
#     batch = next(iter(dm.train_dataloader()))
    
#     net = KGAugmentedLM(
#         latent_dim=128, 
#         num_unimodal_encoder_layers=2, 
#         num_unimodal_encoder_heads=2, 
#         fusion_model='bottleneck',
#         num_fusion_layers=2, 
#         num_fusion_heads=2,
#         text_vocab_size=dm.text_vocab_size,
#         graph_vocab_size=dm.graph_vocab_size, 
#         num_edge_types=dm.num_edge_types,
#         modality_encoding='sinusoidal',)
    
#     optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    
#     model = KGAugmentedLMLitModule(net, optimizer, scheduler)
#     model.on_train_start()
#     outputs = model.training_step(**batch)