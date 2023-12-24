import pytest
import torch

from src.data.wikigraphs_datamodule import WikigraphsDataModule
from src.models.components.kg_augmented_lm import KGAugmentedLM
from src.models.lit_module import KGAugmentedLMLitModule
from src.data.components.token_masking import mask_nodes_and_adj


def return_faux_batch(batch_size, timesteps, text_vocab_size, graph_vocab_size, num_edge_types, pad_value, max_length):
    """Return a fake batch with the right shapes and dimensions."""
    input_seq = torch.randint(
        low=0, high=text_vocab_size, size=(batch_size, timesteps))
    input_attention_masks = torch.ones(
        (batch_size, timesteps), dtype=torch.float32)
    target_seq = torch.randint(
        low=0, high=text_vocab_size, size=(batch_size, timesteps))
    target_attention_masks = torch.ones(
        (batch_size, timesteps), dtype=torch.float32)
    node_features = torch.randint(
        low=0, high=graph_vocab_size, size=(batch_size, timesteps))
    adj_mats = torch.randint(
        low=0, high=num_edge_types, size=(batch_size, timesteps, timesteps))
    masked_nodes, node_masks, masked_adj, adj_masks = mask_nodes_and_adj(
        node_features, adj_mats, pad_value)

    return dict(
        batch_size=batch_size,
        input_seq=input_seq,
        input_attention_masks=input_attention_masks,
        target_seq=target_seq,
        target_attention_masks=target_attention_masks, 
        node_features=node_features, 
        masked_nodes_features=masked_nodes,
        node_masks=node_masks,
        adj_mats=adj_mats,
        masked_adj_mats=masked_adj,
        adj_masks=adj_masks,
        timesteps=timesteps,
        graph_vocab_size=graph_vocab_size,
        text_vocab_size=text_vocab_size,
        max_length=max_length)

# pytest tests/test_models.py


@pytest.mark.parametrize("latent_dim, fusion_model", [(128, 'bottleneck'),( 128, 'perceiver'), (256, 'bottleneck'), (256, 'perceiver')])
def test_kg_lm_fast(latent_dim: int, fusion_model: str) -> None:
    input = return_faux_batch(32, 64, 32768, 8192, 256, 0, 98)
    model = KGAugmentedLM(latent_dim=latent_dim, fusion_model=fusion_model)
    
    node_logits, text_logits, text_target = model(
        src_ids=input['input_seq'], 
        src_attention_mask=input['input_attention_masks'],
        trg_ids=input['target_seq'],
        trg_attention_mask=input['target_attention_masks'],
        node_ids=input['node_features'],
        adj_mat=input['adj_mats'],
        node_mask=input['node_masks'],
        mode='train')
    
    assert node_logits.shape == (input['batch_size'], input['timesteps'], input['graph_vocab_size'])
    assert text_logits.shape == (input['batch_size'], input['timesteps'] - 1, input['text_vocab_size'])
    assert text_target.shape == (input['batch_size'], input['timesteps'] - 1)
    
    generated_ids = model(
        src_ids=input['input_seq'],
        src_attention_mask=input['input_attention_masks'],
        node_ids=input['node_features'],
        adj_mat=input['adj_mats'], 
        node_mask=input['node_masks'],
        mode='predict',
        max_length=input['max_length'])
    
    assert generated_ids.shape == (input['batch_size'], input['max_length'])


@pytest.mark.slow
@pytest.mark.parametrize("latent_dim, fusion_model", [(128, 'bottleneck'),( 128, 'perceiver'), (256, 'bottleneck'), (256, 'perceiver')])
def test_wikigraphs_kg_lm(latent_dim: int) -> None:
    dm = WikigraphsDataModule(batch_size=32)
    dm.prepare_data()
    dm.setup('fit')
    
    model = KGAugmentedLM(
        latent_dim=latent_dim, 
        fusion_model='bottleneck', 
        text_vocab_size=dm.text_vocab_size,
        graph_vocab_size=dm.graph_vocab_size, 
        num_edge_types=dm.num_edge_types,
        )
    
    batch = next(iter(dm.train_dataloader()))
    
    node_logits, text_logits, text_target = model(**batch, mode='train')
    
    generated_ids = model(**batch, mode='predict', max_length=98)
    

@pytest.mark.slow
@pytest.mark.parametrize("latent_dim, fusion_model", [(128, 'bottleneck'),( 128, 'perceiver'), (256, 'bottleneck'), (256, 'perceiver')])
def test_wikigraphs_kg_lm_lit_module_pretraining(latent_dim: int) -> None:
    dm = WikigraphsDataModule(batch_size=32)
    dm.prepare_data()
    dm.setup('fit')
    
    model = KGAugmentedLM(
        latent_dim=latent_dim, 
        fusion_model='bottleneck', 
        text_vocab_size=dm.text_vocab_size,
        graph_vocab_size=dm.graph_vocab_size, 
        num_edge_types=dm.num_edge_types,
        )
    
    batch = next(iter(dm.train_dataloader()))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    
    lit_module = KGAugmentedLMLitModule(model, optimizer, scheduler)
    lit_module.on_train_start()
    
    losses = lit_module.training_step(**batch)
    
    assert len(losses) == 2