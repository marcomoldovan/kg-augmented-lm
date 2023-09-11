from pathlib import Path

import pytest
import torch

from src.data.wikigraphs_datamodule import WikigraphsDataModule
from src.data.wikidata5m_datamodule import Wikidata5mDataModule

# pytest tests/test_datamodules.py

@pytest.mark.parametrize("batch_size", [1, 32, 128])
def test_wikigraphs_datamodule(batch_size: int) -> None:
    """Tests `WikigraphsDataModule` to verify that it can be downloaded correctly, that the necessary
    attributes were created (e.g., the dataloader objects), and that dtypes and batch sizes
    correctly match.

    :param batch_size: Batch size of the data to be loaded by the dataloader.
    """
    data_dir = 'data/wikigraphs'
    
    dm = WikigraphsDataModule(batch_size=batch_size)
    
    print('yes')
    
    dm.prepare_data()
    # No state assigned in prepare_data()
    assert not dm.data_train and not dm.data_val and not dm.data_test
    # Downloads
    assert Path(data_dir, 'wikitext-103-raw', 'wiki.train.raw').exists()
    assert Path(data_dir, 'wikitext-103', 'wiki.train.tokens').exists()
    assert Path(data_dir, 'freebase', 'max256', 'train.gz').exists()
    # Build wikitext vocab
    assert Path(data_dir, 'wikitext-vocab.csv').exists()
    # Pair graphs with wikitext
    assert Path(data_dir, 'wikigraphs', 'max256', 'train.gz').exists()
    # Build graph vocab
    assert Path(data_dir, 'graph-vocab.csv').exists()
    # Build text vocab
    assert Path(data_dir, 'text-vocab.csv').exists()

    dm.setup()
    assert dm.data_train and dm.data_val and dm.data_test
    
    # assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    # num_datapoints = len(dm.data_train) + len(dm.data_val) + len(dm.data_test)
    # assert num_datapoints == 23522

    # batch = next(iter(dm.train_dataloader()))
    # x, y = batch
    # assert len(x) == batch_size
    # assert len(y) == batch_size
    # assert x.dtype == torch.float32
    # assert y.dtype == torch.int64
    
    
# @pytest.mark.parametrize("batch_size", [32, 128])
# def test_wikidata5m_datamodule(batch_size: int) -> None:
#     """Tests `Wikidata5mDataModule` to verify that it can be downloaded correctly, that the necessary
#     attributes were created (e.g., the dataloader objects), and that dtypes and batch sizes
#     correctly match.

#     :param batch_size: Batch size of the data to be loaded by the dataloader.
#     """
#     data_dir = "data/"

#     dm = Wikidata5mDataModule(data_dir=data_dir, batch_size=batch_size)
#     dm.prepare_data()

#     assert not dm.data_train and not dm.data_val and not dm.data_test
#     assert Path(data_dir, "MNIST").exists()
#     assert Path(data_dir, "MNIST", "raw").exists()

#     dm.setup()
#     assert dm.data_train and dm.data_val and dm.data_test
#     assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

#     num_datapoints = len(dm.data_train) + len(dm.data_val) + len(dm.data_test)
#     assert num_datapoints == 70_000

#     batch = next(iter(dm.train_dataloader()))
#     x, y = batch
#     assert len(x) == batch_size
#     assert len(y) == batch_size
#     assert x.dtype == torch.float32
#     assert y.dtype == torch.int64
