import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, AutoModel

import src.data.components.wikidata5m.download as download
from src.data.components.wikidata5m import preprocessing as preprocess
from src.data.components.entity_linker import EntityLinker
from src.data.components.graph_processor import GraphProcessor
from src.data.components.wikidata5m.dataset import Wikidata5MDataset



class Wikidata5mDataModule(LightningDataModule): # https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.lightning.LightningDataset.html
    """`LightningDataModule` for the Wikidata5m dataset.

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "data/wikidata5m",
        use_entity_embeddings: bool = False,
        train_val_test_split: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        batch_size: int = 64,
        k_hop: int = 2,
        subgraph_retrieval: str = 'static', # or 'dynamic'
        min_doc_len: int = 256,
        max_nodes: int = 256,
        timestep: int = 128,
        mentions_threshold: int = 5,
        degree_threshold: int = 8,
        eigenvector_threshold: float = 0.5,
        node_embedding_dim: int = 128,
        max_graph_size_after_pruning: int = 524288,
        chunk_documents: bool = True,
        split_into_src_tgt: bool = True,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        """Initialize a `MNISTDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_val_test_split: The train, validation and test split. Defaults to `(55_000, 5_000, 10_000)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        
        self.corpus_version = f'_min-{self.hparams.min_doc_len}' if self.hparams.min_doc_len > 0 else ''

        self.data_train: Optional[Wikidata5MDataset] = None
        self.data_val: Optional[Wikidata5MDataset] = None
        self.data_test: Optional[Wikidata5MDataset] = None
        
        if self.hparams.use_entity_embeddings:
            if self.hparams.node_embedding_dim == 128:
                node_embedding_model_name = "google/bert_uncased_L-2_H-128_A-2"
            elif self.hparams.node_embedding_dim == 256:
                node_embedding_model_name = "google/bert_uncased_L-4_H-256_A-4"
            elif self.hparams.node_embedding_dim == 512:
                node_embedding_model_name = "google/bert_uncased_L-8_H-512_A-8"
            else:
                raise ValueError(f"Invalid node embedding dimension: {self.hparams.node_embedding_dim}")
        
        self.node_tokenizer = AutoTokenizer.from_pretrained(node_embedding_model_name)
        self.node_embedding_model = AutoModel.from_pretrained(node_embedding_model_name)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.entity_linker: EntityLinker = EntityLinker(self.hparams.data_dir, self.corpus_version, self.hparams.use_entity_embeddings, embedding_dim=self.hparams.node_embedding_dim)
        self.graph_processor: GraphProcessor = GraphProcessor(self.hparams.data_dir)
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")


    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        download.download_wikidata_dataset(self.hparams.data_dir)
        
        preprocess.prune_corpus(self.hparams.data_dir, self.tokenizer, self.hparams.min_doc_len)
        
        preprocess.whitelist_nq_entities(self.entity_linker, self.hparams.data_dir)
        
        preprocess.embed_nodes(self.hparams.data_dir, self.node_tokenizer, self.node_embedding_model, self.device, self.hparams.node_embedding_dim)
        
        preprocess.build_faiss_index(self.hparams.data_dir, self.hparams.node_embedding_dim)
        
        self.entity_linker.link_entities_for_corpus(self.hparams.use_entity_embeddings)
        
        self.graph_processor.build_full_graph()
        
        self.graph_processor.prune_graph(self.hparams.mentions_threshold, self.hparams.degree_threshold, self.hparams.eigenvector_threshold, self.hparams.max_graph_size_after_pruning)
        
        if self.hparams.subgraph_retrieval == 'static':
            self.graph_processor.find_subgraphs_for_corpus()
        

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        
        dataset = Wikidata5MDataset(
            self.hparams.data_dir,
            self.corpus_version,
            self.hparams.subgraph_retrieval,
            self.graph_processor,
            self.hparams.k_hop,
            self.hparams.max_nodes, 
            self.hparams.timestep, 
            self.hparams.chunk_documents,
            self.hparams.split_into_src_tgt,
            self.tokenizer) 
                
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train, self.data_val, self.data_test = random_split(
                dataset, self.hparams.train_val_test_split
            )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=self.data_train.collate_fn
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=self.data_val.collate_fn
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=self.data_test.collate_fn
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    # _ = Wikidata5mDataModule()
    
    dm = Wikidata5mDataModule(use_entity_embeddings=True)
    dm.prepare_data()
