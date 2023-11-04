import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from typing import Any, Dict, Optional, Tuple

from lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split


from src.data.components.wikidata5m.dataset import Wikidata5MDataset
import src.data.components.wikidata5m.download as download
from src.data.components.entity_linker import EntityLinker
from src.data.components.graph_processor import GraphProcessor



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
        max_nodes: int = 256,
        timestep: int = 128,
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

        self.data_train: Optional[Wikidata5MDataset] = None
        self.data_val: Optional[Wikidata5MDataset] = None
        self.data_test: Optional[Wikidata5MDataset] = None
        
        self.entity_linker: EntityLinker = None
        self.graph_processor: GraphProcessor = None
        self.tokenizer = None


    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        download.download_wikidata_dataset(self.hparams.data_dir)
        

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        self.entity_linker = EntityLinker(self.hparams.data_dir, self.hparams.use_entity_embeddings)
        
        self.graph_processor = GraphProcessor(self.hparams.data_dir)
        
        # dataset = Wikidata5MDataset(
        #     self.hparams.data_dir, 
        #     self.entity_linker, 
        #     self.graph_processor, 
        #     self.tokenizer, 
        #     self.hparams.k_hop, 
        #     self.hparams.max_nodes, 
        #     self.hparams.timestep, 
        #     self.hparams.chunk_documents,
        #     self.hparams.split_into_src_tgt)
                
        # # load and split datasets only if not loaded already
        # if not self.data_train and not self.data_val and not self.data_test:
        #     self.data_train, self.data_val, self.data_test = random_split(
        #         dataset, self.hparams.train_val_test_split
        #     )

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
    
    dm = Wikidata5mDataModule()
    dm.setup()

    from time import time
    
    start = time()
    entities = dm.entity_linker.link_entities("The Monumentum Ancyranum (Latin 'Monument of Ancyra') or Temple of Augustus and Rome in Ancyra is an Augusteum in Ankara (ancient Ancyra), Turkey. The text of the Res Gestae Divi Augusti ('Deeds of the Divine Augustus') is inscribed on its walls, and is the most complete copy of that text. The temple is adjacent to the Hadji Bairam Mosque in the Ulus quarter.")
    subgraph = dm.graph_processor.get_subgraph_for_entities(entities, 2, 256)
    
    print(f"Retrieving entities and subgraph took {time() - start} seconds.")
