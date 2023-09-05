import os
import subprocess
from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

from src.data.components.wikigraphs.dataset import WikigraphDataset
from src.data.components.wikigraphs.preprocessing import build_wikitext_vocab, build_graph_vocab, build_text_vocab, pair_graphs_with_wikitext

class Wikidata5mDataModule(LightningDataModule):
    """`LightningDataModule` for the WikiGraphs dataset.

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
        data_dir: str = "data/",
        version: str = "max256",
        train_val_test_split: Tuple[int, int, int] = (55_000, 5_000, 10_000),
        batch_size: int = 64,
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

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None


    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        #TODO implement checks if all these files are already present
        BASE_DIR = self.hparams.data_dir

        # wikitext-103
        TARGET_DIR = os.path.join(BASE_DIR, 'wikitext-103')
        os.makedirs(TARGET_DIR, exist_ok=True)
        subprocess.run(['wget', 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip', '-P', TARGET_DIR])
        subprocess.run(['unzip', os.path.join(TARGET_DIR, 'wikitext-103-v1.zip'), '-d', TARGET_DIR])
        subprocess.run(['mv', os.path.join(TARGET_DIR, 'wikitext-103', '*'), TARGET_DIR])
        # Remove the extracted files and zip
        subprocess.run(['rm', '-rf', os.path.join(TARGET_DIR, 'wikitext-103'), os.path.join(TARGET_DIR, 'wikitext-103-v1.zip')])

        # wikitext-103-raw
        TARGET_DIR = os.path.join(BASE_DIR, 'wikitext-103-raw')
        os.makedirs(TARGET_DIR, exist_ok=True)
        subprocess.run(['wget', 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip', '-P', TARGET_DIR])
        subprocess.run(['unzip', os.path.join(TARGET_DIR, 'wikitext-103-raw-v1.zip'), '-d', TARGET_DIR])
        subprocess.run(['mv', os.path.join(TARGET_DIR, 'wikitext-103-raw', '*'), TARGET_DIR])
        # Remove the extracted files and zip
        subprocess.run(['rm', '-rf', os.path.join(TARGET_DIR, 'wikitext-103-raw'), os.path.join(TARGET_DIR, 'wikitext-103-raw-v1.zip')])

        # processed freebase graphs
        FREEBASE_TARGET_DIR = self.hparams.data_dir
        os.makedirs(os.path.join(FREEBASE_TARGET_DIR, 'packaged'), exist_ok=True)
        subprocess.run(['wget', '--no-check-certificate', 'https://docs.google.com/uc?export=download&id=1uuSS2o72dUCJrcLff6NBiLJuTgSU-uRo', '-O', os.path.join(FREEBASE_TARGET_DIR, 'packaged', 'max256.tar')])
        subprocess.run(['wget', '--no-check-certificate', 'https://docs.google.com/uc?export=download&id=1nOfUq3RUoPEWNZa2QHXl2q-1gA5F6kYh', '-O', os.path.join(FREEBASE_TARGET_DIR, 'packaged', 'max512.tar')])
        subprocess.run(['wget', '--no-check-certificate', 'https://docs.google.com/uc?export=download&id=1uuJwkocJXG1UcQ-RCH3JU96VsDvi7UD2', '-O', os.path.join(FREEBASE_TARGET_DIR, 'packaged', 'max1024.tar')])

        for version in ['max1024', 'max512', 'max256']:
            output_dir = os.path.join(FREEBASE_TARGET_DIR, 'freebase', version)
            os.makedirs(output_dir, exist_ok=True)
            subprocess.run(['tar', '-xvf', os.path.join(FREEBASE_TARGET_DIR, 'packaged', f'{version}.tar'), '-C', output_dir])

        # Remove the packaged files
        subprocess.run(['rm', '-rf', os.path.join(FREEBASE_TARGET_DIR, 'packaged')])
        
        # Build and store the vocab for WikiText-103
        build_wikitext_vocab(self.hparams.data_dir, self.hparams.vocab_file_path) #TODO ehat is vocab_file_path?
        
        # Pair articels with a Freebase subgraph
        freebase_dir = os.path.join(FREEBASE_TARGET_DIR, 'freebase', self.hparams.version)
        output_dir = os.path.join(BASE_DIR, 'wikigraphs', self.hparams.version)
        for subset in ['train', 'valid', 'test']:
            pair_graphs_with_wikitext(subset, freebase_dir, output_dir)
            
        # Build vocab for the graph data
        build_graph_vocab() #TODO add params
        
        # Build vocab for the text data
        build_text_vocab() #TODO add params

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train : WikigraphDataset = WikigraphDataset()
            self.data_val : WikigraphDataset = WikigraphDataset()
            self.data_test : WikigraphDataset = WikigraphDataset()

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
    _ = Wikidata5mDataModule()
