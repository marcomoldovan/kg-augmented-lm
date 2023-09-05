from typing import Any, Dict, Optional

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.data.components.wikigraphs.dataset import WikigraphDataset
from src.data.components.wikigraphs.tokenizers import WordTokenizer, GraphTokenizer
import src.data.components.wikigraphs.preprocessing as preprocessing
import src.data.components.wikigraphs.download as download


class WikigraphsDataModule(LightningDataModule):
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
        data_dir: str = "data/wikigraphs",
        text_vocab_threshold: int = 3,
        graph_vocab_threshold: int = 15,
        text_only: bool = False,
        batch_size: int = 1,
        timesteps: int = 128,
        subsample_rate: float = 1.0,
        version: str = "max256",
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
        
        print('Hi')

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        
        self.word_tokenizer: WordTokenizer = None
        self.graph_tokenizer: GraphTokenizer = None
        
        self.text_vocab_size: Optional[int] = None
        self.graph_vocab_size: Optional[int] = None


    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        # Download raw data if necessary
        download.download_wikigraphs(self.hparams.data_dir)
        
        # Build and store the vocab for WikiText-103
        preprocessing.build_wikitext_vocab(self.hparams.data_dir)
        
        # Pair articels with a Freebase subgraph
        preprocessing.pair_graphs_with_wikitext_across_splits(self.hparams.data_dir, self.hparams.version)
            
        # Build vocab for the graph data
        preprocessing.build_graph_vocab(self.hparams.data_dir, self.hparams.version, self.hparams.graph_vocab_threshold)
        
        # Build vocab for the text data
        preprocessing.build_text_vocab(self.hparams.data_dir, self.hparams.version, self.hparams.graph_vocab_threshold) 

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        self.word_tokenizer = WordTokenizer()
        self.graph_tokenizer = GraphTokenizer()
        
        self.text_vocab_size = self.word_tokenizer.vocab_size
        self.graph_vocab_size = self.graph_tokenizer.vocab_size
        
        kwargs = dict(
            tokenizer=self.word_tokenizer,
            graph_tokenizer=self.graph_tokenizer,
            data_dir=self.hparams.data_dir,
            text_only=self.hparams.text_only,
            batch_size=self.hparams.batch_size,
            timesteps=self.hparams.timesteps,
            subsample_rate=self.hparams.subsample_rate,
            version=self.hparams.version,
        )
        
        if stage in ['fit', 'validate', 'test']:
            if not self.data_train and not self.data_val and not self.data_test:
                self.data_train: WikigraphDataset = WikigraphDataset(**kwargs, subset='train')
                self.data_val: WikigraphDataset = WikigraphDataset(**kwargs, subset='valid')
                self.data_test: WikigraphDataset = WikigraphDataset(**kwargs, subset='test')
        else:
            raise ValueError("'predict' dataset not implemented.")

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
    _ = WikigraphsDataModule()
