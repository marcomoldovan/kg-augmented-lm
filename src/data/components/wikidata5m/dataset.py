# Roadmap for implementing wikidata5m for large(er) scale pretraining
#TODO 1. pair subgraphs with texts as a preprocessing step
#TODO 2. save subsets of data in separate folders, assign each instance of dataset to one folder
#TODO 3. create concat dataset
#TODO 4. write custom dataloader that exhausts one folder from concatdataset before loading next one into memory

# example of custom dataloader by ChatGPT:

import os
import torch
from torch.utils.data import DataLoader, ConcatDataset
from pytorch_lightning import LightningDataModule

class CustomDataModule(LightningDataModule):
    def __init__(self, data_root, batch_size=32, num_workers=4):
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        # Define your dataset transformations if needed
        transform = transforms.Compose([transforms.ToTensor()])

        # List all the directories containing your datasets
        dataset_dirs = [os.path.join(self.data_root, folder) for folder in os.listdir(self.data_root)]

        # Create a list of datasets
        self.datasets = [YourCustomDataset(dataset_dir, transform=transform) for dataset_dir in dataset_dirs]

        # Combine all datasets into one using ConcatDataset
        self.combined_dataset = ConcatDataset(self.datasets)

    def train_dataloader(self):
        return CustomDataLoader(
            self.combined_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

class CustomDataLoader(DataLoader):
    def __init__(self, dataset, *args, **kwargs):
        super().__init__(dataset, *args, **kwargs)

    def __iter__(self):
        # Initialize a DataLoader iterator for each dataset in order
        dataset_iters = [iter(dataset) for dataset in self.dataset.datasets]
        current_dataset_idx = 0

        while True:
            try:
                # Yield a batch from the current dataset
                yield next(dataset_iters[current_dataset_idx])
            except StopIteration:
                # When the current dataset is exhausted, move to the next dataset
                current_dataset_idx = (current_dataset_idx + 1) % len(dataset_iters)

# Usage
data_module = CustomDataModule(data_root='path/to/data')
model = YourLightningModule()

trainer = pl.Trainer()
trainer.fit(model, data_module)
