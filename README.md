<div align="center">

# Knowledge Graph-Text Fusion for enhanced Language Modeling

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)

</div>

## Description

What it does

## Installation

It is highly recommended to run on a UNIX system. If you are on Windows, you can use the [Windows Subsystem for Linux](https://docs.microsoft.com/en-us/windows/wsl/install-win10) to run the code.

### Clone Repository

```bash
git clone https://github.com/marcomoldovan/kg-augmented-lm
cd kg-augmented-lm
```

### Virtual Environment & Dependencies

#### Option 1: Poetry

```bash
# install dependencies
poetry install

# activate environment
source $(poetry env info --path)/bin/activate
```	

#### Option 2:PyEnv

```bash
# create environment with specified python version
pyenv virtualenv <python-version> .venv
pyenv activate .venv

# install requirements
pip install -r requirements.txt
```

#### Option 3: Conda

```bash
# create conda environment and install dependencies
conda env create -f environment.yaml -n .venv

# activate conda environment
conda activate .venv
```

#### Option 4: Native Python

```bash
# create environment with specified python version
python -m venv .venv
source .venv/bin/activate

# install requirements
pip install -r requirements.txt
```

## How to run

#### Download and Prepare Data

_Running the downloading and preprocessing scripts is optional. The respective DataModules handle all of this automatically. This is only necessary if you want to run specfic tests before training which need access to the data._
##### Wikigraphs

```bash
bash scripts/download_data/download_wikigraphs.sh
bash scripts/preprocess_data/preprocess_wikigraphs.sh
```

##### Wikidata5m

```bash
bash scripts/download_data/download_wikidata5m.sh
bash scripts/preprocess_data/preprocess_wikidata5m.sh
```


#### Run Tests

```bash
# run all tests
pytest

# run tests from specific file
pytest tests/test_train.py

# run all tests except the ones marked as slow
pytest -k "not slow"
```

#### Single Training Run

Train model with default configuration

```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu

# train on multiple GPUs
python src/train.py trainer=ddp
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 data.batch_size=64
```

#### Hyperparameter Search

To run a hyperparameter search with [Optuna](https://optuna.org/) you can use the following command

```bash
python train.py -m hparams_search=fashion_mnist_optuna experiment=example
```

Running a hyperparameter sweep with [Weights and Biases](https://wandb.ai/site) is also supported.

```bash
wandb sweep configs/hparams_search/fashion_mnist_wandb.yaml
wandb agent <sweep_id>
```

#### SLURM

```bash
bash scripts/slurm/slurm_train.sh
```

#### Docker

```bash
docker build -t kg-augmented-lm .
docker run --gpus all -it kg-augmented-lm
```

#### Inference

```bash
# load checkpoint
# accepted model input
# linking to a KG
```

## Results

Results and some graphs.

## Contributing

Contributions are very welcome. If you know how to make this code better, don't hesitate to open an issue or a pull request.

## License

This project is licensed under the terms of the MIT license. See [LICENSE](LICENSE) for additional details.

## Acknowledgements

- [Lightning-Hydra Template]

## References

- [Knowledge Graph-Text Fusion for enhanced Language Modeling](https://www.nature.com/articles/nature14539)

## Citation

```bibtex
@article{KGTextFusion,
  title={Knowledge Graph-Text Fusion for enhanced Language Modeling},
  author={Marco Moldovan},
  journal={arXiv preprint arXiv:1001.2234},
  year={2023}
}
```
