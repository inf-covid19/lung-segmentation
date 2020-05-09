# lung-segmentation

## Requirements

- [Pipenv](https://github.com/pypa/pipenv)
- [PyTorch](https://pytorch.org/get-started/locally/)

## Installation

Install packages

```
pipenv install
```

Activate pipenv env

```
pipenv shell
```

## Setup GPU

Checking if GPU is available

```
python is_gpu_available.py
```

If GPU is not available, you should check `PyTorch` installation. It is highly recommended that you use GPU, it improves performance to more than 10x faster.

Command to activate PyTorch in Linux with CUDA 10.1. [Click here for more examples](https://pytorch.org/get-started/locally/).

**inside pipenv shell**
```
pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```

**outside pipenv shell**
```
pipenv run pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```

## Usage

Running script to generate images

```
python start_segmentation.py INPUT_FOLDER OUTPUT_FOLDER
```

Showing more options in help
```
python start_segmentation.py  --help
```
