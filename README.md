# RxnIM
This is the offical code of following paper "Towards Large-scale Chemical Reaction Image Parsing via a Multimodal Large Language Model".

## Highlights
<p align="justify">
In this paper, we present RxnIM, a multimodal large language model for different reaction image data extraction tasks such as reaction extraction task, condition OCR and role identification task. We first formulate these tasks into different task instructions. The model then aligns the task instructions with features extracted from reaction images. An LLM-based decoder can further make predictions based on these instructions. For the reaction extraction task, our model can achieve over 84%-92% soft match F1 score on multiple test sets, which significantly outperforms the previous works. 
The experiments also show the outstanding condition OCR and role identification abilities.
  
[comment]: <> ()
![visualization](figure/reactionllm.jpg)
<div align="center">
Overall Architecture of our RxnIM.
</div> 

## Using the code
Please clone the following repositories:
```
git clone https://github.com/CYF2000127/RxnIM
```




## Experiments

### Requirement

1. First create and activate a [conda](https://numdifftools.readthedocs.io/en/stable/how-to/create_virtual_env_with_conda.html) environment with the following command in a Linux, Windows, or MacOS environment (Linux is the most recommended):
```
conda create -n rxnim python=3.10
conda activate rxnim

2. Then Install requirements:
```
pip install -r requirements.txt

### Data preparation
For training and inference, please download the following datasets to your own path.
#### Datasets
1. **Synthetic:**  [Pistachio](https://www.dropbox.com/s/mxvm5i8139y5cvk/pubchem.zip?dl=0)
2. **Realistic:**  [ACS](https://www.dropbox.com/s/3podz99nuwagudy/uspto_mol.zip?dl=0)

#### Data generation
Or use the codes in [`data_generation`](./data_generation) to generate any number of synthetic reaction images.
Note that you should download the original Pistachio dataset first and put it into the same file with the codes.


### Training
1. Change the dataset path and jasonl file path in [`DEFAULT_TRAIN_DATASET.py`](./config/_base_/dataset/DEFAULT_TRAIN_DATASET.py) for different training stages.
2. Change the parameters in [`shikra_fsdp.py`](config/_base_/train/shikra_fsdp.py) for different training stages according to the paper.
3. Run the following command:
```
sh train.sh
```
 

### Inference
Run the following command:
```
sh eval.sh
```

### Web Demo

Go to our [web demo](https://huggingface.co/spaces/CYF200127/RxnIM) to directly use our model!

 


### Acknowledgement
Our code is based on [Shikra](https://github.com/shikras/shikra) and [VisionLLM](https://github.com/OpenGVLab/VisionLLM), thanks their great jobs!
