This is a repo covering the following paper.

[**Generalized Many-Way Few-Shot Video Classification**<br/>](https://arxiv.org/pdf/2007.04755.pdf)
Yongqin Xian, Bruno Korbar, Matthijs Douze, Bernt Schiele, Zeynep Akata, 
Lorenzo Torresani <br/>
ECCV IPCV Workshop 2020  

### Environment Setup
The code was built with PyTorch 1.4.0 and torchvision 0.5.0. 

### Data
Please follow [this repository](https://github.com/kenshohara/3D-ResNets-PyTorch) to prepare the training and test data. 

### Pre-trained models
The pretrained models can be downloaded with [this link](http://datasets.d2.mpi-inf.mpg.de/xian/fsv_pretrained_model.zip). It includes the R(2+1)D model pretrained on Sports1M, the R(2+1)D models (pretrained on Sports1M) finetuned on Kinetics100, UCF101 and SomethingV2 datasets respectively.  

### Evaluation
Please follow the sample_commands in the repo to reproduce the few-shot and generalized few-shot learning results.