# Deep Gradient Compression
Partial implementation of paper "DEEP GRADIENT COMPRESSION: REDUCING THE COMMUNICATION BANDWIDTH FOR DISTRIBUTED TRAINING"

## Installation
for installing required packages run
` pip3 install -r requirements.txt`

## Run project
`python main.py`

## Implementation
Current implementation consist of only
* large gradients selection and passing
* small gradients accumulation
* momentum corelation


## TODO
* masking
* gradient clipping
* learning rate correction

## References
[DEEP GRADIENT COMPRESSION:REDUCING THE COMMUNICATION BANDWIDTH FOR DISTRIBUTED TRAINING](https://arxiv.org/pdf/1712.01887.pdf)
[Pytorch tutorial on distributed training](https://pytorch.org/tutorials/intermediate/dist_tuto.html)
