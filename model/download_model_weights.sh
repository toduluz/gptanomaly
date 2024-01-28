#!/bin/bash

# URL of the model weights
url="https://huggingface.co/roberta-base/resolve/main/pytorch_model.bin?download=true"

# Destination file
dest=./pytorch_model.bin

# Download the file
wget -O $dest $url