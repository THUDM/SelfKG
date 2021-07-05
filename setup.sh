#!bin/bash

echo "Please make sure that you have installed Anaconda"

echo "create conda environment 'selfkg'"
conda create -n selfkg

echo "activate selfkg"
conda activate selfkg

echo "install pytorch"
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch # change according to your need here

echo "install faiss-cpu"
conda install faiss-cpu -c pytorch

echo "install numpy"
conda install numpy

echo "install pandas"
conda install pandas

echo "install tqdm"
conda install tqdm

echo "install transformers"
conda install transformers

echo "Environment setup"