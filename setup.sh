#!bin/bash

echo "Please make sure that you have installed Anaconda"

echo "create conda environment 'selfkg'"
conda create -n selfkg

echo "activate selfkg"
conda activate selfkg

echo "install pytorch"
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch # change according to your need here

echo "install faiss-cpu"
conda install faiss-cpu=1.7.1 -c pytorch

echo "install numpy"
conda install numpy=1.21

echo "install pandas"
conda install pandas=1.2.5

echo "install tqdm"
conda install tqdm=4.61.1

echo "install transformers"
conda install transformers=4.8.2

echo "Environment setup"
