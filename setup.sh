#!bin/bash

echo "Please make sure that you have installed Anaconda"

echo "create conda environment 'selfkg'"
conda create -n selfkg

echo "activate selfkg"
conda activate selfkg

echo "install pytorch"
conda install -n selfkg pytorch=1.9.0 torchvision torchaudio cudatoolkit=10.2 -c pytorch # change according to your need here

echo "install faiss-cpu=1.7.1"
conda install -n selfkg faiss-cpu=1.7.1 -c pytorch

echo "install numpy=1.19.2 pandas=1.0.5 tqdm=4.61.1 transformers=4.8.2"
pip install numpy==1.19.2 pandas==1.0.5 tqdm==4.61.1 transformers==4.8.2 torchtext==0.10.0