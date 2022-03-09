#!bin/bash

echo "Start Training"

echo "Train SeflKG with LaBSE embedding and neighbor information on DBP15K"
python run_LaBSE_neighbor.py --device "cuda:0"  --epoch 150 # you can change the command line parameters as you like

echo "Train SeflKG with LaBSE embedding and neighbor information on DWY100K"
python run_DWY_LaBSE_neighbor.py --device "cuda:0" --epoch 150 # you can change the command line parameters as you like
