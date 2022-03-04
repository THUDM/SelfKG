#!bin/bash

echo "Start Training"

echo "Train SeflKG with LaBSE embedding and neighbor information on DBP15K"
python run_LaBSE_neighbor.py

echo "Train SeflKG with LaBSE embedding and neighbor information on DWY100K"
python run_DWY_LaBSE_neighbor.py
