#!bin/bash

echo "Start Training"

echo "Train SeflKG with LaBSE embedding and neighbor information on DBP15K"
python run_LaBSE_neighbor.py

echo "Train SeflKG with LaBSE embedding and neighbor information on DWY100K"
python run_DWY_LaBSE_neighbor.py

echo "Train SeflKG with only LaBSE embedding on DBP15K"
python run_LaBSE_SSL.py

echo "Train SeflKG with only LaBSE embedding on DWY100K"
python run_LaBSE_SSL_DWY.py

echo "Train SeflKG with LaBSE embedding and neighbor information on DBP15K adding noise"
python run_LaBSE_neighbor_noise.py

