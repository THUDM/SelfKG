echo "Get DBP15K start"
wget https://zenodo.org/record/6326870/files/DBP15K.zip?download=1 -O DBP15K.zip &
echo "Get DBP15K end"

echo "Get DWY100K start"
wget https://zenodo.org/record/6326870/files/DWY100K.zip?download=1 -O DWY100K.zip &
echo "Get DWY100K end"

echo "Get LaBSE start"
wget https://zenodo.org/record/6326870/files/LaBSE.zip?download=1 -O LaBSE.zip &
echo "Get LaBSE end"

wait

unzip DBP15K.zip &
unzip DWY100K.zip &
unzip LaBSE.zip &

wait
