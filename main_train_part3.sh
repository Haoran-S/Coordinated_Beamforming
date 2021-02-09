#!/usr/bin/bash
TrainScript="--hidden_layers 512-512-512 --batch_size 5000 --n_iter 20 --lr 0.001 --mini_batch_size 100 --n_memories 2000 --data_file data/dataset_beamforming --file_ext _bf"

echo "*********************** TL ***********************"
python3 main.py $TrainScript --model single --part 0
python3 main.py $TrainScript --model single --part 1
python3 main.py $TrainScript --model single --part 2
python3 main.py $TrainScript --model single --part 3

echo "*********************** Reservoir ***********************"
python3 main.py $TrainScript --model reservoir_sampling --part 0
python3 main.py $TrainScript --model reservoir_sampling --part 1
python3 main.py $TrainScript --model reservoir_sampling --part 2
python3 main.py $TrainScript --model reservoir_sampling --part 3
 

echo "*********************** BiLevel (Proposed) ***********************"
python3 main.py $TrainScript --model composition --part 0
python3 main.py $TrainScript --model composition --part 1
python3 main.py $TrainScript --model composition --part 2
python3 main.py $TrainScript --model composition --part 3

echo "*********************** Joint ***********************"
python3 main.py $TrainScript --model single --mode joint  --part 0
python3 main.py $TrainScript --model single --mode joint  --part 1
python3 main.py $TrainScript --model single --mode joint  --part 2
python3 main.py $TrainScript --model single --mode joint  --part 3
