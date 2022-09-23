#!/bin/bash
python main.py --dataset=fmnist --model=fmnistCNN --num_clients=100 --num_edges=2 --frac=0.1 --num_local_update=1 --num_edge_aggregation=1 --epochs=5000 --batch_size=20 --percentage=0.0 --iid=1 --lr=0.05 --momentum=0.9 --gpu=3 --compressed=2 --prop=0.01 --model_set=2 --acc_target=0.88