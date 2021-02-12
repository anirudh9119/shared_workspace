#!/bin/bash
#source ~/.bashrc
source ~/anirudh/anaconda3/etc/profile.d/conda.sh
echo Running on $HOSTNAME
conda activate eaitorch1

python sort_of_clevr_generator.py

python main.py --model=RN      --epochs=20      --relation-type=binary

python main.py --model=RN      --epochs=20      --relation-type=ternary

python main.py --model=CNN_MLP --epochs=100
