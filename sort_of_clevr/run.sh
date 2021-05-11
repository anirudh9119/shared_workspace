#!/bin/bash
#source ~/.bashrc
#source ~/anirudh/anaconda3/etc/profile.d/conda.sh
#echo Running on $HOSTNAME
#conda activate eaitorch1

#python3 sort_of_clevr_generator.py

python3 main.py --model=Transformer      --epochs=200      --relation-type=binary --seed $1

#python3 main.py --model=RN      --epochs=20      --relation-type=ternary

#python3 main.py --model=CNN_MLP --epochs=100



