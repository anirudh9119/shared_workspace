#!/bin/bash
#source ~/.bashrc
source ~/anirudh/anaconda3/etc/profile.d/conda.sh
echo Running on $HOSTNAME
conda activate eaitorch1


embed_dim=$1
num_layers=$2
functional=$3
share_vanilla_parameters=$4
use_topk=$5
topk=$6
shared_memory_attention=$7
mem_slots=$8
null_attention=$9
seed=${10}

save_dir=$embed_dim-$num_layers-$functional-$share_vanilla_parameters-$use_topk-$topk-$shared_memory_attention-$mem_slots-$null_attention-$seed

mkdir $save_dir

python main.py --model Transformer --epochs 100 --embed_dim $embed_dim --num_layers $num_layers \
			   --functional $functional --share_vanilla_parameters $share_vanilla_parameters \
			   --use_topk $use_topk --topk $topk --shared_memory_attention $shared_memory_attention \
			   --save_dir $save_dir --mem_slots $mem_slots --null_attention $null_attention \
			   --seed $seed



