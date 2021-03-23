#!/bin/bash
#source ~/.bashrc


embed_dim=$1
num_layers=$2
share_vanilla_parameters=$3
use_topk=$4
topk=$5
shared_memory_attention=$6
mem_slots=$7
null_attention=False
seed=${8}
set_transformer=$9



save_dir=$embed_dim-$num_layers-$set_transformer-$share_vanilla_parameters-$use_topk-$topk-$shared_memory_attention-$mem_slots-$null_attention-$seed

mkdir $save_dir

python main.py --model Transformer --epochs 100 --embed_dim $embed_dim --num_layers $num_layers \
			   --functional $set_transformer --share_vanilla_parameters $share_vanilla_parameters \
			   --use_topk $use_topk --topk $topk --shared_memory_attention $shared_memory_attention \
			   --save_dir $save_dir --mem_slots $mem_slots --null_attention $null_attention \
			   --seed $seed



