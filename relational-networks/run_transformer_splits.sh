#!/bin/bash

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
nb_heldout_colors=${11}

save_dir=$embed_dim-$num_layers-$functional-$share_vanilla_parameters-$use_topk-$topk-$shared_memory_attention-$mem_slots-$null_attention-$seed-$nb_heldout_colors

mkdir $save_dir

python main_splits.py --model Transformer --epochs 100 --embed_dim $embed_dim --num_layers $num_layers \
			   --functional $functional --share_vanilla_parameters $share_vanilla_parameters \
			   --use_topk $use_topk --topk $topk --shared_memory_attention $shared_memory_attention \
			   --save_dir $save_dir --mem_slots $mem_slots --null_attention $null_attention \
			   --seed $seed --nb_heldout_colors $nb_heldout_colors



