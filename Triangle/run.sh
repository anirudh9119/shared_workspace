#!/bin/bash

#run.sh default Triangle 0 4 0.1 100 4 4 8 0.0001 64 128 2 2 True 6 True False 10 False 4 False 3


data='Triangle'
version=0
num_layers=$1
num_templates=1
dropout=0.1
epochs=100
patch_size=4
num_heads=4
batch_size=64
lr=0.0001
h_dim=$2
ffn_dim=$3
num_gru_schemas=2
num_attention_schemas=2
schema_specific=True
num_eval_layers=6
share_vanilla_parameters=${4}
use_topk=${5}
topk=${6}
shared_memory_attention=${7}
null_attention=False 
seed=${8}
mem_slots=${9}
model=${10}



name="HXLN_LN_LSTM-"$model"-data-"$data"-version-"$version"-num_layers-"$num_layers"-num_templates-"$num_layers"-dropout-"$dropout"-epochs-"$epochs"-patch_size-"$patch_size"-num_heads-"$num_heads"-batch_size-"$batch_size"-lr-"$lr-$h_dim-$ffn_dim-$num_gru_schemas-$num_attention_schemas-$schema_specific-$num_eval_layers-$share_vanilla_parameters-$use_topk-$topk-$shared_memory_attention-$mem_slots-$null_attention-$seed

echo $name

python run.py --model $model --data $data --version $version --num_layers $num_layers --num_templates $num_templates --dropout $dropout --epochs $epochs --patch_size $patch_size --num_heads $num_heads --name $name --batch_size $batch_size --lr $lr \
				--h_dim $h_dim --ffn_dim $ffn_dim --num_gru_schemas $num_gru_schemas \
				--num_attention_schemas $num_attention_schemas --schema_specific $schema_specific \
				--num_eval_layers $num_eval_layers --share_vanilla_parameters $share_vanilla_parameters --use_topk $use_topk --topk $topk --shared_memory_attention $shared_memory_attention \
				--mem_slots $mem_slots --null_attention $null_attention \
				--seed $seed

#sh run_local.sh functional cifar10 1 12 3 0.1 200 4 4 128 0.0001
