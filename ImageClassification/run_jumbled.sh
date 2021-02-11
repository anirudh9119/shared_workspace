#!/usr/bin/env bash
#export HOME=`getent passwd $USER | cut -d':' -f6`
#source ~/.bashrc
source ~/anirudh/anaconda3/etc/profile.d/conda.sh
echo Running on $HOSTNAME
conda activate eaitorch1

model=$1
data=$2
version=$3
num_layers=$4
num_templates=1
dropout=$5
epochs=$6
patch_size=$7
num_heads=$8
batch_size=$9
lr=${10}
h_dim=${11}
ffn_dim=${12}
num_gru_schemas=${13}
num_attention_schemas=${14}
schema_specific=${15}
num_eval_layers=${16}
share_vanilla_parameters=${17}
use_topk=${18}
topk=${19}
shared_memory_attention=${20}
seed=${21}
mem_slots=${22}

name="128key_LnNoDropLSTM-"$model"-data-"$data"-version-"$version"-num_layers-"$num_layers"-num_templates-"$num_layers"-dropout-"$dropout"-epochs-"$epochs"-patch_size-"$patch_size"-num_heads-"$num_heads"-batch_size-"$batch_size"-lr-"$lr-$h_dim-$ffn_dim-$num_gru_schemas-$num_attention_schemas-$schema_specific-$num_eval_layers-$share_vanilla_parameters-$use_topk-$topk-$shared_memory_attention-$seed-$mem_slots

echo $name

python run_jumbled.py --model $model --data $data --version $version --num_layers $num_layers --num_templates $num_templates --dropout $dropout --epochs $epochs --patch_size $patch_size --num_heads $num_heads --name $name --batch_size $batch_size --lr $lr \
				--h_dim $h_dim --ffn_dim $ffn_dim --num_gru_schemas $num_gru_schemas \
				--num_attention_schemas $num_attention_schemas --schema_specific $schema_specific \
				--num_eval_layers $num_eval_layers --share_vanilla_parameters $share_vanilla_parameters --use_topk $use_topk --topk $topk --shared_memory_attention $shared_memory_attention --seed $seed \
				--mem_slots $mem_slots

