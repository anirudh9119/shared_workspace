# Coordination Among Neural Modules Through a Shared Global Workspace

This repository contains the code to reproduce the `relational reasoning: sort_of_clever` and `detecting equilateral triangles` tasks from our paper.  


## Install relevant libraries
```
pip install -r requirements.txt 
```
## Detecting Equilateral Triangles 
Folder: Triangle/

The following commands to be executed from inside in the `Triangle` folder.

```
sh run.sh num_layers h_dim ffn_dim share_vanilla_parameters use_topk topk shared_memory_attention seed mem_slots

share_vanilla_parameters: Whether share parameters across layers. If False, it will run TR + HC. For shared workspace experiments it should be True.

use_topk: Whether to use top-k competition

topk: Value of k in top-k competition

shared_memory_attention: Whether to use shared workspace

mem_slots: Number of slots in memory
```

To reproduce experiments in paper:
```
TR + HSW
sh run.sh 4 256 512 True True 20 True 1 8 default

TR + SSW
sh run.sh 4 256 512 True False 20 True 1 8 default

TR 
sh run.sh 4 256 512 True False 20 False 1 8 default

STR 
sh run.sh 4 256 512 True True 20 False 1 8 default

TR + HC
sh run.sh 4 256 512 False False 20 False 1 8 default

ISAB
sh run.sh 4 256 512 False False 20 False 1 8 functional
```

## Sort-of-CLEVR
The following commands to be executed from inside in the `sort_of_clevr` folder.

Dataset generation:
```
python sort_of_clevr_generator.py
```

```
sh run_transformer.sh h_dim num_layers share_vanilla_parameters use_topk topk shared_memory_attention mem_slots seed
```
To reproduce experiments in paper:
```
TR + HSW
sh run_transformer.sh 256 4 True True 5 True 8 1 False

TR
sh run_transformer.sh 256 4 True False 5 False 8 1 False

STR
sh run_transformer.sh 256 4 True True 5 False 8 1 False

TR + HC
sh run_transformer.sh 256 4 False False 5 False 8 1 False

ISAB
sh run_transformer.sh 256 4 False False 5 False 8 1 True


```

