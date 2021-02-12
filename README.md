# SW_ICML


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
sh run.sh 4 256 512 True True 20 True 1 8

TR + SSW
sh run.sh 4 256 512 True False 20 True 1 8

TR 
sh run.sh 4 256 512 True False 20 False 1 8

STR 
sh run.sh 4 256 512 True True 20 False 1 8

TR + HC
sh run.sh 4 256 512 False False 20 False 1 8
```

## Sort-of-CLEVR

```
sh run_transformer.sh h_dim num_layers share_vanilla_parameters use_topk topk shared_memory_attention mem_slots seed
```
To reproduce experiments in paper:
```
TR + HSW
sh run_transformer.sh 256 4 True True 5 True 8 1

TR
sh run_transformer.sh 256 4 True False 5 False 8 1

STR
sh run_transformer.sh 256 4 True True 5 False 8 1

TR + HC
sh run_transformer.sh 256 4 False False 5 False 8 1


```

