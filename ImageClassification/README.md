### Running Instructions for Functional and Default Transformers 

The command for running classification experiments is 
```
sh run.sh model data version num_layers num_templates dropout epochs patch_size num_heads batch_size learning_rate
```
The model argument can be `functional` for functional transformers and `default` for defualt transformer. It supports `cifar10` and `cifar100` and `pathfinder` datasets right now. These are the current arguments supported. For running the pathfinder task, unzip the pathfinder zip file present in data folder.

```
  --model {default,functional}
                        type of transformer to use
  --data {cifar10,cifar100,pathfinder}
                        data to train on
  --version VERSION     version for shared transformer-- 0 or 1
  --num_layers NUM_LAYERS
                        num of layers
  --num_templates NUM_TEMPLATES
                        num of templates for shared transformer
  --num_heads NUM_HEADS
                        num of heads in Multi Head attention layer
  --batch_size BATCH_SIZE
                        batch_size to use
  --patch_size PATCH_SIZE
                        patch_size for transformer
  --epochs EPOCHS       num of epochs to train
  --lr LR               learning rate
  --dropout DROPOUT     dropout
  --name NAME           Model name for logs and checkpoint
  --resume, -r          resume from checkpoint
```
