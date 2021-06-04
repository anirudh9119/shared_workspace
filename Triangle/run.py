import os
import random
import logging
import argparse
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from transformer_utilities.set_transformer import SetTransformer
from transformers import TransformerEncoder #FunctionalVisionTransformer, ViT
from einops import rearrange, repeat
from dataset import TriangleDataset

def str2bool(v):
    """Method to map string to bool for argument parser"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='Image Classification Tasks')
parser = argparse.ArgumentParser(description='Image Classification Tasks')
parser.add_argument('--model', default="functional", type=str, choices=('default','functional') ,help='type of transformer to use')
parser.add_argument('--data', default="cifar10", type=str, choices=('cifar10','cifar100','pathfinder', 'MNIST', 'Triangle') ,help='data to train on')
parser.add_argument('--version', default=0, type=int, help='version for shared transformer-- 0 or 1')
parser.add_argument('--num_layers', default=12, type=int, help='num of layers')
parser.add_argument('--num_templates', default=12, type=int, help='num of templates for shared transformer')
parser.add_argument('--num_heads', default=4, type=int, help='num of heads in Multi Head attention layer')
parser.add_argument('--batch_size', default=64, type=int, help='batch_size to use')
parser.add_argument('--patch_size', default=4, type=int, help='patch_size for transformer')
parser.add_argument('--epochs', default=200, type=int, help='num of epochs to train')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--dropout', default=0.1, type=float, help='dropout')
parser.add_argument('--name', default="model", type=str, help='Model name for logs and checkpoint')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--h_dim', type = int, default = 256)
parser.add_argument('--ffn_dim', type = int, default = 512)
parser.add_argument('--num_gru_schemas', type = int, default = 1)
parser.add_argument('--num_attention_schemas', type = int, default  = 1)
parser.add_argument('--schema_specific', type = str2bool, default = False)
parser.add_argument('--num_eval_layers', type = int, default = 1)
parser.add_argument('--share_vanilla_parameters', type = str2bool, default = False)
parser.add_argument('--num_digits_for_mnist', type = int, default = 3)
parser.add_argument('--use_topk', type = str2bool, default = False)
parser.add_argument('--topk', type = int, default = 3)
parser.add_argument('--shared_memory_attention', type = str2bool, default = False)
parser.add_argument('--mem_slots', type = int, default = 4)
parser.add_argument('--null_attention', type = str2bool, default = False)
parser.add_argument('--seed', type = int, default = 0)
args = parser.parse_args()

MIN_NUM_PATCHES=16


# logging config

#if not os.path.isdir('logs'):
#    os.mkdir('logs')
    
#logging.basicConfig(filename='./logs/%s.log' % args.name,
#                        level=logging.DEBUG, format='%(asctime)s %(levelname)-10s %(message)s')

#logging.info("Using args: {}".format(args))

def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_everything(seed = args.seed)

image_size =0
num_classes=0

#logging.info("Loading data: {}".format(args.data))
if args.data =="cifar10":
    # settings from https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    num_classes =10
    image_size = 32
    channels=3
    
elif args.data =="cifar100":
    # settings from https://github.com/weiaicunzai/pytorch-cifar100/blob/master/conf/global_settings.py
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.5070, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5070, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761)),
    ])
    
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=2)
    
    num_classes = 100
    image_size = 32
    channels=3
    
elif args.data == "pathfinder":
    trainset = np.load('./data/train.npz')
    trainset = torch.utils.data.TensorDataset(torch.Tensor(trainset['x']).reshape(-1,1,32,32),torch.LongTensor(trainset['y']))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=2)
    testset = np.load('./data/test.npz')
    testset = torch.utils.data.TensorDataset(torch.Tensor(testset['x']).reshape(-1,1,32,32),torch.LongTensor(testset['y']))
    
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=2)
    num_classes=2
    image_size = 32
    channels=1
elif args.data == 'MNIST':
    train_dataset = CountingMNISTDataset(split = "train", path = "MNIST", dig_range = [1,3], num_examples = 10000)
    test_dataset = CountingMNISTDataset(split = "test", path = "MNIST", dig_range = [4,5], num_examples = 2000)
    
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size = args.batch_size, num_workers = 2, shuffle = False)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size = args.batch_size, num_workers = 2, shuffle = False)
    num_classes = 10
    image_size = 100
    channels = 1
elif args.data == 'Triangle':
    train_dataset = TriangleDataset(num_examples = 50000)
    test_dataset = TriangleDataset(num_examples = 10000)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size = args.batch_size, num_workers = 2, shuffle = False)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size = args.batch_size, num_workers = 2, shuffle = False)
    num_classes = 4
    image_size = 64
    channels = 1

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


if args.model == "functional":
    transformer = SetTransformer(args.h_dim, dim_hidden = args.h_dim, num_inds = args.mem_slots)
    #net = FunctionalVisionTransformer(
    #    image_size = image_size,
    #    patch_size = args.patch_size,
    #    num_classes = num_classes,
    #    dim = 1024,
    #    depth = args.num_layers,
    #    heads = args.num_heads,
    #    mlp_dim = 2048,
    #    dropout = args.dropout,
    #    emb_dropout = 0.1,
    #    num_templates = args.num_templates,
    #    version = args.version,
    #    channels=channels

    #    )
    
elif args.model == "default":
    transformer = TransformerEncoder(
                            args.h_dim,
                            args.ffn_dim,
                            num_layers = args.num_layers,
                            num_heads = args.num_heads,
                            dropout = args.dropout,
                            share_parameters = args.share_vanilla_parameters,
                            shared_memory_attention = args.shared_memory_attention,
                            use_topk = args.use_topk,
                            topk = args.topk,
                            mem_slots = args.mem_slots,
                            null_attention = args.null_attention,
                            num_steps = int((image_size*image_size) / (args.patch_size * args.patch_size) + 1) )
    #net = ViT(
    #    image_size = image_size,
    #    patch_size = args.patch_size,
    #    num_classes = num_classes,
    #    dim = 1024,
    #    depth = args.num_layers,
    #    heads = args.num_heads,
    #    mlp_dim = 2048,
    #    dropout = args.dropout,
    #    emb_dropout = 0.1,
    #    channels=channels

    #    )
#print(int((image_size*image_size) / (args.patch_size * args.patch_size)))
class model(nn.Module):
    def __init__(self,  net, image_size, patch_size, num_classes):
        super().__init__()
        #print(image_size)
        #print(patch_size)
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        assert num_patches > MIN_NUM_PATCHES, f'your number of patches ({num_patches}) is way too small for attention to be effective (at least 16). Try decreasing your patch size'

        self.net = net
        self.patch_size = patch_size
        #print(patch_dim)
        self.patch_to_embedding = nn.Linear(patch_dim, args.h_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, args.h_dim))

        self.mlp_head = nn.Linear(args.h_dim, num_classes)

    def forward(self, img, mask = None):
        p = self.patch_size
        #print(img.size())
        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)
        #print(x.size())
        #print(x.type())
        x = self.patch_to_embedding(x)

        b, n, _ = x.shape
        #print(x.shape)

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        #print(x.size())

        x = self.net(x)

        x = self.mlp_head(x[:,0])

        return x
#print('line 234')
net = model(transformer, image_size, args.patch_size, num_classes)


net = net.to(device)

if os.path.exists('./checkpoint/'+args.name+'_ckpt.pth'):
    args.resume =True

if False and args.resume:
    # Load checkpoint.
    #logging.info("==> Resuming from checkpoint..")
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/'+args.name+'_ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
try:
    rmc_params = sum(p.numel() for p in net.net.enc.self_attn.relational_memory.parameters() if p.requires_grad)
    print(rmc_params)
except:
    pass
#logging.info("Total number of parameters:{}".format(pytorch_total_params))
print("Total number of parameters:{}".format(pytorch_total_params))

if args.data == 'MNIST':
    pre_loss_fn = nn.Sigmoid()
else:
    pre_loss_fn = nn.Identity()

if args.data == "MNIST":
    criterion = nn.BCELoss()
else:
    criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

def mnist_acc(outputs, targets):
    outputs[outputs >= 0.5] = 1.
    outputs[outputs<0.5] = 0.

    #print(outputs)
    #print(targets)

    equality = torch.eq(outputs, targets)

    equality = equality.int()

    #print(equality)
    print('-----')
    equality  = equality.sum(1)
    equality[equality < num_classes] = 0
    equality[equality == num_classes] = 1

    correct = equality.sum().item()
    print(correct)

    return correct





def train(epoch):
    print('\nEpoch: %d' % epoch)
    #logging.info('Epoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        #print(inputs.shape)
        #print(targets)
        optimizer.zero_grad()
        outputs = net(inputs)
        outputs = pre_loss_fn(outputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        if args.data == "MNIST":
            correct += mnist_acc(outputs, targets)
        else:
            correct += predicted.eq(targets).sum().item()

        
        if batch_idx % 100 == 99:    # print every 100 mini-batches
            #net.net.enc.self_attn.relational_memory.print_log()
            print('[%d, %5d] loss: %.3f accuracy:%.3f' %
                  (epoch + 1, batch_idx + 1, train_loss / (batch_idx+1), 100.*correct/total))
            #logging.info('[%d, %5d] loss: %.3f accuracy:%.3f' %
            #     (epoch + 1, batch_idx + 1, train_loss / (batch_idx+1), 100.*correct/total))
            


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            outputs = pre_loss_fn(outputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            if args.data == "MNIST":
                correct += mnist_acc(outputs, targets)
            else:
                correct += predicted.eq(targets).sum().item()

    # Save checkpoint.
    acc = 100.*correct/total
    print("test_accuracy is %.3f after epochs %d"%(acc,epoch))
    #logging.info("test_accuracy is %.3f after epochs %d"%(acc,epoch))
    if acc > best_acc:
        print('Saving..')
        #logging.info("==> Saving...")
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/'+args.name+'_ckpt.pth')
        best_acc = acc

#logging.info("Starting Training...")
for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)
    scheduler.step()
