import torch
import torch.nn as nn
#from transformer import TransformerEncoder
import types
import math
import numpy as np

args = types.SimpleNamespace()
args.use_module_communication = 'true'
args.encoder_embed_dim = 512
args.encoder_attention_heads = 8 #was 8
args.attention_dropout = 0.1
args.topk_ratio = 1.0
args.dropout = 0.2
args.encoder_normalize_before = True
args.encoder_ffn_embed_dim = 2048
args.use_nfm = 'false'
args.shared_memory_attention = False
args.self_attention = True
args.mem_slots = 4
args.use_topk = False
args.topk = 3
args.num_steps = 5

from transformer_utilities.transformer_layer import TransformerEncoderLayer, TransformerEncoderLayerVanilla
from transformer_utilities.pos_enc import PositionEncoder
from transformer_utilities.GroupLinearLayer import GroupLinearLayer
import math


class SelectAttention(nn.Module):
    """docstring for SelectAttention"""
    def __init__(self, d_read, d_write, d_k = 16, num_read = 5, num_write = 5, share_query = False, share_key = False):
        super(SelectAttention, self).__init__()
        if not share_key:
            self.gll_write = GroupLinearLayer(d_write,d_k, num_write)
        else:
            self.gll_write = nn.Linear(d_write, d_k)

        if not share_query:
            self.gll_read = GroupLinearLayer(d_read,d_k, num_read)
        else:
            self.gll_read = nn.Linear(d_read, d_k)

        self.temperature = math.sqrt(d_k)

    def forward(self, q, k):
        read = self.gll_read(q)
        write = self.gll_write(k)

        return torch.bmm(read, write.permute(0, 2, 1)) / self.temperature

class TransformerEncoder(nn.Module):

    def __init__(self,
                 embed_dim,
                 ffn_dim,
                 num_layers = 6,
                 num_heads = 1,
                 dropout = 0.1,
                 functional = False,
                 shared_memory_attention = False,
                 shared_memory_percentage = 0.1,
                 share_parameters = False,
                 mem_slots = 4,
                 num_attention_schemas = 3,
                 num_gru_schemas = 3,
                 schema_specific = False,
                 use_topk = False,
                 topk = 3,
                 num_steps = 5,
                 null_attention = False,
                 regressive = False):
        super().__init__()

        if schema_specific and (num_gru_schemas != num_attention_schemas):
            print('Cannot use schema specific as num_gru_schemas != num_attention_schemas, continuing without')
            self.schema_specific = False
        else:
            self.schema_specific = schema_specific

        args.mem_slots = mem_slots
        args.encoder_embed_dim = embed_dim
        args.encoder_ffn_embed_dim = ffn_dim
        args.encoder_attention_heads = num_heads
        args.dropout = dropout
        args.shared_memory_attention = shared_memory_attention
        args.num_steps = num_steps
        args.null_attention = null_attention
        args.regressive = regressive


        self.num_layers = num_layers
        self.shared_memory_attention = shared_memory_attention
        self.shared_memory_percentage = shared_memory_percentage

        print('transformer embed_dim', embed_dim)
        self.functional = functional
        print('functional? '+str(self.functional))
        if not self.functional:
            layer_lst = []
            args.use_topk = use_topk
            args.topk = topk


            args.encoder_embed_dim = embed_dim
            self.share_parameters = share_parameters
            if share_parameters:
                self.enc = TransformerEncoderLayerVanilla(args)
            else:
                layer_lst = []
                for i in range(self.num_layers):
                    layer_lst.append(TransformerEncoderLayerVanilla(args))
                    print('flmklsd')
                self.layers = nn.ModuleList(layer_lst)
        else:
            #args.encoder_embed_dim = inp_dim
            #print('init_layer initialize')
            #self.init_layer = TransformerEncoderLayerVanilla(args=args, out_proj=h_dim)
            print('NUM GRU SCHEMAS:' + str(num_gru_schemas))
            print('NUM Attention SCHEMAS:' + str(num_attention_schemas))
            print('SCHEMA SPECIFIC:' + str(self.schema_specific))
            args.use_topk = use_topk
            args.topk = topk
            print('inp_att initialize')
            self.num_gru_schemas = num_gru_schemas
            self.num_att_schemas = num_attention_schemas
            self.schema_stats = np.zeros(self.num_gru_schemas)
            args.self_attention = True
            self.inp_att =  nn.ModuleList([TransformerEncoderLayerVanilla(args=args) for _ in range(num_attention_schemas)])
            self.select_attention_inp_att = SelectAttention( args.encoder_embed_dim, args.encoder_embed_dim, num_read = 1, num_write = num_attention_schemas) 
            print('gru initialize')
            hidden_dim = args.encoder_embed_dim


            self.gru_pool = nn.ModuleList([nn.GRUCell(hidden_dim, hidden_dim) for _ in range(num_gru_schemas)])
            #args.self_attention = True
            #self.state_att = TransformerEncoderLayerVanilla(args=args)
            self.select_attention = SelectAttention( hidden_dim + hidden_dim, hidden_dim, num_read = 1, num_write = num_gru_schemas)

        self.pe = PositionEncoder(args.encoder_embed_dim)
        self.pe_state = PositionEncoder(args.encoder_embed_dim)

    def forward(self, x, mask = None, num_layers = None):

        x = x.permute(1, 0, 2)

        x = self.pe(x)



        if not self.functional:
            if self.shared_memory_attention:
                memory_size = int(self.shared_memory_percentage * x.size(0))

                memory = torch.randn(memory_size, 1, x.size(2)).repeat(1 ,x.size(1), 1).to(x.device)
            else:
                memory = None
            if self.shared_memory_attention:
                if self.share_parameters:
                    if self.enc.self_attn.memory is not None:
                        self.enc.self_attn.init_memory(x.size(1), x.size(0), x.device)#.memory = self.enc.self_attn.memory.detach()
                else:
                    for layer in self.layers:
                        if layer.self_attn.memory is not None:
                            layer.self_attn.init_memory(x.size(1), x.device)#.memory = layer.self_attn.memory.detach()

            
            for i in range(self.num_layers):
                if self.share_parameters:
                    x, memory = self.enc(x, mask, memory = memory)
                else:
                    x, memory = self.layers[i](x, mask, memory = memory)
            return x.permute(1, 0, 2)
        else:
        
            T, B, D = x.size()

            if num_layers is None:
                num_layers = self.num_layers

            
            #state = self.pe_state(torch.randn(x.size()).to(x.device))

            if self.shared_memory_attention:
                memory_size = int(self.shared_memory_percentage * x.size(0))
                memory_inp = torch.randn( memory_size, 1, x.size(2)).repeat(1, x.size(1), 1).to(x.device)
                memory_state = torch.randn(memory_size, 1, x.size(2)).repeat(1, x.size(1), 1).to(x.device)
            else:
                memory_inp = None
                memory_state = None

            if self.shared_memory_attention:
                for inp_att in self.inp_att:
                    if inp_att.self_attn.memory is not None:
                        inp_att.self_attn.init_memory(x.size(1), x.device)#memory = inp_att.self_attn.memory.detach()
            for i in range(0, num_layers):
                gru_ins = []
                for inp_att in self.inp_att:
                    gru_in, memory_inp = inp_att(x, mask, memory = memory_inp)
                    gru_ins.append(gru_in.permute(1, 0, 2))

                gru_ins = torch.stack(gru_ins, dim = 2)
                gru_ins = gru_ins.reshape(B * T, -1, D)


                x = x.permute(1, 0, 2)
                x = x.reshape(B * T, -1).unsqueeze(1)

                attn_scores_inp_att = self.select_attention_inp_att(x, gru_ins)

                attn_scores_inp_att = attn_scores_inp_att.squeeze(1)
                attn_scores_inp_att = torch.nn.functional.gumbel_softmax(attn_scores_inp_att, dim = 1, hard = True, tau = 0.5)

                attn_scores_inp_att = attn_scores_inp_att.unsqueeze(-1)

                gru_in = (gru_ins * attn_scores_inp_att).sum(dim = 1)

                gru_in = gru_in.reshape(B, T, -1)
                x = x.reshape(B, T, -1)

                gru_in = gru_in.reshape(B * T, -1)
                x = x.reshape(B * T, -1)

                gru_outs = []

                for gru in self.gru_pool:
                   gru_outs.append(gru(gru_in, x))

                gru_outs = torch.stack(gru_outs, dim = 1)

                selector = torch.cat((gru_in, x), dim = 1).unsqueeze(1)
                if not self.schema_specific:
                    attn_scores = self.select_attention(selector, gru_outs)


                    attn_scores = attn_scores.squeeze(1)

                    attn_scores = torch.nn.functional.gumbel_softmax(attn_scores, dim = 1, tau = 1.0, hard = True)
                    
                    att_argmax = torch.sum(attn_scores.clone().detach(), dim = 0).cpu().numpy()

                    self.schema_stats += att_argmax


                    attn_scores = attn_scores.unsqueeze(-1)
                else:
                    attn_scores = attn_scores_inp_att
                    att_argmax = torch.sum(attn_scores.squeeze(-1).clone().detach(), dim = 0).cpu().numpy()

                    self.schema_stats += att_argmax

                gru_outs = (gru_outs * attn_scores).sum(dim = 1)
                gru_outs_hidden = gru_outs.reshape(B, T, -1)
                gru_outs_hidden = gru_outs_hidden.permute(1, 0, 2)
                #gru_outs_hidden, memory_state = self.state_att(gru_outs_hidden, mask, memory = memory_state)
                #gru_in = gru_in.reshape(B, T, -1).permute(1, 0, 2)
                #x = gru_in
                x = gru_outs_hidden

            return x.permute(1,0,2)

    def print_schema_stats(self):
        total = np.sum(self.schema_stats)
        for k in range(self.schema_stats.shape[0]):
            print('schema ' + str(k) + ' used ' + str(self.schema_stats[k]) + ' out of ' + str(total) + ' times')


    def reset_schema_stats(self):
        self.schema_stats = np.zeros(self.num_gru_schemas)


if __name__ == "__main__":
    x = torch.randn(8, 20, 256).cuda()
    import time
    TE1 = TransformerEncoder(256, 512, num_layers = 1, functional = False, num_gru_schemas = 3, num_attention_schemas = 3, schema_specific = False, shared_memory_attention = True, mem_slots = 8, num_steps = 20).cuda()
    t1 = time.time()
    for i in range(5):
        
        x = TE1(x)
    print(time.time() - t1)


    x = torch.randn(8, 20, 256).cuda()
    import time
    TE1 = TransformerEncoder(256, 512, num_layers = 1, functional = False, num_gru_schemas = 3, num_attention_schemas = 3, schema_specific = False, shared_memory_attention = True, mem_slots = 8, num_steps = 20).cuda()
    t1 = time.time()
    for i in range(5):
        
        x = TE1(x)
    print(time.time() - t1)
    x = torch.randn(8, 20, 256).cuda()
    TE2 = TransformerEncoder(256, 512, num_layers = 1, functional = False, num_gru_schemas = 3, num_attention_schemas = 3, schema_specific = True, shared_memory_attention = False, mem_slots = 8, num_steps = 20).cuda()
    t1 = time.time()
    for i in range(5):    
        x = TE2(x)
    print(time.time() - t1)
