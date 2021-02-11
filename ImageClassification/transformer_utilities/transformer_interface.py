import torch
import torch.nn as nn
#from transformer import TransformerEncoder
import types
import math

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

from models.transformer_layer import TransformerEncoderLayer, TransformerEncoderLayerVanilla
from models.pos_enc import PositionEncoder
#from transformer_utilities.GroupLinearLayer import GroupLinearLayer
import math
class GroupLinearLayer(nn.Module):
 def __init__(self, din, dout, num_blocks, bias=True, a = None):
     super(GroupLinearLayer, self).__init__()
     self.nb = num_blocks
     #din = din // num_blocks
     #dout = dout // num_blocks
     self.dout = dout
     if a is None:
         a = 1. / math.sqrt(dout)
     self.weight = nn.Parameter(torch.FloatTensor(num_blocks,din,dout).uniform_(-a,a))
     self.bias = bias
     if bias is True:
         self.bias = nn.Parameter(torch.FloatTensor(num_blocks,dout).uniform_(-a,a))
         #self.bias = nn.Parameter(torch.zeros(dout*num_blocks))
     else:
         self.bias = None
 def forward(self,x):
     ts,bs,m = x.shape
     #x = x.reshape((ts*bs, self.nb, m//self.nb))
     x = x.permute(1,0,2)
     x = torch.bmm(x,self.weight)
     x = x.permute(1,0,2)
     if not self.bias is None:
         x = x + self.bias
     #x = x.reshape((ts, bs, self.dout*self.nb))
     return x



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

    def __init__(self, inp_dim, h_dim, inp_nb, nb, functional = True):
        super().__init__()

        args.encoder_embed_dim = h_dim

        print('transformer h_dim', h_dim)

        

        args.encoder_embed_dim = h_dim
        self.functional = functional
        print('functional? '+str(self.functional))
        if not self.functional:
            layer_lst = []

            args.encoder_embed_dim = h_dim
            #layer_lst.append(TransformerEncoderLayer(args=args, nb=inp_nb, blockatt=False, blockatt_memory=True, use_nfm=False, out_proj_dim=h_dim))
            #for j in range(0,6):
            #    layer_lst.append(TransformerEncoderLayer(args=args, nb=nb, blockatt=False, blockatt_memory=True, use_nfm=False))
            self.enc = TransformerEncoderLayerVanilla(args)
            #self.layers = nn.ModuleList(layer_lst)
        else:
            #args.encoder_embed_dim = inp_dim
            #print('init_layer initialize')
            #self.init_layer = TransformerEncoderLayerVanilla(args=args, out_proj=h_dim)
            args.encoder_embed_dim = h_dim
            hidden_dim = args.encoder_embed_dim
            print('inp_att initialize')
            self.inp_att =  TransformerEncoderLayerVanilla(args=args)
            print('gru initialize')
            self.gru_pool = nn.ModuleList([nn.GRUCell(hidden_dim, hidden_dim) for _ in range(1)])
            self.state_att = TransformerEncoderLayerVanilla(args=args)
            self.select_attention = SelectAttention( hidden_dim + hidden_dim, hidden_dim, num_read = 1, num_write = 1)

        self.pe = PositionEncoder(inp_dim)
        self.pe_state = PositionEncoder(args.encoder_embed_dim)

    def forward(self, x, mask = None):

        x = x.permute(1, 0, 2)

        x = self.pe(x)
        if not self.functional:
            """klst = []
            vlst = []

            initial_state = self.layers[0].memory_layer.initial_state(batch_size=x.shape[0]*x.shape[1]).type(x.dtype).to(x.device)
            memory_obj = [initial_state]

            for layer in self.layers:
                layer.klst = klst
                layer.vlst = vlst
                layer.memory_obj = memory_obj

            """
            for i in range(6):
                x = self.enc(x, None)
            return x.permute(1, 0, 2)
        else:
            """
            klst = []
            vlst = []

            initial_state = self.init_layer.memory_layer.initial_state(batch_size=x.shape[0]*x.shape[1]).type(x.dtype).to(x.device)
            memory_obj = [initial_state]

            self.init_layer.klst = klst
            self.init_layer.vlst = vlst
            self.init_layer.memory_obj = memory_obj

            
            self.inp_att.klst = klst
            self.inp_att.vlst = vlst
            self.inp_att.memory_obj = memory_obj

            self.state_att.klst = klst
            self.state_att.vlst = vlst
            self.state_att.memory_obj = memory_obj
            """
            T, B, D = x.size()

            #x = self.init_layer(x, None)
            state = self.pe_state(torch.randn(x.size()).to(x.device))

            

            for i in range(0, 6):
                gru_in = self.inp_att(x, mask, state = state)
                gru_in = gru_in.permute(1, 0, 2)
                state = state.permute(1, 0, 2)

                gru_in = gru_in.reshape(B * T, -1)
                state = state.reshape(B * T, -1)

                gru_outs = []

                for gru in self.gru_pool:
                   gru_outs.append(gru(gru_in, state))

                gru_outs = torch.stack(gru_outs, dim = 1)

                selector = torch.cat((gru_in, state), dim = 1).unsqueeze(1)
                
                attn_scores = self.select_attention(selector, gru_outs)

                attn_scores = attn_scores.squeeze(1)

                attn_scores = torch.nn.functional.gumbel_softmax(attn_scores, dim = 1, tau = 1.0, hard = True)
                attn_scores = attn_scores.unsqueeze(-1)
                gru_outs = (gru_outs * attn_scores).sum(dim = 1)
                gru_outs_hidden = gru_outs.reshape(B, T, -1)
                gru_outs_hidden = gru_outs_hidden.permute(1, 0, 2)
                gru_outs_hidden = self.state_att(gru_outs_hidden, mask)
                gru_in = gru_in.reshape(B, T, -1).permute(1, 0, 2)

                x = gru_in
                state = gru_outs_hidden

            return state.permute(1,0,2)



if __name__ == "__main__":
    x = torch.randn(32, 64, 512)

    TE = TransformerEncoder()

    y = TE(x)

    print(y.shape)

