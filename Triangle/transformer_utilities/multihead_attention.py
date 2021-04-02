# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Dict, Optional, Tuple

import time
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Parameter


#import models.fairseq_util
import transformer_utilities.fairseq_utils as utils
#from fairseq.incremental_decoding_utils import with_incremental_state
from .fairseq_dropout import FairseqDropout
from .attention_rim import MultiHeadAttention as MHAMemory
from .quant_noise import quant_noise

from .group_linear_layer import GroupLinearLayer
from .relational_memory_volatile import RelationalMemory
#from .relational_memory_lstm import RelationalMemory
from .relational_memory_regressive import RelationalMemory as RelationalMemoryRegressive
#from fairseq.modules.shared_group_linear_layer import SharedGroupLinearLayer as GroupLinearLayer

class SelectAttention(nn.Module):
    """docstring for SelectAttention"""
    def __init__(self, d_read, d_write, d_k = 16):
        super(SelectAttention, self).__init__()
        self.gll_write = nn.Linear(d_write, d_k)

        self.gll_read = nn.Linear(d_read, d_k)

        self.temperature = math.sqrt(d_k)

    def forward(self, q, k):
        read = self.gll_read(q)
        write = self.gll_write(k)

        return torch.bmm(read, write.permute(0, 2, 1)) / self.temperature


# head embedding dimension  = (1,n_h,h_emb)
# (1,n_h,d_k) and (tgt,bsz,d_k)
# (1,d_k,n_h)
class MultiheadAttention(nn.Module):
    """Multi-headed attention.
    See "Attention Is All You Need" for more details.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        self_attention=False,
        encoder_decoder_attention=False,
        q_noise=0.0,
        qn_block_size=8,
        nblocks=1,
        top_k_ratio=None,
        use_value_competition=True,
        shared_memory_attention = False,
        use_topk = False,
        topk = 3,
        num_steps = 5,
        mem_slots = 4,
        null_attention = False,
        regressive = False,
        selective = False,
        head_embed_dim=64
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.num_heads = num_heads
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )

        self.head_dim = embed_dim // num_heads
        self.shared_memory_attention = shared_memory_attention
        self.head_embed_dim = head_embed_dim
        self.selective = selective

        print('total heads', self.num_heads)
        print('head dim', self.head_dim)

        self.use_topk = use_topk
        self.topk = topk

        print('use topk?' + str(self.use_topk))
        print('topk:'+str(self.topk))

        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        assert not self.self_attention or self.qkv_same_dim, (
            "Self-attention requires query, key and " "value to be of the same size"
        )
        if self.selective:
            print('USING SELECTIVE ATTENTION')
            self.q_proj_selective = nn.Linear(self.embed_dim,self.embed_dim*self.num_heads)
            w =  torch.randn(1, self.num_heads, self.head_embed_dim).to(self.device)
            self.head_embeddings = nn.Parameter(w)
            self.variable_head_select = SelectAttention( self.embed_dim ,self.head_embed_dim, d_k=32)
            self.kv_selective = nn.Linear(self.embed_dim,self.num_heads*self.embed_dim*2)



            ''' we want something like this:
            query_weights = self.q_proj.weight.reshape(self.embed_dim,self.embed_dim//self.num_heads,self.num_heads).permute()'''
            
        if not self.selective and not self.shared_memory_attention:
            self.k_proj = quant_noise(GroupLinearLayer(self.kdim//nblocks, embed_dim//nblocks, nblocks, bias=bias), q_noise, qn_block_size)
            self.v_proj = quant_noise(GroupLinearLayer(self.vdim//nblocks, embed_dim//nblocks, nblocks, bias=bias), q_noise, qn_block_size)
            self.q_proj = quant_noise(GroupLinearLayer(embed_dim//nblocks, embed_dim//nblocks, nblocks, bias=bias), q_noise, qn_block_size)
            self.out_proj = quant_noise(GroupLinearLayer(embed_dim//nblocks, embed_dim//nblocks, nblocks, bias=bias), q_noise, qn_block_size)

            if add_bias_kv:
                self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
                self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
                if self.shared_memory_attention:
                    self.bias_k_memory = Parameter(torch.Tensor(1, 1, embed_dim))
                    self.bias_v_memory = Parameter(torch.Tensor(1, 1, embed_dim))
            else:
                self.bias_k = self.bias_v = None
                self.bias_k_memory = self.bias_v_memory = None

            self.add_zero_attn = add_zero_attn

            self.reset_parameters()

        self.onnx_trace = False
        self.tpu = False

        if not self.selective and self.shared_memory_attention:
            print('MEM SLOTS:' + str(mem_slots))
            print('Null attention:' + str(null_attention))
            print('USING SHARED MEMORY ATTENTION +++++++++')
            #self.num_heads = 1
            self.regressive = regressive
            if not regressive:            
                self.relational_memory = RelationalMemory(
                        mem_slots=mem_slots,
                        head_size=self.head_dim , #128
                        input_size=embed_dim,
                        output_size=embed_dim,
                        num_heads=self.num_heads, #1
                        num_blocks=1,
                        forget_bias=1,
                        input_bias=0,
                        gate_style="unit",
                        attention_mlp_layers=1,
                        key_size=32,
                        return_all_outputs=False,
                        use_topk = self.use_topk,
                        topk = self.topk,
                        num_steps = num_steps,
                        null_attention = null_attention
                    )
            else:
                print('USING AUTO REGRESSIVE')
                self.relational_memory = RelationalMemoryRegressive(
                        mem_slots=mem_slots,
                        head_size=self.head_dim ,
                        input_size=embed_dim,
                        output_size=embed_dim,
                        num_heads=self.num_heads,
                        num_blocks=1,
                        forget_bias=1,
                        input_bias=0,
                        gate_style="unit",
                        attention_mlp_layers=4,
                        key_size=32,
                        return_all_outputs=False,
                        use_topk = self.use_topk,
                        topk = self.topk,
                        num_steps = num_steps,
                        null_attention = False
                        )
            self.memory_size = 128 #self.head_dim * self.num_heads
            '''
            self.mem_att = MHAMemory(
                n_head=4,
                d_model_read=embed_dim,
                d_model_write=self.memory_size,
                d_model_out=embed_dim,
                d_k=32,
                d_v=32,
                grad_sparse=False,
            )
            '''
            self.memory = None


    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def prepare_for_tpu_(self, **kwargs):
        self.tpu = True

    def reset_parameters(self):
        if self.qkv_same_dim or True: #=====================================CHANGED TO ALWAYS TRUE FOR NOW=======================================================
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
            if self.shared_memory_attention:
                nn.init.xavier_uniform_(self.k_proj_memory.weight, gain=1 / math.sqrt(2))
                nn.init.xavier_uniform_(self.v_proj_memory.weight, gain=1 / math.sqrt(2))
                nn.init.xavier_uniform_(self.q_proj_memory.weight, gain=1 / math.sqrt(2))
            if self.selective:
                nn.init.xavier_uniform_(self.kv_selective.weight, gain=1 / math.sqrt(2))
                nn.init.xavier_uniform_(self.q_proj_selective.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

            #if self.shared_memory_attention:
            #    nn.init.xavier_uniform_(self.k_proj_memory.weight)
            #    nn.init.xavier_uniform_(self.v_proj_memory.weight)
            #    nn.init.xavier_uniform_(self.q_proj_memory.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        #if self.shared_memory_attention:
        #    nn.init.xavier_uniform_(self.out_proj_memory.weight)
        
        if  self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.)

        #if self.shared_memory_attention and self.out_proj_memory.bias is not None:
        #    nn.init.constant_(self.out_proj.bias, 0.)
        
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

        #if self.shared_memory_attention:
        #    if self.bias_k is not None:
        #        nn.init.xavier_normal_(self.bias_k_memory)
        #    if self.bias_v is not None:
        #        nn.init.xavier_normal_(self.bias_v_memory)


    def forward(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = True,
        static_kv: bool = False,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
        comp = None,
        memory = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel
        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True


        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]

        if (
            True
            # not self.onnx_trace
            # and not self.tpu  # don't use PyTorch version on TPUs
            # and incremental_state is None
            # and not static_kv
            # # A workaround for quantization to work. Otherwise JIT compilation
            # # treats bias in linear module as method.
            # and not torch.jit.is_scripting()
            # and False
        ):
            assert key is not None and value is not None
            if self.selective:
                pre_scores = self.variable_head_select(query.permute(1,0,2),self.head_embeddings.repeat(bsz,1,1))
                selected_indices = F.gumbel_softmax(pre_scores, dim = 2, hard = True).permute(1,0,2)
                #selected_indices become tgt_len,bsz,num_heads
                '''self.k_proj_selective = nn.Linear(self.embed_dim,self.embed_dim)
            self.v_proj_selective = nn.Linear(self.embed_dim,self.embed_dim)
            self.q_proj_selective = nn.Linear(self.embed_dim,self.embed_dim)'''
                # kv_weight = self.kv_selective.weight.reshape(self.embed_dim,2*self.embed_dim//self.num_heads,self.num_heads).permute(2,1,0)
                # kv_bias = self.kv_bias.bias.reshape(self.num_heads,2*self.embed_dim//self.num_heads)

                # selected_qkv_weight = torch.matmul(selected_indices.reshape(tgt_len*bsz,num_heads),selected_qkv_weight.reshape(self.num_heads,(self.embed_dim**2)*3//self.num_heads))
                # selected_qkv_bias = torch.matmul(selected_indices.reshape(tgt_len*bsz,num_heads),selected_qkv_bias.reshape(self.num_heads,self.embed_dim*3//self.num_heads))

                # selected_qkv_weight = selected_qkv_weight.reshape(tgt_len,bsz,self.embed_dim,3*self.embed_dim//self.num_heads)
                # selected_qkv_bias = selected_qkv_bias.reshape(tgt_len,bsz,3*self.embed_dim//self.num_heads)

                # q_proj,k_proj,v_proj = selected_qkv_weight.chunk(dim=3)
                # q_bias,k_bias,v_bias = selected_qkv_bias.chunk(dim=2)
                '''
                query_proj_weight  = self.q_proj_selective.weight.reshape(self.embed_dim,self.embed_dim,self.num_heads)
                query_proj_bias  = self.q_proj_selective.bias.reshape(self.num_heads,self.embed_dim)
                selected_indices_reshaped = selected_indices.reshape(tgt_len*bsz,self.num_heads)
               



                

                selected_q_weights = torch.matmul(selected_indices_reshaped,query_proj_weight.permute(2,0,1).reshape(self.num_heads,self.embed_dim**2))
                selected_q_weights = selected_q_weights.reshape(tgt_len*bsz,self.embed_dim,self.embed_dim)
                selected_q_bias = torch.matmul(selected_indices_reshaped,query_proj_bias)
                # selected_q_weights --> tgt_len*bsz,self.embed_dim,self.embed_dim
                # selected_q_bias --> tgt_len*bsz,self.embed_dim


                # queries = torch.bmm(query.reshape(tgt_len*bsz,self.embed_dim),selected_q_weights) + selected_q_bias
                queries = torch.matmul(query.reshape(tgt_len * bsz, self.embed_dim).unsqueeze(0), selected_q_weights).squeeze(0) + selected_q_bias
                queries = queries.reshape(tgt_len,bsz,self.embed_dim)'''

                all_queries = self.q_proj_selective(query)
                all_queries = all_queries.reshape(tgt_len,bsz,self.num_heads,self.embed_dim)
                selected_indices_new = selected_indices.unsqueeze(3).repeat(1,1,1,self.embed_dim)

                queries = (all_queries*selected_indices_new).sum(dim=2)
                    
                # queries = query
                queries = queries.permute(1,0,2)
                # queries--> batch_size, tgt_len, self.embed_dim

                key_value = self.kv_selective(query).reshape(tgt_len,bsz,2*self.embed_dim,self.num_heads)
                key_value = key_value.permute(1,3,0,2)
                # key_value--> batch_size, num_heads, tgt_len, 2*self.embed_dim

                '''
                selected_indices = selected_indices.permute(1,0,2).reshape(bsz,tgt_len,self.num_heads,1,1)
                selected_indices = selected_indices.repeat(1,1,1,tgt_len,2*self.embed_dim)
                key_value = key_value.unsqueeze(1).repeat(1,tgt_len,1,1,1)

                selected_key_value = (selected_indices*key_value).sum(dim=2)
                selected_key_value = selected_key_value.reshape(bsz*tgt_len,tgt_len,2*self.embed_dim)
                '''

                selected_indices = selected_indices.permute(1,0,2)
                # selected_indices--> batch_size, tgt_len, num_heads
                selected_key_value = torch.bmm(selected_indices,key_value.reshape(bsz,self.num_heads,2*tgt_len*self.embed_dim))
                # selected_key_value--> batch_size, tgt_len, 2*tgt_len*self.embed_dim
                selected_key_value = selected_key_value.reshape(bsz*tgt_len,tgt_len,2*self.embed_dim)
                queries = queries.reshape(bsz*tgt_len,self.embed_dim)


                scaling_factor = math.sqrt(self.embed_dim)
                selected_key, selected_value = selected_key_value.chunk(2,dim=2)
                #queries --> bsz*tgt_len , self.embed_dim
                #selected_key --> bsz*tgt_len, tgt_len, self.embed_dim
                #selected_value --> bsz*tgt_len, tgt_len, self.embed_dim

                attention_scores = F.softmax(torch.bmm(selected_key,queries.unsqueeze(2)).squeeze(2)/scaling_factor, dim=1)
                # bsz*tgt_len, tgt_len
                attention_scores = attention_scores.reshape(bsz*tgt_len, tgt_len,1)
                attended_summary = (attention_scores*selected_value).sum(dim=1)
                attended_summary = attended_summary.reshape(bsz,tgt_len,self.embed_dim)
                attended_summary = attended_summary.permute(1,0,2)
                # print(attended_summary)
                # print(attended_summary.shape)

                out=attended_summary
                memory = None
                weights = None






                





                # key_proj_weight = self.k_proj_selective.weight.reshape(self.embed_dim,self.embed_dim//self.num_heads,self.num_heads).permute(2,1,0)
                # key_proj_bias = self.k_proj_selective.bias.reshape(self.num_heads,self.embed_dim//self.num_heads)
                # value_proj_weight = self.v_proj_selective.weight.reshape(self.embed_dim,self.embed_dim//self.num_heads,self.num_heads).permute(2,1,0)
                # value_proj_bias = self.v_proj_selective.bias.reshape(self.num_heads,self.embed_dim//self.num_heads)
                # query_proj_weight = self.q_proj_selective.weight.reshape(self.embed_dim,self.embed_dim//self.num_heads,self.num_heads).permute(2,1,0)
                # query_proj_bias = self.q_proj_selective.bias.reshape(self.num_heads,self.embed_dim//self.num_heads)

                # selected_q_weights = torch.matmul(selected_indices.reshape(tgt_len*bsz,self.num_heads),query_proj_weight.reshape(self.num_heads,self.embed_dim**2//self.num_heads))
                # selected_k_weights = torch.matmul(selected_indices.reshape(tgt_len*bsz,self.num_heads),key_proj_weight.reshape(self.num_heads,self.embed_dim**2//self.num_heads))
                # selected_v_weights = torch.matmul(selected_indices.reshape(tgt_len*bsz,self.num_heads),value_proj_weight.reshape(self.num_heads,self.embed_dim**2//self.num_heads))
                # selected_q_bias = torch.matmul(selected_indices.reshape(tgt_len*bsz,self.num_heads),query_proj_bias.reshape(self.num_heads,self.embed_dim**2//self.num_heads))
                # selected_k_bias = torch.matmul(selected_indices.reshape(tgt_len*bsz,self.num_heads),key_proj_bias.reshape(self.num_heads,self.embed_dim**2//self.num_heads))
                # selected_v_bias = torch.matmul(selected_indices.reshape(tgt_len*bsz,self.num_heads),value_proj_bias.reshape(self.num_heads,self.embed_dim**2//self.num_heads))

                # selected_q_weights = selected_q_weights.reshape(tgt_len,bsz,self.embed_dim,self.embed_dim//self.num_heads)
                # selected_k_weights = selected_k_weights.reshape(tgt_len,bsz,self.embed_dim,self.embed_dim//self.num_heads)
                # selected_v_weights = selected_v_weights.reshape(tgt_len,bsz,self.embed_dim,self.embed_dim//self.num_heads)
                # selected_q_bias = selected_q_bias.reshape(tgt_len,bsz,self.embed_dim//self.num_heads)
                # selected_k_bias = selected_k_bias.reshape(tgt_len,bsz,self.embed_dim//self.num_heads)
                # selected_v_bias = selected_v_bias.reshape(tgt_len,bsz,self.embed_dim//self.num_heads)




                
            if not self.selective and self.shared_memory_attention:
                memory,_ = F.multi_head_attention_forward(
                    memory,
                    key,
                    value,
                    self.embed_dim,
                    self.num_heads,
                    torch.empty([0]),
                    torch.cat((self.q_proj_memory.bias, self.k_proj.bias, self.v_proj.bias)),
                    self.bias_k,
                    self.bias_v,
                    self.add_zero_attn,
                    self.dropout_module.p,
                    self.out_proj_memory.weight,
                    self.out_proj_memory.bias,
                    self.training or self.dropout_module.apply_during_inference,
                    key_padding_mask,
                    need_weights,
                    attn_mask,
                    use_separate_proj_weight=True,
                    q_proj_weight=self.q_proj_memory.weight,
                    k_proj_weight=self.k_proj.weight,
                    v_proj_weight=self.v_proj.weight,
                    )
                out,weights = F.multi_head_attention_forward(
                    query,
                    memory,
                    memory,
                    self.embed_dim,
                    self.num_heads,
                    torch.empty([0]),
                    torch.cat((self.q_proj.bias, self.k_proj_memory.bias, self.v_proj_memory.bias)),
                    self.bias_k_memory,
                    self.bias_v_memory,
                    self.add_zero_attn,
                    self.dropout_module.p,
                    self.out_proj.weight,
                    self.out_proj.bias,
                    self.training or self.dropout_module.apply_during_inference,
                    key_padding_mask,
                    need_weights,
                    attn_mask,
                    use_separate_proj_weight=True,
                    q_proj_weight=self.q_proj.weight,
                    k_proj_weight=self.k_proj_memory.weight,
                    v_proj_weight=self.v_proj_memory.weight,
                    )
            
            elif not self.selective and not self.shared_memory_attention:
                out, weights = F.multi_head_attention_forward(
                    query,
                    key,
                    value,
                    self.embed_dim,
                    self.num_heads,
                    torch.empty([0]),
                    torch.cat((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)),
                    self.bias_k,
                    self.bias_v,
                    self.add_zero_attn,
                    self.dropout_module.p,
                    self.out_proj.weight,
                    self.out_proj.bias,
                    self.training or self.dropout_module.apply_during_inference,
                    key_padding_mask,
                    need_weights,
                    attn_mask,
                    use_separate_proj_weight=True,
                    q_proj_weight=self.q_proj.weight,
                    k_proj_weight=self.k_proj.weight,
                    v_proj_weight=self.v_proj.weight,

                    ) 

            return out, memory, weights

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if saved_state is not None and "prev_key" in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert self.encoder_decoder_attention and not self.self_attention
                    key = value = None
        else:
            saved_state = None
        if not self.selective and not self.shared_memory_attention:

            t1 = time.time()

            if self.self_attention:
                q = self.q_proj(query)
                k = self.k_proj(query)
                v = self.v_proj(query)
            elif self.encoder_decoder_attention:
                # encoder-decoder attention
                q = self.q_proj(query)
                if key is None:
                    assert value is None
                    k = v = None
                else:
                    k = self.k_proj(key)
                    v = self.v_proj(key)

            else:
                assert key is not None and value is not None
                
                q = self.q_proj(query)
                k = self.k_proj(key)
                v = self.v_proj(value)

            if comp is not None:
                v = v * comp
                #v_memory = v_memory * comp
            q *= self.scaling
            #q_memory *= self.scaling

            if self.bias_k is not None:
                assert self.bias_v is not None
                k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
                v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
                if attn_mask is not None:
                    attn_mask = torch.cat(
                        [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                    )
                if key_padding_mask is not None:
                    key_padding_mask = torch.cat(
                        [
                            key_padding_mask,
                            key_padding_mask.new_zeros(key_padding_mask.size(0), 1),
                        ],
                        dim=1,
                    )

            q = (
                q.contiguous()
                .view(tgt_len, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )
            if k is not None:
                k = (
                    k.contiguous()
                    .view(-1, bsz * self.num_heads, self.head_dim)
                    .transpose(0, 1)
                )
            if v is not None:
                v = (
                    v.contiguous()
                    .view(-1, bsz * self.num_heads, self.head_dim)
                    .transpose(0, 1)
                )

            



            if saved_state is not None:
                # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
                if "prev_key" in saved_state:
                    _prev_key = saved_state["prev_key"]
                    assert _prev_key is not None
                    prev_key = _prev_key.view(bsz * self.num_heads, -1, self.head_dim)
                    if static_kv:
                        k = prev_key
                    else:
                        assert k is not None
                        k = torch.cat([prev_key, k], dim=1)
                if "prev_value" in saved_state:
                    _prev_value = saved_state["prev_value"]
                    assert _prev_value is not None
                    prev_value = _prev_value.view(bsz * self.num_heads, -1, self.head_dim)
                    if static_kv:
                        v = prev_value
                    else:
                        assert v is not None
                        v = torch.cat([prev_value, v], dim=1)
                prev_key_padding_mask: Optional[Tensor] = None
                if "prev_key_padding_mask" in saved_state:
                    prev_key_padding_mask = saved_state["prev_key_padding_mask"]
                assert k is not None and v is not None
                key_padding_mask = MultiheadAttention._append_prev_key_padding_mask(
                    key_padding_mask=key_padding_mask,
                    prev_key_padding_mask=prev_key_padding_mask,
                    batch_size=bsz,
                    src_len=k.size(1),
                    static_kv=static_kv,
                )

                saved_state["prev_key"] = k.view(bsz, self.num_heads, -1, self.head_dim)
                saved_state["prev_value"] = v.view(bsz, self.num_heads, -1, self.head_dim)
                saved_state["prev_key_padding_mask"] = key_padding_mask
                # In this branch incremental_state is never None
                assert incremental_state is not None
                incremental_state = self._set_input_buffer(incremental_state, saved_state)
            assert k is not None
            src_len = k.size(1)

            # This is part of a workaround to get around fork/join parallelism
            # not supporting Optional types.
            if key_padding_mask is not None and key_padding_mask.dim() == 0:
                key_padding_mask = None

            if key_padding_mask is not None:
                assert key_padding_mask.size(0) == bsz
                assert key_padding_mask.size(1) == src_len

            if self.add_zero_attn:
                assert v is not None
                src_len += 1
                k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
                v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
                if attn_mask is not None:
                    attn_mask = torch.cat(
                        [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                    )
                if key_padding_mask is not None:
                    key_padding_mask = torch.cat(
                        [
                            key_padding_mask,
                            torch.zeros(key_padding_mask.size(0), 1).type_as(
                                key_padding_mask
                            ),
                        ],
                        dim=1,
                    )

            attn_weights = torch.bmm(q, k.transpose(1, 2))
            attn_weights = MultiheadAttention.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)

            assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

            if attn_mask is not None:
                attn_mask = attn_mask.unsqueeze(0)
                if self.onnx_trace:
                    attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
                attn_weights += attn_mask

            if key_padding_mask is not None:
                # don't attend to padding symbols
                attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
                if not self.tpu:
                    attn_weights = attn_weights.masked_fill(
                        key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                        float("-inf")
                    )
                else:
                    attn_weights = attn_weights.transpose(0, 2)
                    attn_weights = attn_weights.masked_fill(key_padding_mask, float('-inf'))
                    attn_weights = attn_weights.transpose(0, 2)
                attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

            if before_softmax:
                return attn_weights, v

            attn_weights_float = utils.softmax(
                attn_weights, dim=-1, onnx_trace=self.onnx_trace
            )
            attn_weights = attn_weights_float.type_as(attn_weights)
            attn_probs = self.dropout_module(attn_weights)

            assert v is not None
            if self.use_topk:
                k = torch.topk(attn_probs, dim = 2, k = self.topk)
                mask = torch.zeros(attn_probs.size()).to(attn_probs.device)
                mask.scatter_(2, k.indices, 1)
                attn_probs = attn_probs * mask
            attn = torch.bmm(attn_probs, v)
            assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
            if self.onnx_trace and attn.size(1) == 1:
                # when ONNX tracing a single decoder step (sequence length == 1)
                # the transpose is a no-op copy before view, thus unnecessary
                attn = attn.contiguous().view(tgt_len, bsz, embed_dim)
            else:
                attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
            attn = self.out_proj(attn)
            attn_weights: Optional[Tensor] = None
            if need_weights:
                attn_weights = attn_weights_float.view(
                    bsz, self.num_heads, tgt_len, src_len
                ).transpose(1, 0)
                if not need_head_weights:
                    # average attention weights over heads
                    attn_weights = attn_weights.mean(dim=0)
            #print('time taken by default mha:' + str(time.time() - t1))
            return attn, None, attn_weights
        elif not self.selective:
            t1 = time.time()
            if self.memory is None:
                self.memory = self.relational_memory.initial_state(query.size(1), query.size(0)).to(query.device)

            #print(self.memory.size())
                
            
            key = key.transpose(1, 0)

            #print(key.size())
            #memory = self.memory[:key.size(0)]
            #print(self.memory.size())

            t2 = time.time()

            #print(self.memory)

            _,_, self.memory, out_hx_mem_new = self.relational_memory(
                 inputs=key,
                 memory=self.memory#.cuda(),
             )
            #print('time taken by relational:' + str(time.time() - t2))



            #query = query.transpose(1, 0)
            #if self.regressive:
            #    B, T, D = query.size()
            #    query = query.reshape(B * T, -1).unsqueeze(1)
            #out_hx_mem_new, _, _ = self.mem_att(
            #     query,#.reshape((bsz, self.num_blocks_out, self.block_size_out)),
            #     self.memory,
            #     self.memory,
            # )

            #z = torch.zeros(self.memory.size(0) - memory.size(0), memory.size(1), memory.size(2)).to(memory.device)
            #memory = torch.cat((memory, z), dim = 0)
            #self.memory = self.memory + memory
            #print('time taken by shared mha:' + str(time.time() - t1))
            #if self.regressive:
            #    out_hx_mem_new = out_hx_mem_new.squeeze(1)
            #    out_hx_mem_new = out_hx_mem_new.reshape(B, T, -1)

            return out_hx_mem_new.transpose(0, 1), memory, None
            """
            tgt_len = memory.size(0)
            src_len = key.size(0)
            q_memory = self.q_proj_memory(memory)
            k = self.k_proj(key)
            v = self.v_proj(value)
            q_memory = (
                q_memory.contiguous()
                .view(memory.size(0), bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )
            k = (
                k.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )
            v = (
                v.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )
            
            attn_weights_1 = torch.bmm(q_memory, k.transpose(1, 2))
            if key_padding_mask is not None:
                # don't attend to padding symbols
                attn_weights_1 = attn_weights_1.view(bsz, self.num_heads, tgt_len, src_len)
                attn_weights_1 = attn_weights_1.masked_fill(
                        key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                        float("-inf")
                    )
            attn_weights_float_1 = utils.softmax(
                attn_weights_1, dim=-1, onnx_trace=self.onnx_trace
            )
            attn_weights_1 = attn_weights_float_1.type_as(attn_weights_1)
            attn_probs_1 = self.dropout_module(attn_weights_1)
            assert v is not None
            memory = torch.bmm(attn_probs_1, v)
            memory = memory.permute(1, 0, 2)
            memory = memory.reshape(memory.size(0), bsz, self.num_heads, -1)
            memory = memory.reshape(memory.size(0), bsz, -1)
            q = self.q_proj(query)
            
            k_memory = self.k_proj_memory(memory)
            v_memory = self.v_proj_memory(memory)
            q = (
                q.contiguous()
                .view(src_len, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )
            k_memory = (
                k.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )
            v_memory = (
                v.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )
            attn_weights_2 = torch.bmm(q, k_memory.transpose(1, 2))
            
            attn_weights_float_2 = utils.softmax(
                attn_weights_2, dim=-1, onnx_trace=self.onnx_trace
            )
            
            attn_weights_2 = attn_weights_float_2.type_as(attn_weights_2)
            attn_probs_2 = self.dropout_module(attn_weights_2)
            out = torch.bmm(attn_probs_2, v)
            out = out.transpose(0, 1).contiguous().view(src_len, bsz, embed_dim)
            return out, memory, None
            """
    def init_memory(self, bs, ts = None, device = None):
        if not self.regressive:
            self.memory = self.relational_memory.initial_state(bs).to(device)
        else:
            self.memory = self.relational_memory.initial_state(bs, ts).to(device)





    @staticmethod
    def _append_prev_key_padding_mask(
        key_padding_mask: Optional[Tensor],
        prev_key_padding_mask: Optional[Tensor],
        batch_size: int,
        src_len: int,
        static_kv: bool,
    ) -> Optional[Tensor]:
        # saved key padding masks have shape (bsz, seq_len)
        if prev_key_padding_mask is not None and static_kv:
            new_key_padding_mask = prev_key_padding_mask
        elif prev_key_padding_mask is not None and key_padding_mask is not None:
            new_key_padding_mask = torch.cat(
                [prev_key_padding_mask.float(), key_padding_mask.float()], dim=1
            )
        # During incremental decoding, as the padding token enters and
        # leaves the frame, there will be a time when prev or current
        # is None
        elif prev_key_padding_mask is not None:
            filler = torch.zeros(
                (batch_size, src_len - prev_key_padding_mask.size(1)),
                device=prev_key_padding_mask.device,
            )
            new_key_padding_mask = torch.cat(
                [prev_key_padding_mask.float(), filler.float()], dim=1
            )
        elif key_padding_mask is not None:
            filler = torch.zeros(
                (batch_size, src_len - key_padding_mask.size(1)),
                device=key_padding_mask.device,
            )
            new_key_padding_mask = torch.cat(
                [filler.float(), key_padding_mask.float()], dim=1
            )
        else:
            new_key_padding_mask = prev_key_padding_mask
        return new_key_padding_mask

    @torch.jit.export
    def reorder_incremental_state(
        self, incremental_state: Dict[str, Dict[str, Optional[Tensor]]], new_order: Tensor
    ):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer_k = input_buffer[k]
                if input_buffer_k is not None:
                    if self.encoder_decoder_attention and input_buffer_k.size(0) == new_order.size(0):
                        break
                    input_buffer[k] = input_buffer_k.index_select(0, new_order)
            incremental_state = self._set_input_buffer(incremental_state, input_buffer)
        return incremental_state

    def _get_input_buffer(
        self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]
    ) -> Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, "attn_state")
        if result is not None:
            return result
        else:
            empty_result: Dict[str, Optional[Tensor]] = {}
            return empty_result

    def _set_input_buffer(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        buffer: Dict[str, Optional[Tensor]],
    ):
        return self.set_incremental_state(incremental_state, "attn_state", buffer)

    def apply_sparse_mask(attn_weights, tgt_len: int, src_len: int, bsz: int):
        return attn_weights

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + "." if name != "" else ""
        items_to_add = {}
        keys_to_remove = []
        for k in state_dict.keys():
            if k.endswith(prefix + "in_proj_weight"):
                # in_proj_weight used to be q + k + v with same dimensions
                dim = int(state_dict[k].shape[0] / 3)
                items_to_add[prefix + "q_proj.weight"] = state_dict[k][:dim]
                items_to_add[prefix + "k_proj.weight"] = state_dict[k][dim : 2 * dim]
                items_to_add[prefix + "v_proj.weight"] = state_dict[k][2 * dim :]

                keys_to_remove.append(k)

                k_bias = prefix + "in_proj_bias"
                if k_bias in state_dict.keys():
                    dim = int(state_dict[k].shape[0] / 3)
                    items_to_add[prefix + "q_proj.bias"] = state_dict[k_bias][:dim]
                    items_to_add[prefix + "k_proj.bias"] = state_dict[k_bias][
                        dim : 2 * dim
                    ]
                    items_to_add[prefix + "v_proj.bias"] = state_dict[k_bias][2 * dim :]

                    keys_to_remove.append(prefix + "in_proj_bias")

        for k in keys_to_remove:
            del state_dict[k]

        for key, value in items_to_add.items():
            state_dict[key] = value