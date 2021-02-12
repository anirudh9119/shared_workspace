
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import transformer_utilities.fairseq_utils as utils 

from .layer_norm import LayerNorm
from .multihead_attention import MultiheadAttention
from .relational_memory import RelationalMemory
from .group_linear_layer import GroupLinearLayer
#from fairseq.modules.shared_group_linear_layer import SharedGroupLinearLayer

from .basic_mha import MemoryAttention

import random

from .quant_noise import quant_noise
from .fairseq_dropout import FairseqDropout
from torch import Tensor

import torch.nn.functional as F


class TransformerEncoderLayerVanilla(nn.Module):
    """Encoder layer block.
    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.
    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args, out_proj = None):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.self_attn = self.build_self_attention(self.embed_dim, args)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout = args.dropout
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, "activation_fn", "relu")
        )
        self.activation_dropout = getattr(args, "activation_dropout", 0)
        if self.activation_dropout == 0:
            # for backwards compatibility with models that use args.relu_dropout
            self.activation_dropout = getattr(args, "relu_dropout", 0)
        self.normalize_before = args.encoder_normalize_before
        self.fc1 = self.build_fc1(self.embed_dim, args.encoder_ffn_embed_dim)
        self.fc2 = self.build_fc2(args.encoder_ffn_embed_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)

        if out_proj is not None:
            self.final_linear = nn.Linear(args.encoder_embed_dim, out_proj)
        else:
            self.final_linear = None

    def build_fc1(self, input_dim, output_dim):
        return nn.Linear(input_dim, output_dim)

    def build_fc2(self, input_dim, output_dim):
        return nn.Linear(input_dim, output_dim)

    def build_self_attention(self, embed_dim, args):
        return MultiheadAttention(
            embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=args.self_attention,
            shared_memory_attention = args.shared_memory_attention,
            use_topk = args.use_topk,
            topk = args.topk,
            num_steps = args.num_steps,
            mem_slots = args.mem_slots,
            null_attention = args.null_attention,
            regressive = args.regressive
        )

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {"0": "self_attn_layer_norm", "1": "final_layer_norm"}
        for old, new in layer_norm_map.items():
            for m in ("weight", "bias"):
                k = "{}.layer_norms.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                    del state_dict[k]

    def forward(self, x, encoder_padding_mask, attn_mask: Optional[Tensor] = None, state = None, memory = None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape (T_tgt, T_src), where
            T_tgt is the length of query, while T_src is the length of key,
            though here both query and key is x here,
            attn_mask[t_tgt, t_src] = 1 means when calculating embedding
            for t_tgt, t_src is excluded (or masked out), =0 means it is
            included in attention
        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)
        # anything in original attn_mask = 1, becomes -1e8
        # anything in original attn_mask = 0, becomes 0
        # Note that we cannot use -inf here, because at some edge cases,
        # the attention weight (before softmax) for some padded element in query
        # will become -inf, which results in NaN in model parameters
        # TODO: to formally solve this problem, we need to change fairseq's
        # MultiheadAttention. We will do this later on.
        #print(state is not None)
        x, memory, _ = self.self_attn(
            query=state if state is not None else x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            attn_mask=attn_mask,
            memory = memory
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=float(self.activation_dropout), training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        if self.final_linear is not None:
            x = self.final_linear(x)
        return x, memory

class Attention(nn.Module):
    def __init__(self, n_heads, n_blocks, dim, use_nfm):
        super(Attention, self).__init__()

        self.use_nfm = use_nfm

        #self.n_heads = n_heads
        self.n_heads = 12
        self.n_blocks = n_blocks
        self.dim = dim
        self.block_dim = dim // self.n_blocks
        #self.head_dim = self.block_dim // self.n_heads
        self.head_dim = 64
        self.scale = self.head_dim ** -0.5

        self.query_net = GroupLinearLayer(self.block_dim, self.head_dim * self.n_heads, n_blocks)
        self.key_net = GroupLinearLayer(self.block_dim, self.head_dim * self.n_heads, n_blocks)
        self.value_net = GroupLinearLayer(self.block_dim, self.head_dim * self.n_heads, n_blocks)
        self.final = GroupLinearLayer(self.head_dim * self.n_heads, self.block_dim, n_blocks)

    def forward(self, x, qkv=None):

        use_exshare = False

        if qkv is not None:
            klst, vlst = qkv

        seq_len, bsz, _ = x.shape


        if use_exshare:
            x = x.view(seq_len, bsz, self.n_blocks * self.block_dim)
            q = self.query_net(x).view(seq_len, 1, bsz*self.n_blocks, self.n_heads, self.head_dim)
            k = self.key_net(x).view(seq_len, 1, bsz*self.n_blocks, self.n_heads, self.head_dim)
            v = self.value_net(x).view(seq_len, 1, bsz*self.n_blocks, self.n_heads, self.head_dim)
        else:
            x = x.view(seq_len, bsz, self.n_blocks * self.block_dim)
            q = self.query_net(x).view(seq_len, bsz, self.n_blocks, self.n_heads, self.head_dim)
            k = self.key_net(x).view(seq_len, bsz, self.n_blocks, self.n_heads, self.head_dim)
            v = self.value_net(x).view(seq_len, bsz, self.n_blocks, self.n_heads, self.head_dim)

        q = q.transpose(2,3) * self.scale
        k = k.transpose(2,3)
        v = v.transpose(2,3)

        if random.uniform(0,1) < 0.00001:
            print('use NFM?', self.use_nfm)

        if self.use_nfm:
            if qkv is not None:
                klst.append(k)
                vlst.append(v)
                #print('len qlst', len(qlst))
                #for kval in klst:
                #    print(kval.shape)

            k = torch.cat(klst, dim=3)
            v = torch.cat(vlst, dim=3)


        #should return these q,k,v and save to a big list.  Also pull in from the list passed in and concat along dim=3, i.e. so that it's nblocks * nlayers.
        #print('running comm attention with shapes', q.shape, k.shape, v.shape)

        score = torch.matmul(q, k.transpose(3,4))
        #print('score shape', score.shape)
        score = F.softmax(score, dim=-1)
        out = torch.matmul(score, v).transpose(2,3)
        #print('out shape', out.shape)
        score = score.mean(dim=2)

        out = out.reshape(seq_len, bsz, self.n_blocks * self.head_dim * self.n_heads)
        out = self.final(out)
        out = out.view(seq_len, bsz, self.dim)

        return out, score

class NormLayer(nn.Module):
    def __init__(self, num_rims, dim, export=False):
        super(NormLayer, self).__init__()

        self.num_rims = num_rims
        self.dim = dim

        self.weight = nn.Parameter(torch.ones(1,1,dim*num_rims,))
        self.bias = nn.Parameter(torch.zeros(1,1,dim*num_rims,))

        self.norm = LayerNorm(dim, export=export, elementwise_affine=False)

    def forward(self, x):
        seq_len, bsz, _ = x.shape
        x = x.view(seq_len, bsz, self.num_rims, self.dim)

        x = self.norm(x)

        x = x.view(seq_len, bsz, self.num_rims * self.dim)

        weight_use = self.weight.repeat(seq_len, bsz, 1)
        bias_use = self.bias.repeat(seq_len, bsz, 1)

        x = x * weight_use + bias_use

        return x

class TransformerEncoderLayer(nn.Module):
    """Encoder layer block.
    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.
    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args, nb, blockatt, blockatt_memory, use_nfm, out_proj_dim=None):
        super().__init__()

        self.blockatt = blockatt
        self.blockatt_memory = blockatt_memory

        self.embed_dim = args.encoder_embed_dim
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)


        self.use_nfm = use_nfm

        print('using nfm?', self.use_nfm)

        self.nb = nb

        self.norm_blocks = self.nb

        self.self_attn = self.build_self_attention(self.embed_dim, args) #should divide embed_dim by nb.  Then raise embed_dim in args
        self.self_attn_layer_norm = NormLayer(self.norm_blocks, self.embed_dim // self.norm_blocks)
        self.dropout_module = FairseqDropout(args.dropout, module_name=self.__class__.__name__)
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, "activation_fn", "relu")
        )

        print("SETUP TRANSFORMER LAYER", 'blocks', self.nb)

        activation_dropout_p = getattr(args, "activation_dropout", 0)
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0)
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = args.encoder_normalize_before
        self.fc1 = self.build_fc1(
            self.embed_dim, args.encoder_ffn_embed_dim, self.quant_noise, self.quant_noise_block_size
        )
        self.fc2 = self.build_fc2(
            args.encoder_ffn_embed_dim, self.embed_dim, self.quant_noise, self.quant_noise_block_size
        )

        self.final_layer_norm = NormLayer(self.norm_blocks, self.embed_dim // self.norm_blocks)

        if self.blockatt:
            self.comm = Attention(args.encoder_attention_heads, self.nb, self.embed_dim, self.use_nfm)
            self.comm_norm = NormLayer(self.norm_blocks, self.embed_dim // self.norm_blocks)

        if self.blockatt_memory:
            memory_slots = 4
            memory_head_size = 128
            memory_num_heads = 1
            gate_style = 'memory'
            print('not using special key size gate_style is', gate_style, memory_slots, memory_num_heads, memory_head_size)

            self.memory_layer = RelationalMemory(mem_slots=memory_slots, head_size=memory_head_size, input_size=self.embed_dim, output_size=self.embed_dim,
                                                 num_heads=memory_num_heads, num_blocks=1, forget_bias=1., input_bias=0.,
                                                 attention_mlp_layers=5, gate_style=gate_style)

            #self.n_blocks_val * self.block_dim_val
            #self.block_dim_val = dim_val // self.n_blocks_val
            self.memory_attention = MemoryAttention(n_blocks_query=self.nb, n_blocks_val=8, dim_query=self.embed_dim, dim_val=memory_head_size*memory_num_heads*memory_slots)
            self.self_mem_norm = NormLayer(self.norm_blocks, self.embed_dim // self.norm_blocks)

        #self.competition = GroupLinearLayer(self.embed_dim//self.nb, 1, self.nb, a=0.05)
        #self.comp_sm = nn.Softmax(dim=2)
        self.competition = None

        if out_proj_dim is not None:
            self.out_proj = GroupLinearLayer(self.embed_dim//self.nb, out_proj_dim//self.nb, self.nb)
        else:
            self.out_proj = None

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(GroupLinearLayer(input_dim//self.nb, output_dim//self.nb, self.nb), p=q_noise, block_size=qn_block_size)
        #return quant_noise(nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(GroupLinearLayer(input_dim//self.nb, output_dim//self.nb, self.nb), p=q_noise, block_size=qn_block_size)
        #return quant_noise(nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size)

    def build_self_attention(self, embed_dim, args):
        return MultiheadAttention(
            embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            nblocks=self.nb,
            top_k_ratio = args.topk_ratio,
            use_value_competition = False,


        )

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {"0": "self_attn_layer_norm", "1": "final_layer_norm"}
        for old, new in layer_norm_map.items():
            for m in ("weight", "bias"):
                k = "{}.layer_norms.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                    del state_dict[k]

    def forward(self, x, encoder_padding_mask, attn_mask: Optional[Tensor] = None, state = None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape `(tgt_len, src_len)`,
                where `tgt_len` is the length of output and `src_len` is the
                length of input, though here both are equal to `seq_len`.
                `attn_mask[tgt_i, src_j] = 1` means that when calculating the
                embedding for `tgt_i`, we exclude (mask out) `src_j`. This is
                useful for strided self-attention.
        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        # anything in original attn_mask = 1, becomes -1e8
        # anything in original attn_mask = 0, becomes 0
        # Note that we cannot use -inf here, because at some edge cases,
        # the attention weight (before softmax) for some padded element in query
        # will become -inf, which results in NaN in model parameters

        seq_len, bsz, _ = x.shape

        if self.competition is not None:
            comp = self.competition(x)
            comp = self.comp_sm(comp)
            #comp = F.gumbel_softmax(comp, tau=0.5, hard=False, dim=2)
            comp = comp.unsqueeze(-1).repeat(1,1,1,self.embed_dim//self.nb)
            comp = comp.view((x.shape[0], x.shape[1], self.embed_dim))
        else:
            comp = None

        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)

        x, _ = self.self_attn(
                query=state if state is not None else x,
                key=x,
                value=x,
                key_padding_mask=encoder_padding_mask,
                attn_mask=attn_mask,
            )

        x = self.dropout_module(x)
        x = residual + x

        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if self.blockatt:
            if self.normalize_before:
                x = self.comm_norm(x)

            residual = x
            x, _ = self.comm(x)
            x = self.dropout_module(x)
            x = residual + x

            if not self.normalize_before:
                x = self.comm_norm(x)

        if self.blockatt_memory:
            if self.normalize_before:
                x = self.self_mem_norm(x)
            residual = x
            T,bsz,nhid = x.shape
            if comp is not None:
                x_write = comp * x
            else:
                x_write = x*1.0
            _, new_memory = self.memory_layer.forward_step(x_write.reshape((T*bsz, nhid)), self.memory_obj[0])
            self.memory_obj[0] = new_memory
            Tbs,num_slots,nhid_slot = new_memory.shape
            mem_read = new_memory.reshape((T, bsz, num_slots*nhid_slot))
            x,_ = self.memory_attention(x, mem_read)
            x = residual + x

            if not self.normalize_before:
                x = self.self_mem_norm(x)


        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        #print('fc1 on shape', x.shape, 'in encoder')
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        #print('fc2 on shape', x.shape, 'in encoder')
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)

        if self.out_proj is not None:
            x = self.out_proj(x)

        return x


class TransformerDecoderLayer(nn.Module):
    """Decoder layer block.
    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.decoder_normalize_before* to ``True``.
    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
        self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False, layer_ind=None, out_proj_dim=None
    ):
        super().__init__()

        if True or (layer_ind >= 2 and layer_ind < args.decoder_layers - 1):
            self.blockatt = args.use_module_communication == "True" or args.use_module_communication == "true"
        else:
            self.blockatt = False


        self.use_nfm = args.use_nfm == 'True' or args.use_nfm == 'true'

        print('using nfm?', self.use_nfm)

        self.embed_dim = args.decoder_embed_dim
        self.dropout_module = FairseqDropout(args.dropout, module_name=self.__class__.__name__)
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)

        self.embed_dim = args.decoder_embed_dim
        self.decoder_ffn_embed_dim = args.decoder_ffn_embed_dim
        self.decoder_attention_heads = args.decoder_attention_heads


        self.cross_self_attention = getattr(args, "cross_self_attention", False)

        if layer_ind >= 2 and layer_ind < args.decoder_layers - 1:
            self.nb = args.num_modules
        else:
            self.nb = 1



        print('embed dim', self.embed_dim, 'ffn dim', self.decoder_ffn_embed_dim)

        if layer_ind == 2:
            self.in_proj = nn.Linear(args.decoder_embed_dim,self.embed_dim)
        else:
            self.in_proj = None

        if out_proj_dim is not None:
            self.out_proj = GroupLinear(self.embed_dim//self.nb, out_proj_dim//self.nb, self.nb)
        else:
            self.out_proj = None

        self.layer_ind = layer_ind
        if True and self.nb >= 2:
            self.competition = GroupLinearLayer(self.embed_dim//self.nb, 1, self.nb, a=0.05)
            self.comp_sm = nn.Softmax(dim=2)
            print('using competition!')
        else:
            self.competition = None

        self.norm_blocks = args.num_modules
        print("SETUP TRANSFORMER DECODER LAYER")

        self.self_attn = self.build_self_attention(
            self.embed_dim,
            args,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )
        if self.blockatt:
            self.self_comm = Attention(self.decoder_attention_heads, self.nb, self.embed_dim, self.use_nfm)
            self.self_comm_norm = NormLayer(self.norm_blocks, self.embed_dim // self.norm_blocks)
            self.self_mem_norm = NormLayer(self.norm_blocks, self.embed_dim // self.norm_blocks)


            self.memory_layer = RelationalMemory(mem_slots=5, head_size=32, input_size=self.embed_dim, output_size=self.embed_dim, num_heads=4, num_blocks=1, forget_bias=1., input_bias=0., gate_style='unit')

            self.memory_attention = MemoryAttention(n_blocks_query=self.nb, n_blocks_val=10, dim_query=self.embed_dim, dim_val=5*32*4)

        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, "activation_fn", "relu")
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0)
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0)
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__)
        self.normalize_before = args.decoder_normalize_before

        # use layerNorm rather than FusedLayerNorm for exporting.
        # char_inputs can be used to determint this.
        # TODO  remove this once we update apex with the fix
        export = getattr(args, "char_inputs", False)
        self.self_attn_layer_norm = NormLayer(self.norm_blocks, self.embed_dim // self.norm_blocks, export=export)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
            self.encoder_comm_norm = None
            self.encoder_comm = None
        else:
            self.encoder_attn = self.build_encoder_attention(self.embed_dim, args)
            self.encoder_attn_layer_norm = NormLayer(self.norm_blocks, self.embed_dim // self.norm_blocks, export=export)
            if self.blockatt:
                self.encoder_comm_norm = NormLayer(self.norm_blocks, self.embed_dim // self.norm_blocks)
                self.encoder_comm = Attention(self.decoder_attention_heads, self.nb, self.embed_dim, self.use_nfm)

        print('setup transformer layer decoder blocks: ', self.nb)

        self.fc1 = self.build_fc1(
            self.embed_dim, self.decoder_ffn_embed_dim, self.quant_noise, self.quant_noise_block_size
        )
        self.fc2 = self.build_fc2(
            self.decoder_ffn_embed_dim, self.embed_dim, self.quant_noise, self.quant_noise_block_size
        )

        print('params in self-attn', sum(p.numel() for p in self.self_attn.parameters()))
        print('params in fc', sum(p.numel() for p in self.fc1.parameters()) + sum(p.numel() for p in self.fc2.parameters()))

        self.final_layer_norm = NormLayer(self.norm_blocks, self.embed_dim // self.norm_blocks, export=export)
        self.need_attn = True

        self.onnx_trace = False

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(GroupLinearLayer(input_dim//self.nb, output_dim//self.nb, self.nb), q_noise, qn_block_size)
        #return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(GroupLinearLayer(input_dim//self.nb, output_dim//self.nb, self.nb), q_noise, qn_block_size)
        #return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_self_attention(self, embed_dim, args, add_bias_kv=False, add_zero_attn=False):
        return MultiheadAttention(
            embed_dim,
            self.decoder_attention_heads,
            dropout=args.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=not getattr(args, "cross_self_attention", False),
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            nblocks=self.nb,
            top_k_ratio = args.topk_ratio,
            use_value_competition = False
        )

    def build_encoder_attention(self, embed_dim, args):
        kdim = getattr(args, "encoder_embed_dim", None)
        vdim = getattr(args, "encoder_embed_dim", None)

        return MultiheadAttention(
            embed_dim,
            self.decoder_attention_heads,
            kdim=kdim,
            vdim=vdim,
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            nblocks=self.nb,
            top_k_ratio = args.topk_ratio,
            use_value_competition = False
        )

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
        state = None
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).
        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True

        if self.in_proj is not None:
            x = self.in_proj(x)

        if self.competition is not None:
            comp = self.competition(x)
            comp = self.comp_sm(comp)
            #comp = F.gumbel_softmax(comp, tau=0.5, hard=False, dim=2)
            comp = comp.unsqueeze(-1).repeat(1,1,1,self.embed_dim//self.nb)
            comp = comp.view((x.shape[0], x.shape[1], self.embed_dim))
        else:
            comp = None

        #print('x shape', x.shape)
        #print('self attn mask', self_attn_mask.shape)
        #print('self attn', self_attn_mask[0])

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        if self.cross_self_attention and not (
            incremental_state is not None
            and _self_attn_input_buffer is not None
            and "prev_key" in _self_attn_input_buffer
        ):
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat(
                    (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
                )
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out.size(1), encoder_out.size(0)
                    )
                self_attn_padding_mask = torch.cat(
                    (encoder_padding_mask, self_attn_padding_mask), dim=1
                )
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x

        x, attn = self.self_attn(
            query=state if state is not None else x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask
        )

        x = self.dropout_module(x)
        if comp is None:
            x = residual + x
        else:
            x = residual + x*comp
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

#        if self.blockatt:
#            if self.normalize_before:
#                x = self.self_comm_norm(x)

#            residual = x
#            x, _ = self.self_comm(x, (self.klst,self.vlst))
#            x = self.dropout_module(x)
#            x = residual + x

#            if not self.normalize_before:
#                x = self.self_comm_norm(x)

        if self.blockatt:
            if self.normalize_before:
                x = self.self_mem_norm(x)
            residual = x
            T,bsz,nhid = x.shape
            if comp is not None:
                x_write = comp * x
            else:
                x_write = x*1.0
            _, new_memory = self.memory_layer.forward_step(x_write.reshape((T*bsz, nhid)), self.memory_obj[0])
            self.memory_obj[0] = new_memory
            Tbs,num_slots,nhid_slot = new_memory.shape
            mem_read = new_memory.reshape((T, bsz, num_slots*nhid_slot))
            x,_ = self.memory_attention(x, mem_read)
            x = residual + x

            if not self.normalize_before:
                x = self.self_mem_norm(x)


        if self.encoder_attn is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )

            x = self.dropout_module(x)
            x = residual + x
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

            if self.blockatt:
                if self.normalize_before:
                    x = self.encoder_comm_norm(x)

                residual = x
                x, _ = self.encoder_comm(x)
                x = self.dropout_module(x)
                x = residual + x

                if not self.normalize_before:
                    x = self.encoder_comm_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)

        if self.out_proj is not None:
            x = self.out_proj(x)

        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state



        return x, attn, None

    def make_generation_fast_(self, need_attn: bool = False, **kwargs):
        self.need_attn = need_attn


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


