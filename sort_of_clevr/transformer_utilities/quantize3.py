
"""
The critical quantization layers that we sandwich in the middle of the autoencoder
(between the encoder and decoder) that force the representation through a categorical
variable bottleneck and use various tricks (softening / straight-through estimators)
to backpropagate through the sampling process.
"""

import torch
from torch import nn, einsum
import torch.nn.functional as F
import random

from scipy.cluster.vq import kmeans2

# -----------------------------------------------------------------------------



class Quantize(nn.Module):
    """
    Neural Discrete Representation Learning, van den Oord et al. 2017
    https://arxiv.org/abs/1711.00937

    Follows the original DeepMind implementation
    https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    https://github.com/deepmind/sonnet/blob/v2/examples/vqvae_example.ipynb
    """
    def __init__(self, num_hiddens, n_embed, groups):
        super().__init__()

        embedding_dim = num_hiddens
        self.embedding_dim = embedding_dim
        self.n_embed = n_embed
        self.groups = groups

        self.kld_scale = 10.0

        self.out_proj = nn.Linear(num_hiddens, num_hiddens)
        self.embed = nn.Embedding(n_embed, embedding_dim//groups)

        self.ind_lst = []

        self.register_buffer('data_initialized', torch.zeros(1))

    def forward(self, z):
        B, H, C = z.size()
        W = 1

        # project and flatten out space, so (B, C, H, W) -> (B*H*W, C)
        #z_e = self.proj(z)
        #z_e = z_e.permute(0, 2, 3, 1) # make (B, H, W, C)
        z_e = z.reshape((B, H, self.groups, self.embedding_dim//self.groups)).reshape((B, H*self.groups, self.embedding_dim//self.groups))
        flatten = z_e.reshape(-1, self.embedding_dim//self.groups)

        #flatten = flatten.reshape((flatten.shape[0], self.groups, self.embedding_dim//self.groups)).reshape((flatten.shape[0] * self.groups, self.embedding_dim//self.groups))

        # DeepMind def does not do this but I find I have to... ;\
        if self.training and self.data_initialized.item() == 0:
            print('running kmeans!!') # data driven initialization for the embeddings
            rp = torch.randperm(flatten.size(0))
            kd = kmeans2(flatten[rp[:20000]].data.cpu().numpy(), self.n_embed, minit='points')
            self.embed.weight.data.copy_(torch.from_numpy(kd[0]))
            self.data_initialized.fill_(1)
            # TODO: this won't work in multi-GPU setups

        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed.weight.t()
            + self.embed.weight.pow(2).sum(1, keepdim=True).t()
        )
        _, ind = (-dist).max(1)
        ind = ind.view(B, H*self.groups)


        # vector quantization cost that trains the embedding vectors
        z_q = self.embed_code(ind) # (B, H, W, C)
        commitment_cost = 0.25
        diff = commitment_cost * (z_q.detach() - z_e).pow(2).mean() + (z_q - z_e.detach()).pow(2).mean()
        diff *= self.kld_scale

        z_q = z_e + (z_q - z_e).detach() # noop in forward pass, straight-through gradient estimator in backward pass

        z_q = z_q.reshape((B, H, self.groups, self.embedding_dim//self.groups)).reshape((B, H, self.embedding_dim))

        if True and random.uniform(0,1) < 0.00001:
            print('encoded ind', ind)
            print('before', z[0,0])
            print('after', z_q[0,0])
            print('extra loss on layer', diff)

        if random.uniform(0,1) < 0.0001 or (not self.training and random.uniform(0,1) < 0.001): 
            if self.training:
                print('training mode!')
            else:
                print('test mode!')
            ind_lst = list(set(ind.flatten().cpu().numpy().tolist()))
            
            if self.training: 
                self.ind_lst += ind_lst
                self.ind_lst = self.ind_lst[:50000]
                print('train ind lst', sorted(list(set(self.ind_lst))))
            else:
                print('test ind lst', sorted(list(set(ind_lst))))


        z_q = self.out_proj(z_q)

        return z_q, diff, ind


    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.weight)


class GumbelQuantize(nn.Module):
    """
    Gumbel Softmax trick quantizer
    Categorical Reparameterization with Gumbel-Softmax, Jang et al. 2016
    https://arxiv.org/abs/1611.01144
    """
    def __init__(self, num_hiddens, n_embed, embedding_dim, straight_through=False):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.n_embed = n_embed

        self.straight_through = straight_through
        self.temperature = 1.0
        self.kld_scale = 5e-4

        self.proj = nn.Conv2d(num_hiddens, n_embed, 1)
        self.embed = nn.Embedding(n_embed, embedding_dim)

    def forward(self, z):

        # force hard = True when we are in eval mode, as we must quantize
        hard = self.straight_through if self.training else True

        logits = self.proj(z)
        soft_one_hot = F.gumbel_softmax(logits, tau=self.temperature, dim=1, hard=hard)
        z_q = einsum('b n h w, n d -> b d h w', soft_one_hot, self.embed.weight)

        # + kl divergence to the prior loss
        qy = F.softmax(logits, dim=1)
        diff = self.kld_scale * torch.sum(qy * torch.log(qy * self.n_embed + 1e-10), dim=1).mean()

        ind = soft_one_hot.argmax(dim=1)
        return z_q, diff, ind


if __name__ == "__main__":

    Q = VQVAEQuantize(64, 200, 64, groups=32)

    x = torch.randn(10, 10, 64)

    #loss, q, p, e = Q(x)

    opt = torch.optim.Adam(Q.parameters())

    for iteration in range(0,1000):

        q, loss, e = Q(x)

        print('x shape', x.shape, 'q shape', q.shape)
        #print('loss', loss)
        print('diff', ((q-x)**2).sum())
        print(x[0,0])
        print(q[0,0])

        #loss *= 10.0
        loss.backward()
        opt.step()
        opt.zero_grad()


