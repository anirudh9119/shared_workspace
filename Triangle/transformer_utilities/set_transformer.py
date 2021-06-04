from .isab import *
from .pos_enc import PositionEncoder

class DeepSet(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output, dim_hidden=128):
        super(DeepSet, self).__init__()
        self.num_outputs = num_outputs
        self.dim_output = dim_output
        self.enc = nn.Sequential(
                nn.Linear(dim_input, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden))
        self.dec = nn.Sequential(
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, num_outputs*dim_output))

    def forward(self, X):
        X = self.enc(X).mean(-2)
        X = self.dec(X).reshape(-1, self.num_outputs, self.dim_output)
        return X

class SetTransformer(nn.Module):
    def __init__(self, dim_input,
            num_inds=32, dim_hidden=128, num_heads=4, ln=True, num_layers = 4):
        super(SetTransformer, self).__init__()
        self.pe = PositionEncoder(dim_input)
        layers = []
        layers.append(ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln))
        for _ in range(num_layers-1):
            layers.append(ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln))
        self.layers = nn.ModuleList(layers)
        # self.enc = nn.Sequential(
        #         ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
        #         ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln),
        #         ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln),
        #         ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln))
        
    def forward(self, X):
        X=X.permute(1,0,2) #self.pe expects T,B,D
        X = self.pe(X)
        X=X.permute(1,0,2) #layer expects B,T,D
        for layer in self.layers:
            X=layer(X)
        return X
