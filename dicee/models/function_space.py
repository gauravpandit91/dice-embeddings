from .base_model import BaseKGE
from .static_funcs import quaternion_mul
from ..types import Tuple, Union, torch, np
import torch.nn.functional as F

class FMult(BaseKGE):
    """ Learning Knowledge Neural Graphs"""
    """ Learning Neural Networks for Knowledge Graphs"""

    def __init__(self, args):
        super().__init__(args)
        self.name = 'FMult'
        self.entity_embeddings = torch.nn.Embedding(self.num_entities, self.embedding_dim)
        self.relation_embeddings = torch.nn.Embedding(self.num_relations, self.embedding_dim)
        self.param_init(self.entity_embeddings.weight.data), self.param_init(self.relation_embeddings.weight.data)
        self.n_layers = 2
        self.k = int(np.sqrt(self.embedding_dim // self.n_layers))
        assert self.k**2 * self.n_layers == self.embedding_dim, f"cannot split embedding dimension into {self.k} * n^2, and n an integer"
        self.n = 20
        self.a, self.b = 0.0, 1.0
        #self.score_func = "vtp" # "vector triple product"
        #self.score_func = "trilinear"
        #self.score_func = "compositional"
        self.score_func = "full-compositional"
        self.discrete_points = torch.linspace(self.a, self.b, steps=self.n)#.float()
        
    def build_func(self, Vec):
        n = len(Vec)
        # (1) Construct self.n_layers layered neural network
        W = list(torch.hsplit(Vec, self.n_layers))
        # (2) Reshape weights of the layers
        for i,w in enumerate(W):
            W[i] = w.reshape(n, self.k, self.k)
        return W
    
    def build_chain_funcs(self, list_Vec):
        list_W = [self.build_func(Vec) for Vec in list_Vec]
        W = list_W[-1][1:]
        for i in range(len(list_W)-1):
            for j, w in enumerate(list_W[i]):
                if i == 0 and j == 0:
                    W_temp = w
                else:
                    W_temp = w @ W_temp
        W_temp = list_W[-1][0] @ W_temp / (len(list_Vec) * w.shape[1])
        W.insert(0, W_temp)
        return W
    
    def compute_func(self, W, x) -> torch.FloatTensor:
        out = W[0] @ x.repeat(W[0].shape[1], 1)  # x shape (self.n, ) W[0] shape (batch_size, self.k, self.k)
        for w in W[1:]:
            if np.random.rand() > 1: # no non-linearity => better results
                out = out + torch.tanh(w @ out)
            else:
                out = out + w @ out
        return out.sum(dim=1) 
    
    def integration_func(self, list_W):
        def f(x):
            if len(list_W) == 1:
                return self.compute_func(list_W[0], x)
            score = self.compute_func(list_W[0], x)
            for W in list_W[1:]:
                score = score * self.compute_func(W, x)
            return score
        return f
        
    def trapezoid(self, list_W):
        return torch.trapezoid(self.integration_func(list_W)(self.discrete_points), x=self.discrete_points)
    
    def forward_triples(self, idx_triple: torch.Tensor) -> torch.Tensor:
        # (1) Retrieve embeddings: batch, \mathbb R^d
        head_ent_emb, rel_emb, tail_ent_emb = self.get_triple_representation(idx_triple)
        if self.score_func == "vtp":
            h_W = self.build_func(head_ent_emb)
            r_W = self.build_func(rel_emb)
            t_W = self.build_func(tail_ent_emb)
            out = -self.trapezoid([h_W])*self.trapezoid([r_W, t_W]) - self.trapezoid([t_W])*self.trapezoid([h_W, r_W])# + 2*self.trapezoid([r_W])*self.trapezoid([h_W, t_W])
        elif self.score_func == "compositional":
            t_W = self.build_func(tail_ent_emb)
            out = self.trapezoid([self.build_chain_funcs([head_ent_emb, rel_emb]), t_W])
        elif self.score_func == "full-compositional":
            out = self.trapezoid([self.build_chain_funcs([head_ent_emb, rel_emb, tail_ent_emb])])
        elif self.score_func == "trilinear":
            h_W = self.build_func(head_ent_emb)
            r_W = self.build_func(rel_emb)
            t_W = self.build_func(tail_ent_emb)
            out = self.trapezoid([h_W, r_W, t_W])
        return out