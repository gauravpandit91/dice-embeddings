from .base_model import BaseKGE
from .static_funcs import quaternion_mul
from ..types import Tuple, Union, torch, np

class FMult(BaseKGE):
    """ Learning Knowledge Neural Graphs"""
    """ Learning Neural Networks for Knowledge Graphs"""

    def __init__(self, args):
        super().__init__(args)
        self.name = 'FMult'
        self.n_layers = 4
        tuned_embedding_dim = False
        while int(np.sqrt((self.embedding_dim-1) / self.n_layers)) != np.sqrt((self.embedding_dim-1) / self.n_layers):
            self.embedding_dim += 1
            tuned_embedding_dim = True
        if tuned_embedding_dim:
            print(f"\n\n*****Embedding dimension reset to {self.embedding_dim} to fit model architecture!*****\n")
        self.k = int(np.sqrt((self.embedding_dim-1) // self.n_layers))
        self.n = 40
        self.a, self.b = -1.0, 1.0
        self.score_func = "vtp" # "vector triple product"
        #self.score_func = "trilinear"
        #self.score_func = "compositional"
        #self.score_func = "full-compositional"
        self.discrete_points = torch.linspace(self.a, self.b, steps=self.n)#.float()
        
        self.entity_embeddings = torch.nn.Embedding(self.num_entities, self.embedding_dim)
        self.relation_embeddings = torch.nn.Embedding(self.num_relations, self.embedding_dim)
        self.param_init(self.entity_embeddings.weight.data), self.param_init(self.relation_embeddings.weight.data)
        
    def build_func(self, Vec):
        n = len(Vec)
        # (1) Construct self.n_layers layered neural network
        W = list(torch.hsplit(Vec[:, :-1], self.n_layers))
        # (2) Reshape weights of the layers
        for i,w in enumerate(W):
            W[i] = w.reshape(n, self.k, self.k)
        return W, Vec[:, -1]
    
    def build_chain_funcs(self, list_Vec):
        list_W = []
        list_b = []
        for Vec in list_Vec:
            W_, b = self.build_func(Vec)
            list_W.append(W_)
            list_b.append(b)
            
        W = list_W[-1][1:]
        for i in range(len(list_W)-1):
            for j, w in enumerate(list_W[i]):
                if i == 0 and j == 0:
                    W_temp = w
                else:
                    W_temp = w @ W_temp
            W_temp = W_temp + list_b[i]
        W_temp = list_W[-1][0] @ W_temp / (len(list_Vec) * w.shape[1])
        W.insert(0, W_temp)
        return W, list_b[-1]
    
    def compute_func(self, W, b, x) -> torch.FloatTensor:
        out = W[0] @ x.repeat(W[0].shape[1], 1)  # x shape (self.n, ) W[0] shape (batch_size, self.k, self.k)
        for w in W[1:]:
            if np.random.rand() > 0.4: # no non-linearity => better results
                out = out + torch.tanh(w @ out)
            else:
                out = out + w @ out
        #print(out.sum(dim=1).shape)
        return out.sum(dim=1) + b.reshape(-1, 1)
    
    def integration_func(self, list_W, list_b):
        def f(x):
            if len(list_W) == 1:
                return self.compute_func(list_W[0], list_b[0], x)
            score = self.compute_func(list_W[0], list_b[0], x)
            for W, b in zip(list_W[1:], list_b[1:]):
                score = score * self.compute_func(W, b, x)
            return score
        return f
        
    def trapezoid(self, list_W, list_b):
        return torch.trapezoid(self.integration_func(list_W, list_b)(self.discrete_points), x=self.discrete_points)
    
    def forward_triples(self, idx_triple: torch.Tensor) -> torch.Tensor:
        # (1) Retrieve embeddings: batch, \mathbb R^d
        head_ent_emb, rel_emb, tail_ent_emb = self.get_triple_representation(idx_triple)
        if self.discrete_points.device != head_ent_emb.device:
            self.discrete_points = self.discrete_points.to(head_ent_emb.device)
        if self.score_func == "vtp":
            h_W, h_b = self.build_func(head_ent_emb)
            r_W, r_b = self.build_func(rel_emb)
            t_W, t_b = self.build_func(tail_ent_emb)
            out = -self.trapezoid([t_W], [t_b])*self.trapezoid([h_W, r_W], [h_b, r_b]) + self.trapezoid([r_W], [r_b])*self.trapezoid([t_W, h_W], [t_b, h_b])
        elif self.score_func == "compositional":
            t_W, t_b = self.build_func(tail_ent_emb)
            chain_W, chain_b = self.build_chain_funcs([head_ent_emb, rel_emb])
            out = self.trapezoid([chain_W, t_W], [chain_b, t_b])
        elif self.score_func == "full-compositional":
            chain_W, chain_b = self.build_chain_funcs([head_ent_emb, rel_emb, tail_ent_emb])
            out = self.trapezoid([chain_W], [chain_b])
        elif self.score_func == "trilinear":
            h_W, h_b = self.build_func(head_ent_emb)
            r_W, r_b = self.build_func(rel_emb)
            t_W, t_b = self.build_func(tail_ent_emb)
            out = self.trapezoid([h_W, r_W, t_W], [h_b, r_b, t_b])
        return out