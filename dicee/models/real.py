import torch
from .base_model import *
import numpy as np
from math import sqrt
from .static_funcs import quaternion_mul


class CLf(BaseKGE):
    """Clifford:Embedding Space Search in Clifford Algebras


    h = A_{d \times 1}, B_{d \times p}, C_{d \times q}

    r = A'_{d \times 1}, B'_{d \times p}, C'_{d \times q}

    t = A''_{d \times 1}, B''_{d \times p}, C_{d \times q}

    """

    def __init__(self, args):
        super().__init__(args)
        self.name = 'CLf'
        self.entity_embeddings = nn.Embedding(self.num_entities, self.embedding_dim)
        self.relation_embeddings = nn.Embedding(self.num_relations, self.embedding_dim)
        self.param_init(self.entity_embeddings.weight.data), self.param_init(self.relation_embeddings.weight.data)
        self.p = self.args['p']
        self.q = self.args['q']
        if self.p is None:
            self.p = 0
        if self.q is None:
            self.q = 0

        self.k = self.embedding_dim / (self.p + self.q + 1)
        print(f'k:{self.k}\tp:{self.p}\tq:{self.q}')
        try:
            assert self.k.is_integer()
        except AssertionError:
            raise AssertionError(f'k= embedding_dim / (p + q+ 1) must be a whole number\n'
                                 f'Currently {self.k}={self.embedding_dim} / ({self.p}+ {self.q} +1)')
        self.k = int(self.k)

    def construct_cl_vector(self, batch_x: torch.FloatTensor) -> tuple[
        torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """ Convert a batch of d-dimensional real valued vectors to a batch of k-dimensional Clifford vectors
        batch_x: n \times d

        batch_x_cl: A_{n \times k,1}, B_{n \times k \times p}, C_{n \times k \times q}
        """
        batch_size, d = batch_x.shape
        # (1) A_{n \times k}: take the first k columns
        A = batch_x[:, :self.k].view(batch_size, self.k)
        # (2) B_{n \times p}, C_{n \times q}: take the self.k * self.p columns after the k. column
        if self.p > 0:
            B = batch_x[:, self.k: self.k + (self.k * self.p)].view(batch_size, self.k, self.p)
        else:
            B = torch.zeros((batch_size, self.k, self.p))

        if self.q > 0:
            # (3) B_{n \times p}, C_{n \times q}: take the last self.k * self.q .
            C = batch_x[:, -(self.k * self.q):].view(batch_size, self.k, self.q)
        else:
            C = torch.zeros((batch_size, self.k, self.q))

        return A, B, C

    def clifford_mul_with_einsum(self, A_h, B_h, C_h, A_r, B_r, C_r):
        """ Compute CL multiplication """
        # (1) Compute A: batch_size (b), A \in \mathbb{R}^k
        A = torch.einsum('bk,bk->bk', A_h, A_r) \
            + torch.einsum('bkp,bkp->bk', B_h, B_r) \
            - torch.einsum('bkq,bkq->bk', C_h, C_r)
        # (2) Compute B: batch_size (b), B \in \mathbb{R}^{k \times p}
        B = torch.einsum('bk,bkp->bkp', A_h, B_r) + torch.einsum('bk,bkp->bkp', A_r, B_h)
        # (3) Compute C: batch_size (b), C \in \mathbb{R}^{k \times q}
        C = torch.einsum('bk,bkq->bkq', A_h, C_r) + torch.einsum('bk,bkq->bkq', A_r, C_h)

        # (4) Compute D: batch_size (b), D \in \mathbb{R}^{k \times  p \times p}
        """
        x = B_h.transpose(1, 2)
        b, p, k = x.shape
        results = []
        for i in range(k):
            a = x[:, :, i].view(len(x), self.p, 1)
            b = B_r[:, i, :].view(len(x), 1, self.p)
            results.append(a @ b)
        results = torch.stack(results, dim=1)
        """
        D = torch.einsum('bkp,bkl->bkpl', B_h, B_r) - torch.einsum('bkp,bkl->bkpl', B_r, B_h)
        # (5) Compute E: batch_size (b), E \in \mathbb{R}^{k \times  q \times q}
        E = torch.einsum('bkq,bkl->bkql', C_h, C_r) - torch.einsum('bkq,bkl->bkql', C_r, C_h)
        # (5) Compute F: batch_size (b), E \in \mathbb{R}^{k \times  p \times q}
        F = torch.einsum('bkp,bkq->bkpq', B_h, C_r) - torch.einsum('bkp,bkq->bkpq', B_r, C_h)

        return A, B, C, D, E, F

    def forward_triples(self, x: torch.Tensor) -> torch.FloatTensor:
        # (1) Retrieve real-valued embedding vectors.
        head_ent_emb, rel_ent_emb, tail_ent_emb = self.get_triple_representation(x)
        # (2) Construct k-dimensional vector in CL_{p,q} for head entities.
        A_head, B_head, C_head = self.construct_cl_vector(head_ent_emb)
        # (3) Construct n dimensional vector in CL_{p,q} for relations.
        A_rel, B_rel, C_rel = self.construct_cl_vector(rel_ent_emb)
        # (4) Construct n dimensional vector in CL_{p,q} for tail entities.
        A_tail, B_tail, C_tail = self.construct_cl_vector(tail_ent_emb)

        # (5) Clifford multiplication of (2) and (3)
        A, B, C, D, E, F = self.clifford_mul_with_einsum(A_head, B_head, C_head, A_rel, B_rel, C_rel)
        # (6) Inner product of (5) and (4)
        A_score = torch.einsum('bk,bk->b', A_tail, A)
        B_score = torch.einsum('bkp,bkp->b', B_tail, B)
        C_score = torch.einsum('bkq,bkq->b', C_tail, C)
        D_score = torch.einsum('bkpp->b', D)
        E_score = torch.einsum('bkqq->b', E)
        F_score = torch.einsum('bkpq->b', F)
        return A_score + B_score + C_score + D_score + E_score + F_score

    def forward_k_vs_all(self, x: torch.Tensor) -> torch.FloatTensor:
        # (1) Retrieve real-valued embedding vectors.
        head_ent_emb, rel_ent_emb = self.get_head_relation_representation(x)
        # (2) Construct k-dimensional vector in CL_{p,q} for head entities.
        A_head, B_head, C_head = self.construct_cl_vector(head_ent_emb)
        # (3) Construct n dimensional vector in CL_{p,q} for relations.
        A_rel, B_rel, C_rel = self.construct_cl_vector(rel_ent_emb)
        # (5) Clifford multiplication of (2) and (3).
        A, B, C, D, E, F = self.clifford_mul_with_einsum(A_head, B_head, C_head, A_rel, B_rel, C_rel)

        # (6) Extract embeddings for all entities.
        Emb_all = self.entity_embeddings.weight
        # (7) Compute A
        A_all = Emb_all[:, :self.k]
        A_score = torch.einsum('bk,ek->be', A, A_all)
        # (8) Compute B
        if self.p > 0:
            B_all = Emb_all[:, self.k: self.k + (self.k * self.p)].view(self.num_entities, self.k, self.p)
            B_score = torch.einsum('bkp,ekp->be', B, B_all)
        else:
            B_score = 0
        # (9) Compute C
        if self.q > 0:
            C_all = Emb_all[:, -(self.k * self.q):].view(self.num_entities, self.k, self.q)
            C_score = torch.einsum('bkq,ekq->be', C, C_all)
        else:
            C_score = 0
        # (10) Aggregate (7,8,9)
        A_B_C_score = A_score + B_score + C_score
        # (11) Compute and Aggregate D,E,F
        D_E_F_score = torch.einsum('bkpp->b', D) + torch.einsum('bkqq->b', E) + torch.einsum('bkpq->b', F)
        D_E_F_score = D_E_F_score.view(len(D_E_F_score), 1)
        # (12) Score
        return A_B_C_score + D_E_F_score


class DistMult(BaseKGE):
    """
    Embedding Entities and Relations for Learning and Inference in Knowledge Bases
    https://arxiv.org/abs/1412.6575"""

    def __init__(self, args):
        super().__init__(args)
        self.name = 'DistMult'
        self.entity_embeddings = nn.Embedding(self.num_entities, self.embedding_dim)
        self.relation_embeddings = nn.Embedding(self.num_relations, self.embedding_dim)
        self.param_init(self.entity_embeddings.weight.data), self.param_init(self.relation_embeddings.weight.data)

    def forward_triples(self, x: torch.Tensor) -> torch.Tensor:
        # (1) Retrieve embeddings & Apply Dropout & Normalization.
        head_ent_emb, rel_ent_emb, tail_ent_emb = self.get_triple_representation(x)
        # (2) Compute the score
        return (self.hidden_dropout(self.hidden_normalizer(head_ent_emb * rel_ent_emb)) * tail_ent_emb).sum(dim=1)

    def forward_k_vs_all(self, x: torch.LongTensor):
        emb_head_real, emb_rel_real = self.get_head_relation_representation(x)
        return torch.mm(self.hidden_dropout(self.hidden_normalizer(emb_head_real * emb_rel_real)),
                        self.entity_embeddings.weight.transpose(1, 0))

    def forward_k_vs_sample(self, x: torch.LongTensor, target_entity_idx: torch.LongTensor):
        emb_head_real, emb_rel_real = self.get_head_relation_representation(x)
        hr = self.hidden_dropout(self.hidden_normalizer(emb_head_real * emb_rel_real)).unsqueeze(1)
        t = self.entity_embeddings(target_entity_idx).transpose(1, 2)
        return torch.bmm(hr, t).squeeze(1)


class TransE(BaseKGE):
    """
    Translating Embeddings for Modeling
    Multi-relational Data
    https://proceedings.neurips.cc/paper/2013/file/1cecc7a77928ca8133fa24680a88d2f9-Paper.pdf"""

    def __init__(self, args):
        super().__init__(args)
        self.name = 'TransE'
        self._norm = 2
        self.margin = 4
        self.entity_embeddings = nn.Embedding(self.num_entities, self.embedding_dim)
        self.relation_embeddings = nn.Embedding(self.num_relations, self.embedding_dim)
        self.param_init(self.entity_embeddings.weight.data), self.param_init(self.relation_embeddings.weight.data)

    def forward_triples(self, x: torch.Tensor) -> torch.FloatTensor:
        # (1) Retrieve embeddings & Apply Dropout & Normalization.
        head_ent_emb, rel_ent_emb, tail_ent_emb = self.get_triple_representation(x)
        # Original d:=|| s+p - t||_2 \approx 0 distance, if true
        # if d =0 sigma(5-0) => 1
        # if d =5 sigma(5-5) => 0.5
        # Update: sigmoid( \gamma - d)
        distance = self.margin - torch.nn.functional.pairwise_distance(head_ent_emb + rel_ent_emb, tail_ent_emb,
                                                                       p=self._norm)
        return distance

    def forward_k_vs_all(self, x: torch.Tensor) -> torch.FloatTensor:
        emb_head_real, emb_rel_real = self.get_head_relation_representation(x)
        distance = torch.nn.functional.pairwise_distance(torch.unsqueeze(emb_head_real + emb_rel_real, 1),
                                                         self.entity_embeddings.weight, p=self._norm)
        return self.margin - distance


class Shallom(BaseKGE):
    """ A shallow neural model for relation prediction (https://arxiv.org/abs/2101.09090) """

    def __init__(self, args):
        super().__init__(args)
        self.name = 'Shallom'
        # Fixed
        shallom_width = int(2 * self.embedding_dim)
        self.entity_embeddings = nn.Embedding(self.num_entities, self.embedding_dim)
        self.param_init(self.entity_embeddings.weight.data)
        self.shallom = nn.Sequential(nn.Dropout(self.input_dropout_rate),
                                     torch.nn.Linear(self.embedding_dim * 2, shallom_width),
                                     self.normalizer_class(shallom_width),
                                     nn.ReLU(),
                                     nn.Dropout(self.hidden_dropout_rate),
                                     torch.nn.Linear(shallom_width, self.num_relations))

    def get_embeddings(self) -> Tuple[np.ndarray, None]:
        return self.entity_embeddings.weight.data.detach(), None

    def forward_k_vs_all(self, x) -> torch.FloatTensor:
        e1_idx: torch.Tensor
        e2_idx: torch.Tensor
        e1_idx, e2_idx = x[:, 0], x[:, 1]
        emb_s, emb_o = self.entity_embeddings(e1_idx), self.entity_embeddings(e2_idx)
        return self.shallom(torch.cat((emb_s, emb_o), 1))

    def forward_triples(self, x) -> torch.FloatTensor:
        """

        :param x:
        :return:
        """

        n, d = x.shape
        assert d == 3
        scores_for_all_relations = self.forward_k_vs_all(x[:, [0, 2]])
        return scores_for_all_relations[:, x[:, 1]].flatten()


class Pyke(BaseKGE):
    """ A Physical Embedding Model for Knowledge Graphs """

    def __init__(self, args):
        super().__init__(args)
        self.name = 'Pyke'
        self.entity_embeddings = nn.Embedding(self.num_entities, self.embedding_dim)
        self.relation_embeddings = nn.Embedding(self.num_relations, self.embedding_dim)
        self.param_init(self.entity_embeddings.weight.data), self.param_init(self.relation_embeddings.weight.data)
        self.loss = nn.TripletMarginLoss(margin=1.0, p=2)

    def get_embeddings(self) -> Tuple[np.ndarray, Union[np.ndarray, None]]:
        return self.entity_embeddings.weight.data.data.detach(), None

    def loss_function(self, x: torch.FloatTensor, y=None) -> torch.FloatTensor:
        anchor, positive, negative = x
        return self.loss(anchor, positive, negative)

    def forward_sequence(self, x: torch.LongTensor):
        # (1) Anchor node Embedding: N, D
        anchor = self.entity_embeddings(x[:, 0])
        # (2) Positives and Negatives
        pos, neg = torch.hsplit(x[:, 1:], 2)
        # (3) Embeddings for Pos N, K, D
        pos_emb = self.entity_embeddings(pos)
        # (4) Embeddings for Negs N, K, D
        neg_emb = self.entity_embeddings(neg)
        # (5) Mean.
        # N, D
        mean_pos_emb = pos_emb.mean(dim=1)
        mean_neg_emb = neg_emb.mean(dim=1)
        return anchor, mean_pos_emb, mean_neg_emb


# TODO: need refactoring
class KPDistMult(BaseKGE):
    """
    Named as KD-Rel-DistMult  in our paper
    """

    def __init__(self, args):
        super().__init__(args.learning_rate)
        self.name = 'KPDistMult'
        self.loss = torch.nn.BCEWithLogitsLoss()
        # Init Embeddings # must have valid root
        # (1) Initialize embeddings
        self.embedding_dim = args.embedding_dim
        self.emb_ent_real = nn.Embedding(args.num_entities, args.embedding_dim)
        self.emb_rel_real = nn.Embedding(args.num_relations, int(sqrt(args.embedding_dim)))
        xavier_normal_(self.emb_ent_real.weight.data), xavier_normal_(self.emb_rel_real.weight.data)
        # (2) Initialize Dropouts
        self.input_dp_ent_real = torch.nn.Dropout(args.input_dropout_rate)
        self.input_dp_rel_real = torch.nn.Dropout(args.input_dropout_rate)
        self.hidden_dropout = torch.nn.Dropout(args.hidden_dropout_rate)

        # (3) Initialize Batch Norms
        self.bn_ent_real = torch.nn.BatchNorm1d(args.embedding_dim)
        self.bn_rel_real = torch.nn.BatchNorm1d(args.embedding_dim)
        self.bn_hidden_real = torch.nn.BatchNorm1d(args.embedding_dim)

    def get_embeddings(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.emb_ent_real.weight.data.data.detach(), self.emb_rel_real.weight.data.detach()

    def forward_k_vs_all(self, x):
        e1_idx: torch.Tensor
        rel_idx: torch.Tensor
        e1_idx, rel_idx = x[:, 0], x[:, 1]

        # (1) Retrieve  head entity embeddings and apply BN + DP
        emb_head_real = self.input_dp_ent_real(self.bn_ent_real(self.emb_ent_real(e1_idx)))
        emb_rel_real = self.emb_rel_real(rel_idx)
        # (2) Retrieve  relation embeddings and apply kronecker_product
        emb_rel_real = batch_kronecker_product(emb_rel_real.unsqueeze(1), emb_rel_real.unsqueeze(1)).flatten(1)
        # (3) Apply BN + DP on (2)
        emb_rel_real = self.input_dp_rel_real(self.bn_rel_real(emb_rel_real))
        # (4) Compute scores
        return torch.mm(self.hidden_dropout(self.bn_hidden_real(emb_head_real * emb_rel_real)),
                        self.emb_ent_real.weight.transpose(1, 0))


class KronE(BaseKGE):
    """ Kronecker Decomposition applied on Entitiy and Relation Embedding matrices KP-DistMult """

    def __init__(self, args):
        super().__init__(args.learning_rate)
        self.name = 'KronE'
        self.loss = torch.nn.BCEWithLogitsLoss()
        # Init Embeddings # must have valid root
        # (1) Initialize embeddings
        self.embedding_dim = int(sqrt(args.embedding_dim))
        self.embedding_dim_rel = int(sqrt(args.embedding_dim))
        self.emb_ent_real = nn.Embedding(args.num_entities, self.embedding_dim)
        self.emb_rel_real = nn.Embedding(args.num_relations, self.embedding_dim_rel)
        xavier_normal_(self.emb_ent_real.weight.data), xavier_normal_(self.emb_rel_real.weight.data)
        # (2) Initialize Dropouts
        self.input_dp_ent_real = torch.nn.Dropout(args.input_dropout_rate)
        self.input_dp_rel_real = torch.nn.Dropout(args.input_dropout_rate)
        """
        # Linear transformation W is a by m by n matrix ,
        # where n is the kronecker product of h and r
        self.m = self.embedding_dim
        self.n = int((self.embedding_dim * self.embedding_dim_rel))
        # (2) With additional parameters
        self.m1, self.n1 = self.m, self.n // self.m
        self.A = nn.parameter.Parameter(torch.randn(self.m1, self.n1, requires_grad=True))

        self.m2, self.n2 = self.m // self.m1, self.n // self.n1
        self.B = nn.parameter.Parameter(torch.randn(self.m2, self.n2, requires_grad=True))
        """

        # (3) Initialize Batch Norms
        self.bn_ent_real = torch.nn.BatchNorm1d(self.embedding_dim)
        self.bn_rel_real = torch.nn.BatchNorm1d(self.embedding_dim_rel)

    def get_embeddings(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.emb_ent_real.weight.data.data.detach(), self.emb_rel_real.weight.data.detach()

    def construct_entity_embeddings(self, e1_idx: torch.Tensor):
        emb_head = self.bn_ent_real(self.emb_ent_real(e1_idx)).unsqueeze(1)
        return batch_kronecker_product(emb_head, emb_head).flatten(1)

    def construct_relation_embeddings(self, rel_idx):
        emb_rel = self.bn_rel_real(self.emb_rel_real(rel_idx)).unsqueeze(1)
        return batch_kronecker_product(emb_rel, emb_rel).flatten(1)

    def forward_k_vs_all(self, x):
        e1_idx: torch.Tensor
        rel_idx: torch.Tensor
        e1_idx, rel_idx = x[:, 0], x[:, 1]
        # (1) Prepare compressed embeddings, from d to d^2.
        # (1.1) Retrieve compressed embeddings
        # (1.2) Apply BN (1.1)
        # (1.3) Uncompress (1.2)
        # (1.4) Apply DP (1.3)
        emb_head_real = self.input_dp_ent_real(self.construct_entity_embeddings(e1_idx))
        # (1) Prepare compressed embeddings, from d to d^2.
        # (1.1) Retrieve compressed embeddings
        # (1.2) Apply BN (1.1)
        # (1.3) Uncompress (1.2)
        # (1.4) Apply DP (1.3)
        emb_rel_real = self.input_dp_rel_real(self.construct_relation_embeddings(rel_idx))
        # (3)
        # (3.1) Capture interactions via Hadamard Product (1) and (2);
        feature = emb_head_real * emb_rel_real
        n, dim = feature.shape
        n_rows = dim // self.embedding_dim
        feature = feature.reshape(n, n_rows, self.embedding_dim)
        # (6) Compute sum of logics Logits
        logits = torch.matmul(feature, self.emb_ent_real.weight.transpose(1, 0)).sum(dim=1)
        return logits


class KronELinear(BaseKGE):
    def __init__(self, args):
        super().__init__(args)
        self.name = 'KronELinear'
        # Init Embeddings # must have valid root
        # (1) Initialize embeddings
        self.entity_embedding_dim = args.embedding_dim
        self.rel_embedding_dim = args.embedding_dim
        self.emb_ent_real = nn.Embedding(args.num_entities, self.entity_embedding_dim)
        self.emb_rel_real = nn.Embedding(args.num_relations, self.rel_embedding_dim)
        xavier_normal_(self.emb_ent_real.weight.data), xavier_normal_(self.emb_rel_real.weight.data)

        # (2) Initialize Dropouts
        self.input_dp_ent_real = torch.nn.Dropout(args.input_dropout_rate)
        self.input_dp_rel_real = torch.nn.Dropout(args.input_dropout_rate)

        # Linear transformation W is a by mp by nq matrix
        # where
        # mp is the kronecker product of h and r and
        # nq is the entity_embedding_dim
        # W: X \otimes Z : W mp by nq
        # X m1 by n1
        # Z mp/m1 by nq/n1

        # output features
        mp = self.entity_embedding_dim
        # Input features
        nq = int((self.entity_embedding_dim ** 2))
        # (2) With additional parameters
        self.m1, self.n1 = mp // 4, nq // 4
        self.X = nn.parameter.Parameter(torch.randn(self.m1, self.n1, requires_grad=True))

        self.m2, self.n2 = mp // self.m1, nq // self.n1
        self.Z = nn.parameter.Parameter(torch.randn(self.m2, self.n2, requires_grad=True))

        # (3) Initialize Batch Norms
        self.bn_ent_real = torch.nn.BatchNorm1d(self.entity_embedding_dim)
        self.bn_rel_real = torch.nn.BatchNorm1d(self.rel_embedding_dim)

    def get_embeddings(self):
        return self.emb_ent_real.weight.data.data.detach(), self.emb_rel_real.weight.data.detach()

    def construct_entity_embeddings(self, e1_idx: torch.Tensor):
        emb_head = self.bn_ent_real(self.emb_ent_real(e1_idx)).unsqueeze(1)
        return batch_kronecker_product(emb_head, emb_head).flatten(1)

    def construct_relation_embeddings(self, rel_idx):
        emb_rel = self.bn_rel_real(self.emb_rel_real(rel_idx)).unsqueeze(1)
        return batch_kronecker_product(emb_rel, emb_rel).flatten(1)

    def forward_k_vs_all(self, x):
        e1_idx: torch.Tensor
        rel_idx: torch.Tensor
        e1_idx, rel_idx = x[:, 0], x[:, 1]
        # (1) Prepare compressed embeddings, from d to d^2.
        # (1.1) Retrieve compressed embeddings
        # (1.2) Apply BN (1.1)
        # (1.3) Uncompress (1.2)
        # (1.4) Apply DP (1.3)
        emb_head_real = self.input_dp_ent_real(self.construct_entity_embeddings(e1_idx))
        # (1) Prepare compressed embeddings, from d to d^2.
        # (1.1) Retrieve compressed embeddings
        # (1.2) Apply BN (1.1)
        # (1.3) Uncompress (1.2)
        # (1.4) Apply DP (1.3)
        emb_rel_real = self.input_dp_rel_real(self.construct_relation_embeddings(rel_idx))
        # (3)
        # (3.1) Capture interactions via Hadamard Product (1) and (2);
        feature = emb_head_real + emb_rel_real
        feature = kronecker_linear_transformation(self.X, self.Z, feature)
        # (6) Compute sum of logics Logits
        logits = torch.matmul(feature, self.emb_ent_real.weight.transpose(1, 0))
        return logits


def batch_kronecker_product(a, b):
    """
    Kronecker product of matrices a and b with leading batch dimensions.
    Batch dimensions are broadcast. The number of them mush
    :type a: torch.Tensor
    :type b: torch.Tensor
    :rtype: torch.Tensor
    """
    siz1 = torch.Size(torch.tensor(a.shape[-2:]) * torch.tensor(b.shape[-2:]))
    res = a.unsqueeze(-1).unsqueeze(-3) * b.unsqueeze(-2).unsqueeze(-4)
    siz0 = res.shape[:-4]
    res = res.reshape(siz0 + siz1)
    return res


def kronecker_linear_transformation(X, Z, x):
    """
    W:X\otimes Z: mp by nq matrix
      X :m1 by n1
      Z : mp/m1 by nq/n1

    1) R(x) nq/n1 by n1 matrix
    2) Z (1)
    Let a linear transformation defined by $W\ in R^{mp\times nq}$
    Let a matrix $A \in \mathbb{R}^{m_1  \times n_1} $ and
    a matrix $ B \in \mathbb{R}^{ \frac{mp}{m_1} \times \frac{nq}{n_1}}$.

    (A\otimes B)x=\mathcal{V}(B \; \mathcal{R}_{\frac{n}{n_1} \times n_1}(x) A^\top), \label{Eq:kronecker}
    \end{equation}
    where
    \begin{enumerate}
        \item $x \in \mathbb{R}^n$ represent input feature vector,
        \item $\mathcal{V}$ transforms a matrix to a vector by stacking its columns,
        \item $ \mathcal{R}_{ \frac{n}{n_1} \times n_1} $
        converts x to a $\frac{n}{n_1}$ by $n_1$ matrix by dividing the vector to columns of size $\frac{n}{n_1}$
        and concatenating the resulting columns together
    For more details, please see this wonderful paper
    KroneckerBERT: Learning Kronecker Decomposition for Pre-trained Language Models via Knowledge Distillation

    :type A: torch.Tensor
    :type B: torch.Tensor
    :type x: torch.Tensor

    :rtype: torch.Tensor
    """
    m1, n1 = X.shape
    mp_div_m1, nq_div_n1 = Z.shape
    n, dim = x.shape

    x = x.reshape(n, n1, nq_div_n1)  # x tranpose for the batch computation
    Zx = torch.matmul(x, Z).transpose(1, 2)
    out = torch.matmul(Zx, X.T)
    return out.flatten(1)