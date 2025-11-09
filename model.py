import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ProteinSA(nn.Module): # is this too slow? is ESM-IF1 already contextualised? can always remove this if necessary
    def __init__(self, embed_dim, dropout_rate=0.1):
        super(ProteinSA, self).__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            self.dropout,
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, x, mask=None):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        d_k = Q.size(-1)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)  # apply dropout to attention weights

        attended = torch.matmul(attention_weights, V)

        # residual connection and ff
        attended = x + self.dropout(self.out(attended))
        output = attended + self.dropout(self.ffn(attended))
        return output

    # def scaled_dot_product_attention(self, Q, K, V, mask=None):
    #     # Compute the dot products between Q and K, then scale by the square root of the key dimension
    #     d_k = Q.size(-1)
    #     scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

    #     # Apply mask if provided (useful for masked self-attention in transformers)
    #     if mask is not None:
    #         scores = scores.masked_fill(mask == 0, float('-inf'))

    #     # Softmax to normalize scores, producing attention weights
    #     attention_weights = F.softmax(scores, dim=-1)
        
    #     # Compute the final output as weighted values
    #     output = torch.matmul(attention_weights, V)
    #     return output, attention_weights

class DrugConv(nn.Module): # can afford to be cheap on drug side because UniMol is quite comprehensive


class CrossAttention(nn.Module): # refer to CAT-DTI
# combines CA, getting fixed length representations, and concat 



class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.5):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.out = nn.Linear(hidden_dims[2], output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = nn.SELU()(self.fc1(x))
        x = self.dropout(x)
        x = nn.SELU()(self.fc2(x))
        x = self.dropout(x)
        x = nn.SELU()(self.fc3(x))
        x = self.dropout(x)
        x = nn.functional.softplus(x) + 1

        return x

class Model(nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()
        self.protein_la = ProteinLA(cfg.protein_input_dim, cfg.hidden_dim)
        self.drug_conv = DrugConv(cfg.drug_input_dim, cfg.hidden_dim)
        self.cross_attention = CrossAttention(cfg.hidden_dim)
        self.fusion = Fusion(cfg.hidden_dim)
        self.mlp = MLP(cfg.hidden_dim * 2, [cfg.hidden_dim, cfg.hidden_dim // 2, cfg.hidden_dim // 4], cfg.hidden_dim)

    def forward(self, protein_seq, drug_graph):
        protein_features = self.protein_la(protein_seq)
        drug_features = self.drug_conv(drug_graph)
        attended_protein, attended_drug = self.cross_attention(protein_features, drug_features)
        fused_features = self.fusion(attended_protein, attended_drug)





        output = self.evidential_head(fused_features)
        return output

