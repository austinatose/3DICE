import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ProteinSA(nn.Module): # is this too slow? is ESM-IF1 already contextualised? can always remove this if necessary
    def __init__(self, embed_dim, num_heads=8, dropout_rate=0.1):
        super(ProteinSA, self).__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.multiheadattention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate, batch_first=True)

    def forward(self, x, mask=None):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        output, _ = self.multiheadattention(Q, K, V, key_padding_mask=mask)
        return output

class DrugConv(nn.Module): # can afford to be cheap on drug side because UniMol is quite comprehensive
    def __init__(self, input_dim, hidden_dims, dropout_rate=0.1):
        super(DrugConv, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dims[0], kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dims[0], hidden_dims[1], kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class CrossAttention(nn.Module): # refer to CAT-DTI
    def __init__(self, embed_dim, num_heads=8, dropout_rate=0.1):
        super(CrossAttention, self).__init__()
        self.CA = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate, batch_first=True)

    def forward(self, protein_features, drug_features, protein_mask=None, drug_mask=None):
        attended_protein_features, attentionp =self.CA(protein_features, drug_features, drug_features, key_padding_mask=drug_mask)
        attended_drug_features, attentiond = self.CA(drug_features, protein_features, protein_features, key_padding_mask=protein_mask)
        return attended_protein_features, attended_drug_features

class Fusion(nn.Module): # get fixed length representations and concat
    def __init__(self, drug_embed_dim, drug_hidden_dims, protein_embed_dim, protein_hidden_dims, dropout_rate=0.2):
        super(Fusion, self).__init__()
        # 2 for drug but 1 for protein
        self.drug_fc1 = nn.Linear(drug_embed_dim, drug_hidden_dims[0])
        self.drug_fc2 = nn.Linear(drug_hidden_dims[0], drug_hidden_dims[1])

        self.protein_linear = nn.Sequential(
            nn.Linear(protein_embed_dim, protein_hidden_dims[0]),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.BatchNorm1d(protein_hidden_dims[0]),
        )

    def forward(self, protein_features, drug_features): # TODO: Think about pooling strategies
        # pooling strategy from evidti
        # protein_features = torch.mean(protein_features, dim=1)  # mean pooling
        protein_features = self.protein_linear(protein_features) 

        # mean pooling for drugs as recommended by unimol
        # drug_features = torch.mean(drug_features, dim=1)
        drug_features = self.drug_fc1(drug_features)
        drug_features = self.drug_fc2(drug_features)

        # now both drug and protein features are of same dimension of 256

        res = torch.cat((protein_features, drug_features), dim=-1)
        return res

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout_rate=0.2):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.out = nn.Linear(hidden_dims[2], hidden_dims[3])
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = nn.SELU()(self.fc1(x))
        x = self.dropout(x)
        x = nn.SELU()(self.fc2(x))
        x = self.dropout(x)
        x = nn.SELU()(self.fc3(x))
        x = self.dropout(x)
        x = self.out(x)
        x = F.softplus(x) + 1 # !

        return x

class Model(nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()
        # TODO: Fill in the data
        self.protein_sa = ProteinSA(cfg["PROTEIN"]["EMBEDDING_DIM"])
        self.drug_conv = DrugConv(cfg["DRUG"]["EMBEDDING_DIM"], cfg["DRUG"]["DIMS"])
        self.cross_attention = CrossAttention(cfg["PROTEIN"]["EMBEDDING_DIM"])
        self.fusion = Fusion(cfg["DRUG"]["EMBEDDING_DIM"], cfg["DRUG"]["DIMS"], cfg["PROTEIN"]["EMBEDDING_DIM"], cfg["PROTEIN"]["DIMS"])
        self.mlp = MLP(cfg["MLP"]["INPUT_DIM"], cfg["MLP"]["DIMS"])

    def forward(self, protein_seq, drug_graph, mode="train"):
        # i should be able to easily turn off SA and the drug CNN
        protein_features = self.protein_sa(protein_seq)
        drug_features = self.drug_conv(drug_graph)
        attended_protein_features, attended_drug_features = self.cross_attention(protein_features, drug_features)
        fused_features = self.fusion(attended_protein_features, attended_drug_features)
        output = self.mlp(fused_features)

        return output
