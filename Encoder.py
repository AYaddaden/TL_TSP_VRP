#self.demands.append(torch.randint(low=0, high=9, size=(size,), dtype=torch.float32) + 1.0)
import torch
import torch.nn as nn
from math import sqrt
import torch.nn.functional as F

class Depot_Embedding(nn.Module):
    def __init__(self, depot_dim, embedding_dim):
        super(Depot_Embedding, self).__init__()
        self.embedding  = nn.Linear(depot_dim, embedding_dim)

    def forward(self, depot):
        """

        :param depot: [ batch_size, 1, depot_dim]
        :return: depot_embedding:  [batch_size, 1, embedding_dim]
        """
        depot_embedding = self.embedding(depot)

        return depot_embedding

class City_Embedding(nn.Module):
    def __init__(self, city_dim, embedding_dim):
        super(City_Embedding, self).__init__()
        self.embedding = nn.Linear(city_dim, embedding_dim)


    def forward(self, inputs):
        """
        :param inputs: [batch_size, seq_len, 2]
        :return: embedded_capacity: [batch_size, seq_len, embedding_dim]
        """
        embedded_cities = self.embedding(inputs)

        return embedded_cities


class Capacity_Embedding(nn.Module):
    def __init__(self, capacity_dim, embedding_dim):
        super(Capacity_Embedding, self).__init__()
        self.d_cap = nn.Linear(capacity_dim, embedding_dim)

    def forward(self, inputs):
        """

        :param inputs: [batch_size, 1, 1]
        :return: embedded_capacity: [batch_size, 1, embedding_dim]
        """
        embedded_capacity = self.d_cap(inputs)

        return embedded_capacity

class Demand_Embedding(nn.Module):
    def __init__(self, demand_dim, embedding_dim):
        super(Demand_Embedding, self).__init__()
        self.d_dem = nn.Linear(demand_dim, embedding_dim)

    def forward(self, inputs):
        """

        :param inputs: [batch_size, seq_len, 1]
        :return: embedded_demand: [batch_size, seq_len, embedding_dim]
        """
        embedded_demand = self.d_dem(inputs)

        return embedded_demand

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, embedding_dim, q_dim, k_dim, v_dim):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_heads

        self.embedding_dim = embedding_dim
        self.hidden_dim = self.embedding_dim // self.n_head


        self.q = nn.Linear(q_dim, self.embedding_dim, bias=False)
        self.k = nn.Linear(k_dim, self.embedding_dim, bias=False)
        self.v = nn.Linear(v_dim, self.embedding_dim, bias=False)

        self.out = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
    def forward(self, query, key, value, mask=None):
        """

        :param query, key, value: [batch_size, seq_len, embedding_dim]
        :param mask : [batch_size, seq_len]
        :return: out : [batch_size, seq_len, embedding_dim]
        """
        batch_size = query.size(0)

        Q = self.q(query).reshape(batch_size, -1, self.n_head, self.hidden_dim).permute(0, 2, 1, 3) # [batch_size, seq_len, embedding_dim] --> [batch_size, n_head, seq_len, hidden_dim]
        K = self.k(key).reshape(batch_size, -1, self.n_head, self.hidden_dim).permute(0, 2, 1, 3)
        V = self.v(value).reshape(batch_size, -1, self.n_head, self.hidden_dim).permute(0, 2, 1, 3)

        #[batch_size, n_head, seq_len, hidden_dim] * [batch_size, n_head, hidden_dim, seq_len]  --> [batch_size, n_head, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(3,2)) / sqrt(self.hidden_dim)

        if mask is not None:
            #mask = mask.unsqueeze(1).unsqueeze(1)
            scores = scores.masked_fill(mask, float('-inf'))

        attention = torch.nn.functional.softmax(scores, dim=-1)

        # [batch_size, n_head, seq_len, seq_len] * [batch_size, n_head, seq_len, hidden_dim] --> [batch_size, n_head, seq_len, hidden_dim]
        output = torch.matmul(attention, V)

        # [batch_size, n_head, seq_len, hidden_dim] --> [batch_size, seq_len, n_head * hidden_dim] or [batch_size, seq_len, embedding_dim]
        concat_output = output.transpose(1,2).contiguous().view(batch_size,-1, self.n_head * self.hidden_dim)

        out = self.out(concat_output)

        #[batch_size, seq_len, embedding_dim]
        return out

class FeedForwardLayer(nn.Module):
    def __init__(self, embedding_dim, d_ff):
        super(FeedForwardLayer, self).__init__()
        self.embedding_dim = embedding_dim
        self.d_ff = d_ff

        self.lin1 = nn.Linear(self.embedding_dim, self.d_ff)
        self.lin2 = nn.Linear(self.d_ff, self.embedding_dim)

    def forward(self, inputs):
        """

        :param inputs: [batch_size, seq_len, embedding_dim]
        :return: out : [batch_size, seq_len, embedding_dim]
        """

        #  [batch_size, seq_len, embedding_dim] -->  [batch_size, seq_len, d_ff]
        x = F.relu( self.lin1(inputs) )
        #  [batch_size, seq_len, d_ff] -->  [batch_size, seq_len, embedding_dim]
        x = self.lin2(x)

        return x


class EncoderLayer(nn.Module):
    def __init__(self, n_head, embedding_dim, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.n_head = n_head
        self.embedding_dim = embedding_dim
        self.d_ff = d_ff

        self.mha = MultiHeadAttention(self.n_head, self.embedding_dim, self.embedding_dim, self.embedding_dim, self.embedding_dim)
        self.ffl = FeedForwardLayer(self.embedding_dim, self.d_ff)

        self.dropout1 = nn.Dropout(dropout) # MHA dropout
        self.dropout2 = nn.Dropout(dropout) # FFL dropout

        self.bn1 = nn.BatchNorm1d(self.embedding_dim, affine=True)
        self.bn2 = nn.BatchNorm1d(self.embedding_dim, affine=True)

    def forward(self, x):
        """

        :param x: [batch_size, seq_len, embedding_dim]
        :return: out : [batch_size, seq_len, embdding_dim]
        """

        x = x + self.dropout1(self.mha(x,x,x))
        x = self.bn1(x.view(-1, x.size(-1))).view(*x.size())

        x = x + self.dropout2(self.ffl(x))
        x = self.bn2(x.view(-1, x.size(-1))).view(*x.size())

        return x

class TransformerEncoder(nn.Module):
    def __init__(self, n_layers,  n_head, embedding_dim, d_ff, device, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.n_layers = n_layers
        self.layers = [EncoderLayer(n_head,embedding_dim,d_ff,dropout).to(device) for _ in range(n_layers)]
        self.transformer_encoder = nn.Sequential(*self.layers)
    def forward(self, inputs):
        """

        :param inputs: [batch_size, seq_len, embedding_dim]
        :return: [batch_size, seq_len, embedding_dim]
        """
        inputs = self.transformer_encoder(inputs)
        #[batch_size, seq_len, embedding_dim]
        return inputs


if __name__ == "__main__":
    from utils import init_parameters
    inputs = torch.rand(3,20,2)

    input_dim = 2
    embedding_dim = 64
    n_head = 4
    n_layers = 3
    d_ff = 32


    embed = City_Embedding(input_dim, embedding_dim)
    init_parameters(embed)

    enc = TransformerEncoder(n_layers,n_head,embedding_dim,d_ff, torch.device("cpu"))

    e_i = embed(inputs)
    print(e_i.shape)
    ee_i = enc(e_i)
    print(ee_i.shape)

