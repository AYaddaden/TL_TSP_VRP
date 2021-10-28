import torch
import torch.nn as nn
from Decoder import Decoder
from Encoder import TransformerEncoder, City_Embedding, Demand_Embedding

class TSPActor(nn.Module):
    def __init__(self, embedding_dim, n_layers, n_head, d_ff, decode_mode, device, C, dropout=0.1, problem="tsp", embed_demand=True):
        super(TSPActor, self).__init__()
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.n_head = n_head
        self.d_ff = d_ff
        self.decode_mode = decode_mode
        self.device = device
        self.dropout = dropout
        self.C = C
        self.input_dim = 2

        self.problem = problem
        self.embed_demand = embed_demand

        self.demand_embedding = nn.Identity() # for the TSP case
        if self.problem == "vrp":
            if self.embed_demand == True:
                self.demand_embedding = Demand_Embedding(1, self.embedding_dim)

        self.city_embedding = City_Embedding(self.input_dim, self.embedding_dim)

        self.encoder = TransformerEncoder(self.n_layers, self.n_head, self.embedding_dim,
                                          self.d_ff, self.device, self.dropout)
        self.decoder = Decoder(self.n_head, self.embedding_dim, self.decode_mode, self.C, self.problem)

    def forward(self, inputs):
        """

        :param inputs: [batch_size, seq_len, input_dim] for TSP
               inputs : (locations, demands,capacities)
               (locations : [batch_size, seq_len, input_dim],
                demands : [batch_size, seq_len, 1],
                capacities : [batch_size]) for VRP

        :return: raw_logits : [batch_size, seq_len, seq_len],
                 log_prob : [batch_size],
                 solutions : [batch_size, seq_len]
        """


        demands = None
        capacities = None
        dem = None
        if self.problem == "vrp":
            inputs, demands, capacities = inputs
            if self.embed_demand == True :
                dem = demands.unsqueeze(-1)
            else:
                dem = torch.zeros(1).to(self.device)
        elif self.problem == "tsp":
            dem = torch.zeros(1).to(self.device)


        demands_embedding = self.demand_embedding(dem)  # [batch_size, seq_len, embedding_size]

        x = self.city_embedding(inputs) # [batch_size, seq_len, embedding_dim]

        x = x + demands_embedding

        x = self.encoder(x) # [batch_size, seq_len, embedding_dim]

        inp = (x, demands, capacities)
        raw_logits, log_prob, solution = self.decoder(inp, self.device)

        return raw_logits, log_prob, solution

    def set_decode_mode(self, decode_mode):
        self.decoder.set_decode_mode(decode_mode)


