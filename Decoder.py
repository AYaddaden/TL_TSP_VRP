import torch
import torch.nn as nn
from torch.distributions import Categorical
from Encoder import MultiHeadAttention
from load_data import CAPACITIES
from math import sqrt
from copy import deepcopy

class Decoder(nn.Module):
    """
        This class contains the decoder that will be used to compute the probability distribution from which we will sample
        which city to visit next. The class is indended to work on both TSP and VRP.
    """
    def __init__(self, n_head, embedding_dim, decode_mode="sample", C=10, problem="tsp"):
        super(Decoder, self).__init__()
        self.scale = sqrt(embedding_dim)
        self.decode_mode = decode_mode
        self.C = C
        self.problem = problem

        self.vl = nn.Parameter(torch.FloatTensor(size=[1, 1, embedding_dim]).uniform_(-1./embedding_dim,1./embedding_dim), requires_grad=True)
        self.vf = nn.Parameter(torch.FloatTensor(size=[1, 1, embedding_dim]).uniform_(-1./embedding_dim,1./embedding_dim), requires_grad=True)


        self.glimpse = MultiHeadAttention(n_head, embedding_dim, 3 * embedding_dim, embedding_dim, embedding_dim)
        self.project_k = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.CrossEntropy = nn.CrossEntropyLoss(reduction='none')

    def forward(self, encoded_inputs, device):
        """

        :param (encoded_inputs, None): ([batch_size, seq_len, embedding_dim], None) for TSP
               (encoded_inputs, demands, capacities): ([batch_size, seq_len, embedding_dim],[batch_size, seq_len],[batch_size]) for VRP
        :return: log_prob, solutions
        """

        if self.problem == "tsp":
            encoded_inputs, _, _ = encoded_inputs
        else: # problem is vrp
            encoded_inputs, demendes, caps = encoded_inputs

            demands = deepcopy(demendes)
            capacities = deepcopy(caps)
            capacities = capacities.unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1]

        batch_size, seq_len, embedding_dim = encoded_inputs.size() # sel_len = nb_clients + nb_depot (1)

        h_hat = encoded_inputs.mean(-2, keepdim=True) # [batch_size, 1, embedding_dim]

        h_c = None
        proba= None
        city_index = None

        # case of vrp : mask[:, 0] is the depot
        mask = torch.zeros([batch_size, seq_len], device=device).bool()

        solution = torch.tensor([], dtype=torch.long).to(device)
        if self.problem == "vrp":
            solution = torch.zeros([batch_size,1], dtype=torch.long).to(device) # first node is depot
            mask[:, 0] = True # vehicle is in the depot location
            prev_select_depot = mask[:, 0] == True #[batch_size]


        log_probabilities = torch.zeros(batch_size, dtype=torch.float).to(device)

        if self.problem == "tsp":
            last = self.vl.repeat(batch_size, 1, 1)  # batch_size, 1, embedding_dim
        elif self.problem == "vrp":
            last = encoded_inputs[:,0].unsqueeze(1)

        first = self.vf.repeat(batch_size, 1, 1) # batch_size, 1, embedding_dim

        raw_logits = torch.tensor([]).to(device)
        t = 0 # time steps

        #for t in range(seq_len):
        while torch.sum(mask) < batch_size * seq_len:
            t += 1
            if self.problem == "vrp":
                cap_emb = torch.matmul(capacities,first) # embedding the capacity [batch_size, 1, embedding_size]

                h_c = torch.cat((h_hat, last, cap_emb), dim=-1)  # [batch_size, 1, 3 * embedding_size]

            elif self.problem == "tsp":
                h_c = torch.cat((h_hat, last, first), dim=-1)  # [batch_size, 1, 3 * embedding_size]


            context = self.glimpse(h_c,encoded_inputs, encoded_inputs, mask.unsqueeze(1).unsqueeze(1)) # [batch_size, 1, embedding_size]

            k = self.project_k(encoded_inputs) # [batch_size, seq_len, embedding_size]

            #[batch_size, 1, seq_len]
            u = torch.tanh( torch.matmul(context, k.transpose(-2,-1)) / self.scale ) * self.C

            raw_logits = torch.cat((raw_logits, u), dim=1)  # batch_size, seq_len, seq_len

            u = u.masked_fill(mask.unsqueeze(1), float('-inf'))

            probas = nn.functional.softmax(u.squeeze(1),dim=-1) # [batch_size, seq_len]

            one_hot = torch.zeros([seq_len]).to(device)
            one_hot[0] = 1


            if self.decode_mode == "greedy":
                proba, city_index = self.greedy_decoding(probas)
            elif self.decode_mode == "sample":
                proba, city_index = self.sample_decoding(probas)

            log_probabilities += self.CrossEntropy(u.squeeze(1), city_index.view(-1))

            solution = torch.cat((solution, city_index), dim=1)

            # next node for the query
            last = encoded_inputs[[i for i in range(batch_size)], city_index.view(-1), :].unsqueeze(1)

            #update mask
            if self.problem == 'tsp':
                mask = mask.scatter(1, city_index, True)
                # save log probas and solution
                # log_probabilities += torch.log(proba)

                if t == 1 : # next context components for iteration 1 and above
                    first = last

            elif self.problem == 'vrp':
                # update mask : if capacity is 0,
                #               if sum remaining demands > remaining capacity,
                #               if city is selected
                # update capacity : if depot is selected refill, if client is selected decrease by demand
                # update demands

                #if t == 1:
                #    mask[:, 0] = False # unmask depot after 1st iteration

                is_depot = city_index.view(-1) == 0 # [batch_size]

                capacities = capacities.masked_fill(is_depot.unsqueeze(1).unsqueeze(1) == True, CAPACITIES[seq_len]) # [batch_size, 1, 1]

                selected_demands = demands[[i for i in range(batch_size)], city_index.view(-1)] # [batch_size]

                capacities = capacities - selected_demands.unsqueeze(1).unsqueeze(1)

                demands[[i for i in range(batch_size)], city_index.view(-1)] = 0


                # unmask demand > 0
                mask = mask.masked_fill(demands > 0, False)

                # mask demand > capacity
                exceed_capacity = demands > capacities.squeeze(1)

                mask = mask.masked_fill(exceed_capacity == True, True)

                # mask[:, 1:] demand == 0 (mask selected clients only)
                #mask[:, 1:] = mask[:, 1:].masked_fill(demands[:, 1:] == 0, True)
                mask = mask.scatter(1, city_index, True)

                # if depot was previously selected, unmask it
                mask[:, 0] = mask[:, 0].masked_fill(prev_select_depot == True, False)

                eval  = torch.all(mask, dim=-1) # evaluate if there are completed tours

                eval_all_visited = torch.all(eval, dim=-1)


                if torch.all(prev_select_depot == is_depot, dim=-1) and torch.all(prev_select_depot, dim=-1):
                    break


                if eval_all_visited == False: # there is at least one tour completed and others still in construction
                    # put the constructed tour depot to false so that it can be selected
                    # this is to have solutions with the same dimension
                    for pos, e in enumerate(eval) :
                        if e == True:
                            mask[pos,0] = False

                prev_select_depot = is_depot  # update previously selected depot indicator


        if self.problem == "vrp":
            # make sure the last node is the depot for all instances
            dep = torch.zeros([batch_size,1], dtype=torch.long, device=device)
            solution = torch.cat((solution,dep), dim=1)

            #[batch_size] , [batch_size, seq_len]
        return raw_logits, log_probabilities, solution
    def greedy_decoding(self, probas):
        """
        :param probas: [batch_size, seq_len]
        :return: probas : [batch_size],  city_index: [batch_size,1]
        """
        probas, city_index = torch.max(probas, dim=1)

        return (probas, city_index.view(-1, 1))

    def sample_decoding(self, probas):
        """

        :param probas: [ batch_size, seq_len]
        :return: probas : [batch_size],  city_index: [batch_size,1]
        """
        batch_size = probas.size(0)
        m = Categorical(probas)
        city_index = m.sample()
        probas = probas[[i for i in range(batch_size)], city_index]

        return (probas, city_index.view(-1, 1))

    def set_decode_mode(self, decode_mode):
        #print("Decode mode set to : {}\n".format(decode_mode))
        self.decode_mode = decode_mode

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    def plot_tour(instance, tour):
        "Plot the cities as circles and the tour as lines between them."
        points = instance[tour]
        t = [instance[tour[0]]]
        points = np.concatenate((points, t))
        plot_lines(points)
        for i, t in enumerate(tour):
            plt.annotate(t, points[i])

    def plot_lines(points, style='bo-'):
        plt.plot([p[0] for p in points], [p[1] for p in points], style)
        plt.axis('scaled')
        plt.axis('off')


    torch.manual_seed(1234)
    inp = torch.rand(10,20,2)
    encoder = torch.nn.Linear(2,12)
    enc_inp = encoder(inp)

    cap = torch.tensor([15,15,15,15,15,15,15,15,15,15], dtype=torch.float32)
    dem = torch.FloatTensor(10,20).uniform_(0, 6).int() + 1
    dem[:, 0] = 0

    print("Capacities : {}\n".format(cap[0]))
    print("Demands : {}\n".format(dem[0]))

    n_head = 3
    embedding_dim = 12
    dec = Decoder(n_head,embedding_dim, decode_mode="greedy", problem="vrp")

    all_probabilities, log_prob, sol = dec((enc_inp,dem, cap),device='cpu')


    instance = inp[0].numpy()
    tour = sol[0].numpy()

    plot_tour(instance,tour)
    plt.show()
    print(log_prob)
    print(sol[0])
    #print(all_probabilities)