import torch
import torch.nn as nn
from utils import compute_return_vrp

def test_model(model : nn.Module , data, device):
    """

    :param model:
    :param data:
    :return: avg_tl : average tour length
    """

    tour_lengths = torch.tensor([], device=device)
    model.eval()
    model.set_decode_mode("greedy")
    for batch_id, batch in enumerate(data):
        locations, demands, capacities = batch
        locations, demands, capacities = locations.to(device), demands.to(device), capacities.to(device)

        inputs = (locations, demands, capacities.float())
        _, _, solution = model(inputs)
        btl = compute_return_vrp(inputs[0],solution, device)
        tour_lengths =  torch.cat((tour_lengths,btl), dim=0)

    return solution, tour_lengths, tour_lengths.mean().item()


from Model import TSPActor
import pandas as pd
if __name__ == "__main__":
    seed = 1234
    torch.manual_seed(seed)
    vrp20 = pd.read_csv("VRP-20.csv")
    x = torch.tensor(vrp20["x"].values).unsqueeze(1)
    y = torch.tensor(vrp20["y"].values).unsqueeze(1)

    locations = torch.cat((x,y), dim=1).unsqueeze(0)
    demands = torch.tensor(vrp20["d"].values).unsqueeze(0)
    capacity = torch.tensor([30.])
    print(locations.shape)
    print(demands.shape)
    print(capacity.shape)

    data = (locations,demands,capacity)
    n_layers = 3
    n_heads = 8
    embedding_dim = 128
    d_ff = 512
    decode_mode = "greedy"
    C = 10
    dropout = 0.1
    device = torch.device("cpu")
    learning_rate = 1e-4
    seed = 1234
    freq_save = 10
    torch.manual_seed(seed)
    freq_log = 10
    problem = "vrp"
    embed_demand = False
    vrp_model = TSPActor(embedding_dim, n_layers, n_heads, d_ff, decode_mode, device, C, dropout, problem, embed_demand)

    checkpoint = torch.load("RL_vrp20_Epoch_50.pt", map_location=device)
    vrp_model.load_state_dict(checkpoint["model"])
    vrp_model.eval()

    sol, tour_length, _ = test_model(vrp_model,data, device)

    print(sol)
    print(tour_length)



