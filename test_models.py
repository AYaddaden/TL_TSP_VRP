from Model import TSPActor
import torch
import torch.nn as nn
from utils import compute_return_tsp
from load_data import VRPDataset
from torch.utils.data import DataLoader

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
        btl = compute_return_tsp(inputs[0],solution, device)
        tour_lengths =  torch.cat((tour_lengths,btl), dim=0)

    return solution, tour_lengths, tour_lengths.mean().item()




if __name__ == "__main__":
    graph_size = 20

    batch_size = 100
    nb_test_samples = 1000

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

    # model from scratch, no embedding demand
    fs_ckpt_nodem = torch.load("Scratach_NOEMBDEM_1e-4/RL_vrp20_Epoch_5.pt")
    fs_nodem = TSPActor(embedding_dim,n_layers,n_heads,d_ff,decode_mode,device,C,dropout,problem,embed_demand=False)
    fs_nodem.load_state_dict(fs_ckpt_nodem["model"])

    # model from scrach, with embedding demand
    fs_ckpt_dem = torch.load("Scratch_EMBEDDEM_1e-4/RL_vrp20_Epoch_5.pt")
    fs_dem = TSPActor(embedding_dim, n_layers, n_heads, d_ff, decode_mode, device, C, dropout, problem,
                        embed_demand=True)
    fs_dem.load_state_dict(fs_ckpt_dem["model"])

    # model trained using RL_TSP20 with target net being the TSP target net
    tl_ckpt = torch.load("TL_TSP20_VRP20_NOEMBCIT_1e-5/RL_vrp20_Epoch_5.pt")
    tl_model = TSPActor(embedding_dim, n_layers, n_heads, d_ff, decode_mode, device, C, dropout, problem,
                        embed_demand=False)
    tl_model.load_state_dict(tl_ckpt["model"])

    # model train using RL_TSP20 with target net being a copy of model network
    tl_tn_ckpt = torch.load("RL_vrp20_Epoch_5.pt")
    tl_tn_model = TSPActor(embedding_dim, n_layers, n_heads, d_ff, decode_mode, device, C, dropout, problem,
                        embed_demand=False)
    tl_tn_model.load_state_dict(tl_tn_ckpt["model"])

    random_model = TSPActor(embedding_dim, n_layers, n_heads, d_ff, decode_mode, device, C, dropout, problem,
             embed_demand=False)

    models = [fs_nodem,fs_dem,tl_model,tl_tn_model, random_model]

    tour_lengths = []

    test_dataset = VRPDataset(size=graph_size, num_samples=nb_test_samples)

    test_dataloader = DataLoader(test_dataset, batch_size,shuffle=False)

    for model  in models:
        tl = test_model(model, test_dataloader,device)
        tour_lengths.append(tl)

    print('Tour lengths per model\nFS_NOEMB: {}\nFS_EMB: {}\nTL : {}\nTL tn: {}\nRandom model: {}'.
          format(tour_lengths[0], tour_lengths[1], tour_lengths[2], tour_lengths[3], tour_lengths[4]))

