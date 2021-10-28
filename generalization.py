import torch
from Model import TSPActor
from utils import compute_return_vrp
from load_data import VRPDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv

if __name__ == "__main__":
    batch_size = 500
    nb_test_samples = 3
    problem = "vrp"
    device = torch.device("cpu")

    # test with different distributions
    seeds = [1234, 5678]
    # test with muliple graph sizes
    graph_sizes = [10, 20, 30, 40, 50]
    #list of models to test
    models = ["/media/ali/Media/ResultsTL/VRP20/64000/TL_TNCopy/RL_vrp20_Epoch_50.pt"]


    # model params
    n_layers = 3
    n_heads = 8
    embedding_dim = 128
    d_ff = 512
    decode_mode = "greedy"
    C = 10
    dropout = 0.1
    embed_demand = False

    vrp_model = TSPActor(embedding_dim,n_layers,n_heads,d_ff,decode_mode,device,C,dropout,problem,embed_demand)
    vrp_model.eval()

    log_file_name = "vrp_generalization"
    f = open(log_file_name, 'w', newline='')
    log_file = csv.writer(f, delimiter=",")

    header = ["distribution", "model_name", "graph_size", "avg_tl_test"]
    log_file.writerow(header)

    for seed in seeds:
        torch.manual_seed(seed)
        for graph_size in graph_sizes:

            test_dataset = VRPDataset(size=graph_size, num_samples=nb_test_samples)
            test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=0)

            for model_name in models:
                checkpoint = torch.load(model_name, map_location=device)
                vrp_model.load_state_dict(checkpoint["model"], strict=False)
                tl = torch.zeros([], device=device)

                for batch_id, batch in enumerate(tqdm(test_dataloader)):
                    locations, demands, capacities = batch
                    locations, demands, capacities = locations.to(device), demands.to(device), capacities.to(device)

                    inputs = (locations, demands, capacities.float())

                    _, _, solution = vrp_model(inputs)
                    print("Solution : {}\nDemands: {}\n Capacities: {}\n".format(solution, demands, capacities))
                    tour_lengths = compute_return_vrp(inputs[0], solution, device)

                    tl = torch.cat((tl, tour_lengths), dim=0)
                avg_tl = tl.mean()

                log_file.writerow([seed, model_name, graph_size, avg_tl.item()])


