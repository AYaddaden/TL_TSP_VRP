from torch.utils.data import Dataset
import torch
from tqdm import tqdm
import numpy as np

class TSPDataset(Dataset):

    def __init__(self, size=20, num_samples=1000, random_seed=1234):
        super(TSPDataset, self).__init__()

        self.data_set = [torch.FloatTensor(size, 2).uniform_(0, 1) for _ in range(num_samples)]
        self.size = len(self.data_set)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data_set[idx]


class SupervisedDataloader(Dataset):
    def __init__(self, path):
        super(SupervisedDataloader, self).__init__()
        self.dataset = []
        x, y = [], []
        with open(path) as f:
            for l in tqdm(f):
                inputs, outputs = l.split(' output ')
                x = np.array(inputs.split(), dtype=np.float32).reshape([-1, 2])
                y = np.array(outputs.split(), dtype=np.int64)
                self.dataset.append((x, y))
        self.size = len(self.dataset)


    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.dataset[idx]

CAPACITIES = {
             5: 15.,
            10: 20.,
            20: 30.,
            40: 40.,
            50: 40.,
           100: 50.
}


class VRPDataset(Dataset):
    # code copied from https://github.com/wouterkool/attention-learn-to-route/blob/master/problems/vrp/problem_vrp.py
    def __init__(self, size, num_samples, distribution="uniform"):

        self.graph_size = size
        # locations[0] is the depot, locations[1:] are the clients
        self.locations = []

        # depot demand : demands[0]= 0, client demands : demands[1:] uniform(1,9)
        self.demands = []

        for i in range(num_samples):
            if distribution == "uniform":
                self.locations.append(torch.FloatTensor(size,2).uniform_(0,1))
            elif distribution == "normal":
                # truncated normal distribution
                # https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/12
                self.locations.append(torch.fmod(torch.randn(size,2), 0.5) + 0.5)

            self.demands.append(torch.randint(low=0, high=9, size=(size,), dtype=torch.float32) + 1.0)
            self.demands[-1][0] = 0


        self.size = len(self.locations)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return (self.locations[idx],self.demands[idx], CAPACITIES[self.graph_size])

from torch.utils.data import DataLoader
import csv
if __name__ == "__main__":
    #dataloader = SupervisedDataloader("TSP-size-20-len-3-train.txt")
    torch.manual_seed(1234)
    #log_file_name = "VRP-50.csv"
    #f = open(log_file_name, 'w', newline='')
    #log_file = csv.writer(f, delimiter=",")

    #header = ["x", "y", "d"]
    #log_file.writerow(header)

    dataloader = VRPDataset(size=50,num_samples=1)
    dataloader = DataLoader(dataloader, batch_size=2)
    for b in dataloader:
        print(b[0].shape)
        print(b[1].shape)
        print(b[2].shape)
    """
    for b in dataloader:
        xy = b[0][0]
        dem = b[1][0]
        #print(dem.shape)
        size = xy.size(0)
        for c in range(size):
           log_file.writerow([xy[c][0].item(), xy[c][1].item(), dem[c].item()])
    """