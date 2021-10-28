import torch
from torch.optim import Adam
from Model import TSPActor
from utils import compute_return_tsp, compute_return_vrp
from load_data import TSPDataset, SupervisedDataloader, VRPDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.stats import ttest_rel
import numpy as np
import matplotlib.pyplot as plt
import csv

class Trainer:
    def __init__(self, problem, graph_size, n_epochs, batch_size, nb_train_samples,
                 nb_val_samples, n_layers, n_heads, embedding_dim,
                 d_ff, decode_mode, C, dropout, device, learning_rate, freq_save, freq_log, embed_demand
                 ):
        self.problem = problem
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.graph_size = graph_size
        self.nb_train_samples = nb_train_samples
        self.nb_val_samples = nb_val_samples
        self.device = device
        self.freq_save = freq_save
        self.freq_log  = freq_log

        self.model = TSPActor(embedding_dim,n_layers,n_heads,
                              d_ff, decode_mode, device, C, dropout, problem, embed_demand) # embedding, encoder, decoder

        self.model.to(device)

        self.target_net = TSPActor(embedding_dim, n_layers, n_heads,
                                   d_ff, decode_mode="greedy", device=device,
                                   C=C, dropout=dropout, problem=problem, embed_demand=embed_demand)
        self.target_net.to(device)

        self.target_net.load_state_dict(self.model.state_dict())

        self.optimizer = Adam(self.model.parameters(),lr=learning_rate)

        log_file_name = "{}-{}-logs.csv".format(problem,graph_size)

        f = open(log_file_name, 'w', newline='')
        self.log_file = csv.writer(f, delimiter=",")

        header = ["epoch", "losses_per_batch", "avg_tl_batch_train", "avg_tl_epoch_train", "avg_tl_epoch_val"]
        self.log_file.writerow(header)

    def rl_train(self):
        if self.problem == "tsp":
            validation_dataset = TSPDataset(size=self.graph_size, num_samples=self.nb_val_samples)
        elif self.problem == "vrp":
            validation_dataset = VRPDataset(size=self.graph_size, num_samples=self.nb_val_samples)

        validation_dataloader = DataLoader(validation_dataset, batch_size=self.batch_size, shuffle=True,
                                           num_workers=0)
        losses = []
        avg_tour_length_batch = []
        avg_tour_length_epoch = []
        avg_tl_epoch_val = []
        for epoch in range(self.n_epochs):

            all_tour_lengths = torch.tensor([], dtype=torch.float32).to(self.device)

            self.model.set_decode_mode("sample")
            self.model.train()

            if self.problem == "tsp":
                train_dataset = TSPDataset(size=self.graph_size, num_samples=self.nb_train_samples)
            elif self.problem == "vrp":
                train_dataset = VRPDataset(size=self.graph_size, num_samples=self.nb_train_samples)

            train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
            nb_batches = len(train_dataloader)


            for batch_id, batch in enumerate(tqdm(train_dataloader)):
                if self.problem == "tsp":
                    inputs = batch.to(self.device)
                elif self.problem == "vrp":
                    locations, demands, capacities = batch
                    locations, demands, capacities = locations.to(self.device), demands.to(self.device), capacities.to(self.device)

                    inputs = (locations, demands, capacities.float())

                _, log_prob, solution = self.model(inputs)


                with torch.no_grad():
                    _, _, rollout_sol = self.target_net(inputs)

                    if self.problem == "tsp":
                        tour_lengths = compute_return_tsp(inputs, solution, self.device)
                        baseline_tour_lengths = compute_return_tsp(inputs, rollout_sol, self.device)
                    elif self.problem == 'vrp':
                        tour_lengths = compute_return_vrp(inputs[0], solution, self.device)
                        baseline_tour_lengths = compute_return_vrp(inputs[0], rollout_sol, self.device)

                    advantage = tour_lengths - baseline_tour_lengths



                loss = advantage * (-log_prob)
                loss = loss.mean()
                if batch_id % self.freq_log == 0:
                    print("\nEpoch: {}\tBatch: {}\nLoss: {}\nAverage tour length model : {}\nAverage tour length baseline : {}\n".format(
                        epoch, batch_id, loss.item(), tour_lengths.mean(), baseline_tour_lengths.mean()
                    ))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # save data for plot
                losses.append(loss.item())
                avg_tour_length_batch.append(tour_lengths.mean().item())
                all_tour_lengths = torch.cat((all_tour_lengths,tour_lengths), dim=0)

            avg_tour_length_epoch.append(all_tour_lengths.mean().item())

            print("Validation and rollout update check\n")

            # t-test :
            self.model.set_decode_mode("greedy")
            self.target_net.set_decode_mode("greedy")
            self.model.eval()
            self.target_net.eval()
            with torch.no_grad():
                rollout_tl = torch.tensor([], dtype=torch.float32)
                policy_tl  = torch.tensor([], dtype=torch.float32)

                for batch_id, batch in enumerate(tqdm(validation_dataloader)):
                    if self.problem == "tsp":
                        inputs = batch.to(self.device)
                    elif self.problem == "vrp":
                        locations, demands, capacities = batch
                        locations, demands, capacities = locations.to(self.device), demands.to(
                            self.device), capacities.to(self.device)

                        inputs = (locations, demands, capacities.float())

                    _, _, solution  = self.model(inputs)
                    _, _, rollout_sol = self.target_net(inputs)

                    if self.problem == 'tsp':
                        tour_lengths = compute_return_tsp(inputs, solution, self.device)
                        baseline_tour_lengths = compute_return_tsp(inputs, rollout_sol, self.device)
                    elif self.problem == 'vrp':
                        tour_lengths = compute_return_vrp(inputs[0], solution, self.device)
                        baseline_tour_lengths = compute_return_vrp(inputs[0], rollout_sol, self.device)

                    rollout_tl = torch.cat((rollout_tl,baseline_tour_lengths.view(-1).cpu()), dim=0)
                    policy_tl  = torch.cat((policy_tl, tour_lengths.view(-1).cpu()), dim=0)

                rollout_tl = rollout_tl.cpu().numpy()
                policy_tl = policy_tl.cpu().numpy()

                avg_ptl = np.mean(policy_tl)
                avg_rtl = np.mean(rollout_tl)

                avg_tl_epoch_val.append(avg_ptl.item())

                print("Average tour length by policy: {}\nAverage tour length by rollout: {}\n".format(avg_ptl,avg_rtl))

                self.log_file.writerow([epoch, losses[-nb_batches:],
                                        avg_tour_length_batch[-nb_batches:],
                                        avg_tour_length_epoch[-1],
                                        avg_ptl.item()
                                        ])

                if (avg_ptl - avg_rtl) < 0:
                    # t-test
                    _, pvalue = ttest_rel(policy_tl, rollout_tl)
                    pvalue = pvalue / 2 # one-sided ttest [refer to the original implementation]
                    if pvalue < 0.05:
                        print("Rollout network update...\n")
                        self.target_net.load_state_dict(self.model.state_dict())
                        print("Generate new vlaidation dataset\n")

                        if self.problem == "tsp":
                            validation_dataset = TSPDataset(size=self.graph_size, num_samples=self.nb_val_samples)
                        elif self.problem == "vrp":
                            validation_dataset = VRPDataset(size=self.graph_size, num_samples=self.nb_val_samples)

                        validation_dataloader = DataLoader(validation_dataset, batch_size=self.batch_size, shuffle=True,
                                                           num_workers=0)



            if epoch % self.freq_save == 0:
                model_name = "RL_{}{}_Epoch_{}.pt".format(self.problem, self.graph_size,epoch)
                self.save_model(epoch,model_name)

        model_name = "RL_{}{}_Epoch_{}.pt".format(self.problem, self.graph_size,self.n_epochs)
        self.save_model(self.n_epochs, model_name)

        # plot at the end of training
        self.plot_stats(losses,"{}-RL-Losses per batch {}".format(self.problem,self.graph_size),"Batch", "Loss")
        self.plot_stats(avg_tour_length_epoch, "{}-RL-Average tour length per epoch train {}".format(self.problem,self.graph_size), "Epoch", "Average tour length")
        self.plot_stats(avg_tour_length_batch,  "{}-RL-Average tour length per batch train {}".format(self.problem,self.graph_size), "Batch", "Average tour length")
        self.plot_stats(avg_tl_epoch_val, "{}-RL-Average tour length per epoch validation {}".format(self.problem,self.graph_size), "Epoch", "Average tour length")

    def sl_train(self, train_dataset, test_dataset):
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')

        # load train dataset
        supervised_dataset_train = SupervisedDataloader(train_dataset)
        supervised_dataloader_train = DataLoader(supervised_dataset_train, batch_size=self.batch_size,shuffle=False, num_workers=0)

        # load test dataset
        supervised_dataset_test = SupervisedDataloader(test_dataset)
        supervised_dataloader_test = DataLoader(supervised_dataset_test, batch_size=self.batch_size, shuffle=False,
                                                 num_workers=0)

        losses = []
        avg_tour_length_batch = []
        avg_tour_length_epoch = []
        avg_tour_length_epoch_test = []
        avg_tour_length_batch_test = []

        opt_gap = []
        self.model.set_decode_mode("greedy")
        for epoch in range(self.n_epochs):
            all_tour_lengths = torch.tensor([], dtype=torch.float32).to(self.device)

            for batch_id, batch in enumerate(tqdm(supervised_dataloader_train)):
                inputs, labels = batch

                inputs, labels = inputs.to(device), labels.to(device)

                num_cities = inputs.size(1)

                raw_logits, _, solution = self.model(inputs)

                loss = criterion(raw_logits.contiguous().view(-1, num_cities), labels.view(-1))

                tour_lengths = compute_return_tsp(inputs,solution, self.device)

                if batch_id % self.freq_log == 0:
                    print("\nEpoch: {}\tBatch: {}\nLoss: {}\nAverage tour length : {}".format(
                        epoch, batch_id, loss.item(), tour_lengths.mean()
                    ))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


                losses.append(loss.item())

                avg_tour_length_batch.append(tour_lengths.mean())

                all_tour_lengths = torch.cat((all_tour_lengths, tour_lengths), dim=0)

            avg_tour_length_epoch.append(all_tour_lengths.mean())

            print("Validation...\n")
            all_tour_lengths_test = torch.tensor([], dtype=torch.float32).to(self.device)
            for batch_id, batch in enumerate(tqdm(supervised_dataloader_test)):
                inputs, labels = batch

                inputs, labels = inputs.to(device), labels.to(device)

                _, _, solution = self.model(inputs)

                tour_length = compute_return_tsp(inputs, solution, self.device)

                optim_tour_length = compute_return_tsp(inputs, labels, self.device)

                print("\noptim: {}\nheuristic: {}\n".
                      format(optim_tour_length.mean(), tour_length.mean()))

                avg_tour_length_batch_test.append(tour_length.mean())
                all_tour_lengths_test = torch.cat((all_tour_lengths_test, tour_length), dim=0)
                opt_gap.append(100 * ((tour_length.mean() / optim_tour_length.mean())-1))
            avg_tour_length_epoch_test.append(all_tour_lengths_test.mean())

            if epoch % self.freq_save == 0:
                model_name = "SL_TSP_Epoch_" + str(epoch) + ".pt"
                self.save_model(epoch, model_name)

        # save last model
        model_name = "SL_TSP_Epoch_" + str(n_epochs) + ".pt"
        self.save_model(n_epochs, model_name)

        # plot at the end of training
        self.plot_stats(losses, "Losses per batch train", "Batch", "Loss")
        self.plot_stats(avg_tour_length_epoch, "Average tour length per epoch train", "Epoch", "Average tour length")
        self.plot_stats(avg_tour_length_batch, "Average tour length per batch train", "Batch", "Average tour length")
        self.plot_stats(opt_gap, "Validation Optimality gap (%)", "Batch", "Opt gap (%)")

        self.plot_stats(avg_tour_length_epoch_test, "Average tour length per epoch test", "Epoch", "Average tour length")
        self.plot_stats(avg_tour_length_batch_test, "Average tour length per batch test", "Batch", "Average tour length")

    def plot_stats(self, stats_array, plot_name, x_name, y_name):
        plt.plot(stats_array)
        plt.xlabel(x_name)
        plt.ylabel(y_name)
        plt.title(plot_name)
        plt.savefig(plot_name+".png")
        plt.cla()
        plt.clf()

    def save_model(self, epoch, model_name):
        torch.save({
            "epoch": epoch,
            "model": self.model.state_dict(),
            "baseline": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }, model_name)

    def plot_tour(self, instance, tour):
        "Plot the cities as circles and the tour as lines between them."
        points = instance[tour]
        t = [instance[tour[0]]]
        points = np.concatenate((points, t))
        self.plot_lines(points)
        for i, t in enumerate(tour):
            plt.annotate(t, points[i])

    def plot_lines(self, points, style='bo-'):
        plt.plot([p[0] for p in points], [p[1] for p in points], style)
        plt.axis('scaled');
        plt.axis('off')


from utils import freeze_parameters

if __name__ == "__main__":
    graph_size = 20
    n_epochs =  50
    batch_size = 512
    nb_train_samples = 32000
    nb_val_samples = 10000
    n_layers = 3
    n_heads = 8
    embedding_dim = 128
    d_ff = 512
    decode_mode = "sample"
    C = 10
    dropout = 0.1
    device = torch.device("cuda:0")
    learning_rate = 1e-5
    seed = 1234
    freq_save = 10
    torch.manual_seed(seed)
    freq_log = 10
    problem = "vrp"
    embed_demand = False
    torch.autograd.set_detect_anomaly(True)

    pretrained_tsp = "RL_TSP50_Epoch_100.pt"

    checkpoint = torch.load(pretrained_tsp)

    trainer = Trainer(problem,graph_size,n_epochs,batch_size,nb_train_samples,nb_val_samples,
                      n_layers,n_heads,embedding_dim,d_ff,decode_mode, C,
                      dropout,device,learning_rate, freq_save, freq_log, embed_demand)



    trainer.model.load_state_dict(checkpoint["model"], strict=False)
    trainer.target_net.load_state_dict(trainer.model.state_dict())
    trainer.optimizer.load_state_dict(checkpoint["optimizer"])

    #freeze_parameters(trainer.model.city_embedding)
    #freeze_parameters(trainer.model.encoder)

    trainer.rl_train()

    #trainer.sl_train(train_dataset="TSP-size-20-len-1000-train.txt",
    #                test_dataset="TSP-size-20-len-100-test.txt")
