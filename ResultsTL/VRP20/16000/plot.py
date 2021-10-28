import matplotlib.pyplot as plt
import pandas as pd

def plot_stats(stats_arrays, legends, plot_name, x_name, y_name):
    for stats_array in stats_arrays:
        plt.plot(stats_array)
    plt.legend(legends)
    plt.xlabel(x_name)
    plt.ylabel(y_name)

    plt.title(plot_name)
    plt.savefig( plot_name + ".png")
    plt.cla()
    plt.clf()



if __name__ == "__main__":
    Scratch = pd.read_csv("Scratch/vrp-20-logs.csv")
    TL_TNCopy = pd.read_csv("TL_TNCopy/vrp-20-logs.csv")
    #Scratch_results_noembdem = pd.read_csv("Scratach_NOEMBDEM_1e-4/vrp-20-logs.csv")
    #Scratch_results_embdem = pd.read_csv("Scratch_EMBEDDEM_1e-4/vrp-20-logs.csv")
    plot_stats([TL_TNCopy["avg_tl_epoch_train"],
                Scratch["avg_tl_epoch_train"]],
               ["TL", "Scratch"],
               "comparision between average tour lengths per epoch in train",
               "epoch", "average tour length")





