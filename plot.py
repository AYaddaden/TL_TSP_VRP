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
    TL_results = pd.read_csv("TL_TSP20_VRP20_NOEMBCIT_1e-5/vrp-20-logs.csv")
    TL_results_tn = pd.read_csv("vrp-20-logs.csv")
    Scratch_results_noembdem = pd.read_csv("Scratach_NOEMBDEM_1e-4/vrp-20-logs.csv")
    Scratch_results_embdem = pd.read_csv("Scratch_EMBEDDEM_1e-4/vrp-20-logs.csv")
    plot_stats([TL_results["avg_tl_epoch_val"],
                TL_results_tn["avg_tl_epoch_val"],
                Scratch_results_noembdem["avg_tl_epoch_val"],
                Scratch_results_embdem["avg_tl_epoch_val"]],
               ["TL", "TL_TN","FS_NOEMB", "FS_EMB"],
               "comparision between average tour lengths per epoch in validation",
               "epoch", "average tour length")





