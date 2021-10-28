import matplotlib.pyplot as plt
import pandas as pd

def plot_stats(stats_arrays, legends, plot_name, x_name, y_name):
    for stats_array in stats_arrays:
        plt.plot(stats_array)
    plt.legend(legends)
    plt.xlabel(x_name)
    plt.ylabel(y_name)

    #plt.title(plot_name)
    plt.savefig( plot_name + ".png")
    plt.cla()
    plt.clf()



if __name__ == "__main__":
    Scratch = pd.read_csv("Scratch/vrp-50-logs.csv")
    TL_TNCopy = pd.read_csv("TL_TNCopy/vrp-50-logs.csv")
    TL_TSP20 = pd.read_csv("TL_TSP20/vrp-50-logs.csv")

    column = "avg_tl_epoch_val"
    title  = "comparision between average tour lengths per epoch in validation using pretrained models on different size"
    plot_stats([TL_TNCopy[column],
                TL_TSP20[column],
                Scratch[column]],
               ["TL TSP50", "TL TSP20" ,"NO-TL"],
               title,
               "epoch", "average tour length")





