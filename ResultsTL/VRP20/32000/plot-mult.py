import matplotlib.pyplot as plt
import pandas as pd

def plot_stats(stats_arrays, legends, plot_titles, plot_name, x_name, y_name):
    _, axs = plt.subplots(1, 2, figsize=(10,5))
    
    for (stats_array, ax,plot_title ) in zip(stats_arrays, axs, plot_titles):
      ax.plot(stats_array[0])
      ax.plot(stats_array[1])
      ax.plot(stats_array[2])
      ax.legend(legends)
      ax.set_xlabel(x_name)
      ax.set_ylabel(y_name)
      ax.set_title(plot_title)

    #plt.title(plot_name)
    plt.savefig( plot_name + ".png", bbox_inches="tight")
    plt.cla()
    plt.clf()



if __name__ == "__main__":

    scratch = pd.read_csv("Scratch/vrp-20-logs.csv")
    tl_tsp20      = pd.read_csv("TL_TNCopy/vrp-20-logs.csv")
    tl_tsp50      = pd.read_csv("TL_TSP50/vrp-20-logs.csv")
    ttl = ["(a) training", "(b) validation"]
    column_train = "avg_tl_epoch_train"
    column_val = "avg_tl_epoch_val"
    title = "VRP20 using TSP50"
    plot_stats([ [ scratch[column_train], tl_tsp20[column_train], tl_tsp50[column_train]],
                 [ scratch[column_val], tl_tsp20[column_val], tl_tsp50[column_val]]
               ],
               ["NO-TL", "TL-TSP20", "TL-TSP50"],
               ttl,
               title,
               "epoch", "average tour length")





