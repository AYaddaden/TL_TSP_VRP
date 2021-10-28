import matplotlib.pyplot as plt
import pandas as pd

def plot_stats(stats_arrays, legends, plot_titles, plot_name, x_name, y_name):
    _, axs = plt.subplots(1, 2, figsize=(13,5))
    
    for (stats_array, ax,plot_title ) in zip(stats_arrays, axs, plot_titles):
      ax.plot(stats_array[0])
      ax.plot(stats_array[1])
      ax.legend(legends)
      ax.set_xlabel(x_name)
      ax.set_ylabel(y_name)
      ax.set_title(plot_title)

    #plt.title(plot_name)
    plt.savefig( plot_name + ".png", bbox_inches="tight")
    plt.cla()
    plt.clf()



if __name__ == "__main__":

    plt.rcParams.update({'font.size': 15})

    scratch_normal_vrp20 = pd.read_csv("normal_scratch_vrp20/vrp-20-logs.csv")
    tl_normal_vrp20      = pd.read_csv("normal_tl_vrp20/vrp-20-logs.csv")
    
    
    scratch_normal_vrp50 = pd.read_csv("normal_scratch_vrp50/vrp-50-logs.csv")
    tl_normal_vrp50      = pd.read_csv("normal_tl_vrp50/vrp-50-logs.csv") 

    
    ttl = ["(a) VRP20", "(b) VRP50"]
    column_train = "avg_tl_epoch_train"
    column_val = "avg_tl_epoch_val"
    title = "average tour length with normal distribution val"
    plot_stats([ [ scratch_normal_vrp20[column_val], tl_normal_vrp20[column_val]],
                 [ scratch_normal_vrp50[column_val], tl_normal_vrp50[column_val]]
               ],
               ["NO-TL", "TL"],
               ttl,
               title,
               "epoch", "average tour length")





