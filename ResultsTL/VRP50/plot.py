import matplotlib.pyplot as plt
import pandas as pd

def plot_stats(stats_arrays, legends, plot_titles, plot_name, x_name, y_name):
    _, axs = plt.subplots(1, 3, figsize=(15,5))
    
    for (stats_array, ax, plot_title) in zip(stats_arrays, axs, plot_titles):
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

    scratch_50_16 = pd.read_csv("16000/Scratch/vrp-50-logs.csv")
    scratch_50_32 = pd.read_csv("32000/Scratch/vrp-50-logs.csv")
    scratch_50_64 = pd.read_csv("64000/Scratch/vrp-50-logs.csv")

    tl_50_16      = pd.read_csv("16000/TL_TNCopy/vrp-50-logs.csv")
    tl_50_32      = pd.read_csv("32000/TL_TNCopy/vrp-50-logs.csv")
    tl_50_64      = pd.read_csv("64000/TL_TNCopy/vrp-50-logs.csv")
    
    ttl = ["16k inst./epoch", "32k inst./epoch", "64k inst./epoch"]
    column = "avg_tl_epoch_train"
    title = "VRP50 - average tour lengths per epoch in training"
    plot_stats([ [ scratch_50_16[column], tl_50_16[column]],
                 [ scratch_50_32[column], tl_50_32[column]], 
                 [ scratch_50_64[column], tl_50_64[column]]
                ],
               ["NO-TL", "TL"],
               ttl,
               title,
               "epoch", "average tour length")

    column = "avg_tl_epoch_val"
    title = "VRP50 - average tour lengths per epoch in validation"
    plot_stats([ [ scratch_50_16[column], tl_50_16[column]],
                 [ scratch_50_32[column], tl_50_32[column]], 
                 [ scratch_50_64[column], tl_50_64[column]]
                ],
               ["NO-TL", "TL"],
               ttl,
               title,
               "epoch", "average tour length")





