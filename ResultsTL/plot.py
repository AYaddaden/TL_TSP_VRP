import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy

def plot_stats(stats_arrays, legends, plot_titles, plot_name, x_name, y_name):
    _, axs = plt.subplots(2, 3, figsize=(15,10), sharey="row")
    for(stats_arrays_, axs_, plot_titles_) in zip(stats_arrays, axs, plot_titles):

      for (stats_array, ax,plot_title ) in zip(stats_arrays_, axs_, plot_titles_):
        ax.plot(stats_array[0])
        ax.plot(stats_array[1])
        ax.legend(legends)
        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)
        ax.set_title(plot_title)

      #axs_.set_title(plot_sttl)
    
    plt.savefig( plot_name + ".png", bbox_inches="tight")
    plt.cla()
    plt.clf()



if __name__ == "__main__":
    plt.rcParams.update({'font.size': 12})
    scratch_20_16 = pd.read_csv("VRP20/16000/Scratch/vrp-20-logs.csv")
    scratch_20_32 = pd.read_csv("VRP20/32000/Scratch/vrp-20-logs.csv")
    scratch_20_64 = pd.read_csv("VRP20/64000/Scratch/vrp-20-logs.csv")

    tl_20_16      = pd.read_csv("VRP20/16000/TL_TNCopy/vrp-20-logs.csv")
    tl_20_32      = pd.read_csv("VRP20/32000/TL_TNCopy/vrp-20-logs.csv")
    tl_20_64      = pd.read_csv("VRP20/64000/TL_TNCopy/vrp-20-logs.csv")
    
    
    scratch_50_16 = pd.read_csv("VRP50/16000/Scratch/vrp-50-logs.csv")
    scratch_50_32 = pd.read_csv("VRP50/32000/Scratch/vrp-50-logs.csv")
    scratch_50_64 = pd.read_csv("VRP50/64000/Scratch/vrp-50-logs.csv")

    tl_50_16      = pd.read_csv("VRP50/16000/TL_TNCopy/vrp-50-logs.csv")
    tl_50_32      = pd.read_csv("VRP50/32000/TL_TNCopy/vrp-50-logs.csv")
    tl_50_64      = pd.read_csv("VRP50/64000/TL_TNCopy/vrp-50-logs.csv")

    
    ttl = [ ["(a) VRP20 - 16k inst./epoch", "(b) VRP20 - 32k inst./epoch", "(c) VRP20 - 64k inst./epoch"],
            ["(d) VRP50 - 16k inst./epoch", "(e) VRP50 - 32k inst./epoch", "(f) VRP50 - 64k inst./epoch"]]
    column = "avg_tl_epoch_train"
    title = "average tour lengths per epoch in training"
    plot_stats([ [[ scratch_20_16[column], tl_20_16[column]],
                  [ scratch_20_32[column], tl_20_32[column]], 
                  [ scratch_20_64[column], tl_20_64[column]]
                 ],
                 [
                  [ scratch_50_16[column], tl_50_16[column]],
                  [ scratch_50_32[column], tl_50_32[column]], 
                  [ scratch_50_64[column], tl_50_64[column]]
                 ]
                  
                ],
               ["NO-TL", "TL"],
               ttl,
               title,
               "epoch", "average tour length")

    column = "avg_tl_epoch_val"
    title = "average tour lengths per epoch in validation"
    plot_stats([ [[ scratch_20_16[column], tl_20_16[column]],
                  [ scratch_20_32[column], tl_20_32[column]], 
                  [ scratch_20_64[column], tl_20_64[column]]
                 ],
                 [
                  [ scratch_50_16[column], tl_50_16[column]],
                  [ scratch_50_32[column], tl_50_32[column]], 
                  [ scratch_50_64[column], tl_50_64[column]]
                 ]
                  
                ],
               ["NO-TL", "TL"],
               ttl,
               title,
               "epoch", "average tour length")





