import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

def plot_stats(stats_arrays, delims, legends, plot_titles, plot_name, x_name, y_name):
    _, ((ax0,ax1),(ax2,ax3)) = plt.subplots(2, 2, figsize=(13,5), sharex=True)
    axs = ((ax0,ax2),(ax1,ax3))
    for (stats_array, ax,plot_title, delim ) in zip(stats_arrays, axs, plot_titles, delims):
      ax[0].plot(stats_array[0])
      ax[0].plot(stats_array[1])
      
      ax[1].plot(stats_array[0])
      ax[1].plot(stats_array[1])

      ax[0].set_ylim(delim[0])
      ax[1].set_ylim(delim[1] )
      
      # 0 : top , 1 : bottom
      
      ax[0].spines['bottom'].set_visible(False)
      ax[1].spines['top'].set_visible(False)
      ax[0].xaxis.tick_top()
      ax[0].tick_params(labeltop=False)  # don't put tick labels at the top
      ax[0].yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))
      ax[1].xaxis.tick_bottom()
       
      ax[0].legend(legends)
      ax[1].set_xlabel(x_name)
      ax[0].set_ylabel(y_name)
      ax[0].set_title(plot_title)

    #plt.title(plot_name)
    plt.savefig( plot_name + ".png", bbox_inches="tight")
    plt.cla()
    plt.clf()



if __name__ == "__main__":

    plt.rcParams.update({'font.size': 13})

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
               [
                 [(7.5, 9.),(6.0, 7.5)],
                 [(24.5, 26.5),(11.5, 16.)]
               ]
               ,
               ["NO-TL", "TL"],
               ttl,
               title,
               "epoch", "average tour length")





