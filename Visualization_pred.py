import datashader as ds
import matplotlib.pyplot as plt
import pandas as pd
from datashader.mpl_ext import dsshow
import gc

# res = 19
# model = 'HG'

# # load csv
# centroid_df = pd.read_csv('../Result/NB_centPredict_{}.csv'.format(res), sep=',')

# # plot
# fig, ax = plt.subplots()
# artist = dsshow(centroid_df, ds.Point('lon_c', 'lat_c'), aggregator=ds.mean(model), cmap='gray', plot_width=300, plot_height=300, ax=ax)

# ax.set_xticklabels([])
# ax.set_yticklabels([])
# ax.set_xticks([])
# ax.set_yticks([])
# ax.axis("off")

# plt.show()



res_ls = [19,21,23]
model_ls = ['HG','HGPT','HG8M']

for res in res_ls:
    # load csv
    centroid_df = pd.read_csv('../Result/NB_centPredict_{}.csv'.format(res), sep=',')
    
    for model in model_ls:
       
        # plot
        fig, ax = plt.subplots()
        artist = dsshow(centroid_df, ds.Point('lon_c', 'lat_c'), aggregator=ds.mean(model), cmap='Blues', plot_width=300, plot_height=300, ax=ax)
        
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")
        
        # plt.show()
        
        # save the png
        plt.savefig('../Result/img/NB_pred_{}_{}.png'.format(model,res), bbox_inches='tight', pad_inches=0.0)
        plt.close()
        gc.collect()