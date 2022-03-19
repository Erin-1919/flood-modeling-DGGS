import warnings, sys, time, os, functools
import pandas as pd
import geopandas as gpd
import numpy as np
import DgBaseFunc as dbfc
import GeoBaseFunc as gbfc
import multiprocess as mp

warnings.simplefilter('error', RuntimeWarning) 

# set resolution level
dggs_res = int(sys.argv[1])

# look up cell spacing and vertical resolution
look_up = dbfc.look_up_table()
cell_spacing = look_up.loc[dggs_res,'cell_spacing'] * 1000
vertical_res = look_up.loc[dggs_res,'verti_res']

# read csv
input_csv_path = 'Result/NB_area_wgs84_centroids_{}.csv'.format(dggs_res)
centroid_df = pd.read_csv(input_csv_path, sep=',')
centroid_gdf = gpd.GeoDataFrame(centroid_df, geometry=gpd.points_from_xy(centroid_df.lon_c, centroid_df.lat_c))
centroid_gdf = centroid_gdf.set_index(['i', 'j'])

input_csv_path = 'Data/NB_sample_cent_quanti2_{}.csv'.format(dggs_res)
sdf = pd.read_csv(input_csv_path, sep=',')
sdf = sdf.set_index(['i', 'j'])
sdf = sdf.sort_index()

# filter out those fall out of study area
extent = gpd.GeoDataFrame.from_file('Data/NB_area_wgs84.shp')
centroid_gdf = centroid_gdf[centroid_gdf.geometry.within(extent.geometry[0])]

# extract values by nearest - rasterized NHN
centroid_gdf = gbfc.nearest_interp_df(centroid_gdf,'Data/out_nlflow_1_wgs84.tif','nhn1','lon_c','lat_c')
centroid_gdf = gbfc.nearest_interp_df(centroid_gdf,'Data/out_waterbody_2_wgs84.tif','nhn2','lon_c','lat_c')

# populate column value indicating existence of nhn
centroid_gdf['nhn3'] = np.where((centroid_gdf['nhn1'] != -32767) | (centroid_gdf['nhn2'] != -32767), 1, 0)
centroid_gdf = centroid_gdf[centroid_gdf['nhn3'] == 1]

# calcualte hex distance to the closest nhn for each cell
coords_target = centroid_gdf.index.values.tolist()
hex_dist_comb_df_p = functools.partial(dbfc.hex_dist_comb_df,coords_target=coords_target,var='nhn',res=dggs_res)

n_cores = int(os.environ.get('SLURM_CPUS_PER_TASK',default=1))
sdf_split = np.array_split(sdf, n_cores)
pool = mp.Pool(processes = n_cores)
sdf = pd.concat(pool.map(hex_dist_comb_df_p, sdf_split))
pool.close()
pool.join()

# start_time = time.time()
# sdf['nhn'] = [dbfc.hex_dist_l(coord,coords_target,dggs_res) for coord in sdf.index.values]
# print ("Processing time: %s seconds" % (time.time() - start_time))

# start_time = time.time()

# sdf['nhn'] = np.nan
# sdf['nhn'] = [dbfc.hex_dist_m(coord,coords_target,dggs_res) for coord in sdf.index.values]
# coords_left = sdf[np.isnan(sdf['nhn'])].index.values.tolist()
# dist_left = [dbfc.hex_dist_l(coord,coords_target,dggs_res) for coord in coords_left]
# for c,d in zip(coords_left,dist_left):
#     sdf.at[c,'nhn'] = d
# print ("Processing time: %s seconds" % (time.time() - start_time))

# save csv
output_csv_path = 'Data/NB_sample_cent_quanti3_{}.csv'.format(dggs_res)
sdf.to_csv(output_csv_path, index=True)

