import functools
import pandas as pd
import numpy as np
import geopandas as gpd
import multiprocess as mp
import GeoBaseFunc as gbfc
import DgBaseFunc as dbfc
import DgTopogFunc as dtfc
import DgHydroFunc as dhfc
import warnings,sys,gc,os

warnings.simplefilter('error', RuntimeWarning) 
warnings.simplefilter(action='ignore', category=FutureWarning)

n_cores = int(os.environ.get('SLURM_CPUS_PER_TASK',default=1))
dggs_res = int(sys.argv[1])

# look up cellsize and vertical resolution
look_up = dbfc.look_up_table()
cell_spacing = look_up.loc[dggs_res,'cell_spacing'] * 1000
vertical_res = look_up.loc[dggs_res,'verti_res']
cell_area = look_up.loc[dggs_res,'cell_area'] * 1000000


########### Quantize elevation (dtm) ##############

input_csv_path = 'Result/NB_area_wgs84_centroids_{}.csv'.format(dggs_res)
centroid_df = pd.read_csv(input_csv_path, sep=',')
centroid_gdf = gpd.GeoDataFrame(centroid_df, geometry=gpd.points_from_xy(centroid_df.lon_c, centroid_df.lat_c))

# filter out those fall out of study area
extent = gpd.GeoDataFrame.from_file('Data/NB_area_wgs84.shp')
centroid_gdf = centroid_gdf[centroid_gdf.geometry.within(extent.geometry[0])]

# quantization by bilinear interp
bilinear_interp_df_dtm = functools.partial(gbfc.bilinear_interp_df,tif='Data/NB_DTM_O_30_wgs84.tif',var='model_elev',x_col='lon_c',y_col='lat_c')
centroid_gdf = bilinear_interp_df_dtm(centroid_gdf)
centroid_gdf['model_elev'] = [round(elev,vertical_res) for elev in centroid_gdf['model_elev']]

# save the results as csv
centroid_gdf = centroid_gdf.drop(columns=['geometry'])
output_csv_path = 'Result/NB_elev_{}.csv'.format(dggs_res)
centroid_gdf.to_csv(output_csv_path, index=False)

print ("Finish elevation quantization.")
del centroid_gdf,centroid_df
gc.collect()


########### Compute topographic parameters (slp,asp,tri,tpi,curv,rgh) ##############

input_csv_path = 'Result/NB_elev_{}.csv'.format(dggs_res)
elev_df = pd.read_csv(input_csv_path, sep=',')
elev_df = elev_df.set_index(['i', 'j'])

## parallel compute topographic parameters
slope_aspect_df_p = functools.partial(dtfc.slope_aspect_df,elev_df=elev_df,method='FDA',res=dggs_res,cell_spacing=cell_spacing)
curvature_df_p = functools.partial(dtfc.curvature_df,elev_df=elev_df,res=dggs_res,cell_spacing=cell_spacing)
roughness_df_p = functools.partial(dtfc.roughness_df,elev_df=elev_df,res=dggs_res,vertical_res=vertical_res,rings=1)
TRI_df_p = functools.partial(dtfc.TRI_df,res=dggs_res,elev_df=elev_df)
TPI_df_p = functools.partial(dtfc.TPI_df,res=dggs_res,elev_df=elev_df)

# slp / asp
elev_df_split = np.array_split(elev_df, n_cores)
pool = mp.Pool(processes = n_cores)
elev_df = pd.concat(pool.map(slope_aspect_df_p, elev_df_split))
pool.close()
pool.join()

# curv
elev_df_split = np.array_split(elev_df, n_cores)
pool = mp.Pool(processes = n_cores)
elev_df = pd.concat(pool.map(curvature_df_p, elev_df_split))
pool.close()
pool.join()

# rgh
elev_df_split = np.array_split(elev_df, n_cores)
pool = mp.Pool(processes = n_cores)
elev_df = pd.concat(pool.map(roughness_df_p, elev_df_split))
pool.close()
pool.join()

# tri
elev_df_split = np.array_split(elev_df, n_cores)
pool = mp.Pool(processes = n_cores)
elev_df = pd.concat(pool.map(TRI_df_p, elev_df_split))
pool.close()
pool.join()

# tpi
elev_df_split = np.array_split(elev_df, n_cores)
pool = mp.Pool(processes = n_cores)
elev_df = pd.concat(pool.map(TPI_df_p, elev_df_split))
pool.close()
pool.join()

# save csv
output_csv_path = 'Result/NB_topog_{}.csv'.format(dggs_res)
elev_df.to_csv(output_csv_path, index=True)

print ("Finish topographic parameters computation.")
del elev_df,elev_df_split
gc.collect()


########### Quantize geomorphology parameters (geo,wl,sol,lc,ndvi,msi,ia,fcp) ##############

input_csv_path = 'Result/NB_topog_{}.csv'.format(dggs_res)
topog_df = pd.read_csv(input_csv_path, sep=',')

# convert lat/lon with wgs84 to Canada Atlas Lambert (4326->3979)
reproject_coords_df_p = functools.partial(gbfc.reproject_coords_df,x_col='lon_c',y_col='lat_c',x_new='lon_c_p',y_new='lat_c_p')
topog_df_split = np.array_split(topog_df, n_cores)
pool = mp.Pool(processes = n_cores)
topog_df = pd.concat(pool.map(reproject_coords_df_p, topog_df_split))
pool.close()
pool.join()

# extract values: nearest for factor data, bilinear for numeric data
topog_df = gbfc.nearest_interp_df(topog_df,'Data/out_geology.tif','geo','lon_c_p','lat_c_p')
topog_df = gbfc.nearest_interp_df(topog_df,'Data/NB_WL_O_30.tif','wl','lon_c_p','lat_c_p')
topog_df = gbfc.nearest_interp_df(topog_df,'Data/out_BIO_ECODIST_SOIL_TEXTURE.tif','sol','lon_c_p','lat_c_p')
topog_df = gbfc.nearest_interp_df(topog_df,'Data/out_lc.tif','lc','lon_c_p','lat_c_p')
topog_df = gbfc.bilinear_interp_df(topog_df,'Data/NB_NDVI_O_30.tif','ndvi','lon_c_p','lat_c_p')
topog_df = gbfc.bilinear_interp_df(topog_df,'Data/out_Terra_MODIS_2015_MSI_Probability_100_v4_20180122.tif','msi','lon_c_p','lat_c_p')

# populate ia, fcp, wl values
topog_df['ia'] = np.where(topog_df['lc'] == 17, 1, 0)
topog_df['fcp'] = np.where((topog_df['lc'] == 1) | (topog_df['lc'] == 2) | (topog_df['lc'] == 5) | (topog_df['lc'] == 6) | (topog_df['lc'] == 8), 1, 0)
topog_df['wl'] = np.where(topog_df['wl'] > 0, 1, 0)

# save csv
topog_df = topog_df.drop(columns=['lon_c_p','lat_c_p'])
output_csv_path = 'Result/NB_geom_{}.csv'.format(dggs_res)
topog_df.to_csv(output_csv_path, index=False)

print ("Finish geomorphology quantization.")
del topog_df
gc.collect()


########### Interpolate climate variables (precip,tavg,r10,r25,tm10,sd50,ts,spr) ##############

input_csv_path = 'Result/NB_geom_{}.csv'.format(dggs_res)
geom_df = pd.read_csv(input_csv_path, sep=',')

# csv with climate station data
input_csv_path = 'Data/NB_climate_cent_{}.csv'.format(dggs_res)
cdf = pd.read_csv(input_csv_path, sep=',',usecols=['E_NORMAL_E','MONTH','lon_c','lat_c','i','j','VALUE'])

# interpolate climate data
geom_df = gbfc.IDW_interp_df(geom_df,cdf,'Total precipitation mm',13,'precip',dggs_res)
geom_df = gbfc.IDW_interp_df(geom_df,cdf,'Mean daily temperature deg C',13,'tavg',dggs_res)
geom_df = gbfc.IDW_interp_df(geom_df,cdf,'Days with rainfall GE 10 mm',13,'r10',dggs_res)
geom_df = gbfc.IDW_interp_df(geom_df,cdf,'Days with rainfall GE 25 mm',13,'r25',dggs_res)
geom_df = gbfc.IDW_interp_df(geom_df,cdf,'Days with daily min temperature LT -10 deg C',13,'tm10',dggs_res)
geom_df = gbfc.IDW_interp_df(geom_df,cdf,'Days with snow depth GE 50 cm',13,'sd50',dggs_res)
geom_df = gbfc.IDW_interp_df(geom_df,cdf,'Total snowfall cm',13,'ts',dggs_res)

geom_df = gbfc.IDW_interp_df(geom_df,cdf,'Days with daily min temperature GT 0 deg C',3,'spr1',dggs_res)
geom_df = gbfc.IDW_interp_df(geom_df,cdf,'Days with daily min temperature GT 0 deg C',4,'spr2',dggs_res)
geom_df = gbfc.IDW_interp_df(geom_df,cdf,'Days with daily min temperature GT 0 deg C',5,'spr3',dggs_res)

# populate number of spring days with min temp > 0 degree
geom_df['spr'] = geom_df['spr1'] + geom_df['spr2'] + geom_df['spr3']

# save csv
geom_df = geom_df.drop(columns=['spr1','spr2','spr3'])
output_csv_path = 'Result/NB_climate_{}.csv'.format(dggs_res)
geom_df.to_csv(output_csv_path, index=False)

print ("Finish climate variables interpolation.")
del geom_df
gc.collect()


########### Calculate distance to NHN (nhn) ##############

input_csv_path = 'Result/NB_area_wgs84_centroids_{}.csv'.format(dggs_res)
centroid_df = pd.read_csv(input_csv_path, sep=',')
centroid_gdf = gpd.GeoDataFrame(centroid_df, geometry=gpd.points_from_xy(centroid_df.lon_c, centroid_df.lat_c))
centroid_gdf = centroid_gdf.set_index(['i', 'j'])

input_csv_path = 'Result/NB_climate_{}.csv'.format(dggs_res)
clim_df = pd.read_csv(input_csv_path, sep=',')
clim_df = clim_df.set_index(['i', 'j'])
clim_df = clim_df.sort_index()

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

def hex_dist_comb_df(dataframe,coords_target,var,res):
    dataframe[var] = [dbfc.hex_dist_m(coord,coords_target,res) for coord in list(dataframe.index.values)]
    df_done = dataframe.dropna(subset=[var])
    df_left = dataframe[np.isnan(dataframe[var])]
    df_left[var] = [dbfc.hex_dist_l(coord,coords_target,res) for coord in list(df_left.index.values.tolist())]
    dataframe = pd.concat([df_done,df_left])
    return dataframe

# clim_df = dbfc.hex_dist_comb_df(clim_df,coords_target=coords_target,var='nhn',res=dggs_res)
# clim_df = hex_dist_comb_df(clim_df,coords_target=coords_target,var='nhn',res=dggs_res)

hex_dist_comb_df_p = functools.partial(hex_dist_comb_df,coords_target=coords_target,var='nhn',res=dggs_res)

clim_df_split = np.array_split(clim_df, n_cores)
pool = mp.Pool(processes = n_cores)
clim_df = pd.concat(pool.map(hex_dist_comb_df_p, clim_df_split))
pool.close()
pool.join()

# save csv
output_csv_path = 'Result/NB_nhnDist_{}.csv'.format(dggs_res)
clim_df.to_csv(output_csv_path, index=True)

print ("Finish NHN distance calculation.")
del clim_df,centroid_df,centroid_gdf,coords_target
gc.collect()


########### Compute hydrology parameters (fldir,flacc,twi,spi) ##############

input_csv_path = 'Result/NB_nhnDist_{}.csv'.format(dggs_res)
elev_df = pd.read_csv(input_csv_path, sep=',')
elev_df = elev_df.set_index(['i', 'j'])

# todo

# save csv
elev_df = elev_df.rename(columns={"model_elev": "dtm"})
output_csv_path = 'Result/NB_finalPredict_{}.csv'.format(dggs_res)
elev_df.to_csv(output_csv_path, index=True)

print ("Finish hydrology parameters computation.")