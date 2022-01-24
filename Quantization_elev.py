import GeoBaseFunc as gbfc
import DgBaseFunc as dbfc
import pandas as pd
import geopandas as gpd
import multiprocess as mp
import warnings,sys,time

warnings.simplefilter('error', RuntimeWarning) 

# set resolution level and study area
dggs_res = int(sys.argv[1])

# look up cellsize and vertical resolution
look_up = dbfc.look_up_table()
vertical_res = look_up.loc[dggs_res,'verti_res']

# read the csv into a dataframe
input_csv_path = 'Result/NB_area_wgs84_centroids_{}.csv'.format(dggs_res)
centroid_df = pd.read_csv(input_csv_path, sep=',')
centroid_gdf = gpd.GeoDataFrame(centroid_df, geometry=gpd.points_from_xy(centroid_df.lon_c, centroid_df.lat_c))

# record timing -- start
start_time = time.time()

# filter out those fall out of study area
extent = gpd.GeoDataFrame.from_file('Data/NB_area_wgs84.shp')
centroid_gdf = centroid_gdf[centroid_gdf.geometry.within(extent.geometry[0])]

# non-parallel
centroid_gdf = gbfc.bilinear_interp_df('Data/NB_DTM_O_30_wgs84.tif','model_elev',centroid_gdf,'lon_c','lat_c')
centroid_gdf['model_elev'] = [round(elev,vertical_res) for elev in centroid_gdf['model_elev']]

# # call the function by parallel processing
# n_cores = int(os.environ.get('SLURM_CPUS_PER_TASK',default=1))
# centroid_df_split = np.array_split(centroid_gdf, n_cores)
# pool = mp.Pool(processes = n_cores)
# centroid_gdf = pd.concat(pool.map(gbfc.bilinear_interp_df, centroid_df_split))
# pool.close()
# pool.join()

# record timing -- end
print (dggs_res)
print ("Processing time: %s seconds" % (time.time() - start_time))

# save the results as csv
centroid_gdf = centroid_gdf.drop(columns=['lon_c','lat_c','geometry'])
output_csv_path = 'Result/NB_elev_{}.csv'.format(dggs_res)
centroid_gdf.to_csv(output_csv_path, index=False)
