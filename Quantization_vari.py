import GeoBaseFunc as gbfc
# import DgBaseFunc as dbfc
import pandas as pd
import numpy as np
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# dggs_res = int(sys.argv[1])
dggs_res = 23

# csv with centroids' coords
input_csv_path = '../Data/NB_sample_cent_{}.csv'.format(dggs_res)
sdf = pd.read_csv(input_csv_path, sep=',',usecols=['Cell_address', 'lon_c', 'lat_c','i','j','grid_code'])

# csv with climate station data
input_csv_path = '../Data/NB_climate_cent_{}.csv'.format(dggs_res)
cdf = pd.read_csv(input_csv_path, sep=',',usecols=['E_NORMAL_E','MONTH','lon_c','lat_c','i','j','VALUE'])

# convert lat/lon with wgs84 to Canada Atlas Lambert (4326->3979)
sdf[['lon_c_p','lat_c_p']] = [gbfc.reproject_coords(x,y) for x,y in zip(sdf.lon_c,sdf.lat_c)]

# extract values: nearest for factor data, bilinear for numeric data
sdf = gbfc.nearest_interp_df('../Data_clip_tif/out_geology.tif','geo',sdf,'lon_c_p','lat_c_p')
sdf = gbfc.nearest_interp_df('../Data_clip_tif/NB_WL_O_30.tif','wl',sdf,'lon_c_p','lat_c_p')
sdf = gbfc.nearest_interp_df('../Data_clip_tif/out_BIO_ECODIST_SOIL_TEXTURE.tif','sol',sdf,'lon_c_p','lat_c_p')
sdf = gbfc.nearest_interp_df('../Data_clip_tif/out_lc.tif','lc',sdf,'lon_c_p','lat_c_p')
sdf = gbfc.bilinear_interp_df('../Data_clip_tif/NB_NDVI_O_30.tif','ndvi',sdf,'lon_c_p','lat_c_p')
# sdf = gbfc.bilinear_interp_df('../Data_clip_tif/out_euclidean_NHN.tif','nhn',sdf,'lon_c_p','lat_c_p')
sdf = gbfc.bilinear_interp_df('../Data_clip_tif/out_Terra_MODIS_2015_MSI_Probability_100_v4_20180122.tif','msi',sdf,'lon_c_p','lat_c_p')

# populate ia, fcp, wl values
sdf['ia'] = np.where(sdf['lc'] == 17, 1, 0)
sdf['fcp'] = np.where((sdf['lc'] == 1) | (sdf['lc'] == 2) | (sdf['lc'] == 5) | (sdf['lc'] == 6) | (sdf['lc'] == 8), 1, 0)
sdf['wl'] = np.where(sdf['wl'] > 0, 1, 0)

# interpolate climate data
sdf = gbfc.IDW_interp_df(cdf,'Total precipitation mm',13,'precip',sdf,dggs_res)
sdf = gbfc.IDW_interp_df(cdf,'Mean daily temperature deg C',13,'tavg',sdf,dggs_res)
sdf = gbfc.IDW_interp_df(cdf,'Days with rainfall GE 10 mm',13,'r10',sdf,dggs_res)
sdf = gbfc.IDW_interp_df(cdf,'Days with rainfall GE 25 mm',13,'r25',sdf,dggs_res)
sdf = gbfc.IDW_interp_df(cdf,'Days with daily min temperature LT -10 deg C',13,'tm10',sdf,dggs_res)
sdf = gbfc.IDW_interp_df(cdf,'Days with snow depth GE 50 cm',13,'sd50',sdf,dggs_res)
sdf = gbfc.IDW_interp_df(cdf,'Total snowfall cm',13,'ts',sdf,dggs_res)

sdf = gbfc.IDW_interp_df(cdf,'Days with daily min temperature GT 0 deg C',3,'spr1',sdf,dggs_res)
sdf = gbfc.IDW_interp_df(cdf,'Days with daily min temperature GT 0 deg C',4,'spr2',sdf,dggs_res)
sdf = gbfc.IDW_interp_df(cdf,'Days with daily min temperature GT 0 deg C',5,'spr3',sdf,dggs_res)

# populate number of spring days with min temp > 0 degree
sdf['spr'] = sdf['spr1'] + sdf['spr2'] + sdf['spr3']

# save csv
sdf = sdf.drop(columns=['lon_c_p','lat_c_p','spr1','spr2','spr3'])
output_csv_path = '../Data/NB_sample_cent_quanti_{}.csv'.format(dggs_res)
sdf.to_csv(output_csv_path, index=False)



