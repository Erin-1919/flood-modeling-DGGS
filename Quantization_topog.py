import warnings, sys
import pandas as pd
import DgBaseFunc as dbfc
import DgTopogFunc as dtfc
import GeoBaseFunc as gbfc

warnings.simplefilter('error', RuntimeWarning) 

# set resolution level
dggs_res = int(sys.argv[1])

# look up cell spacing and vertical resolution
look_up = dbfc.look_up_table()
cell_spacing = look_up.loc[dggs_res,'cell_spacing'] * 1000
vertical_res = look_up.loc[dggs_res,'verti_res']

# read csv
input_csv_path = 'Result/NB_elev_{}.csv'.format(dggs_res)
elev_df = pd.read_csv(input_csv_path, sep=',')
elev_df = elev_df.set_index(['i', 'j'])

input_csv_path = 'Data/NB_sample_cent_quanti_{}.csv'.format(dggs_res)
sdf = pd.read_csv(input_csv_path, sep=',')
sdf = sdf.set_index(['i', 'j'])

# compute topographic parameters
sdf = gbfc.bilinear_interp_df('Data/NB_DTM_O_30_wgs84.tif','model_elev',sdf,'lon_c','lat_c')
sdf['model_elev'] = [round(elev,vertical_res) for elev in sdf['model_elev']]

sdf = dtfc.slope_aspect_df(elev_df,sdf,'FDA',dggs_res,cell_spacing)
sdf = dtfc.curvature_df(elev_df,sdf,dggs_res,cell_spacing)
sdf = dtfc.roughness_df(elev_df,sdf,dggs_res,vertical_res,1)
sdf = dtfc.TRI_df(sdf,dggs_res,elev_df)
sdf = dtfc.TPI_df(sdf,dggs_res,elev_df)

# rename
sdf = sdf.rename(columns={"model_elev": "dtm", "gradient_deg": "slp", "aspect_deg": "asp"})

# save csv
output_csv_path = 'Data/NB_sample_cent_quanti2_{}.csv'.format(dggs_res)
sdf.to_csv(output_csv_path, index=True)

