import warnings, sys
import pandas as pd
import DgBaseFunc as dbfc
import DgTopogFunc as dtfc
import DgHydroFunc as dhfc

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

# compute slope and aspect
elev_df_out = dtfc.slope_aspect_df(elev_df,elev_df,'FDA',dggs_res,cell_spacing)

# populate flow direction
elev_df_out = dhfc.flow_direction_restricted_df(elev_df_out,dggs_res)

# save csv
output_csv_path = 'Result/NB_flowdirec_{}.csv'.format(dggs_res)
elev_df_out.to_csv(output_csv_path, index=True)