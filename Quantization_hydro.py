import warnings, sys, time, functools, os, gc
import pandas as pd
import geopandas as gpd
import numpy as np
import DgBaseFunc as dbfc
import DgTopogFunc as dtfc
import DgHydroFunc as dhfc
import multiprocess as mp

warnings.simplefilter('error', RuntimeWarning) 

# set resolution level and sub basin id
dggs_res = int(sys.argv[1])
# sub_id = int(sys.argv[2])
sub_id = int(os.environ.get("SLURM_ARRAY_TASK_ID"))

# look up cell spacing and vertical resolution
look_up = dbfc.look_up_table()
cell_spacing = look_up.loc[dggs_res,'cell_spacing'] * 1000
vertical_res = look_up.loc[dggs_res,'verti_res']
cell_area = look_up.loc[dggs_res,'cell_area'] * 1000000

# # read csv
# input_csv_path = 'Result/NB_elev_{}.csv'.format(dggs_res)
# elev_df = pd.read_csv(input_csv_path, sep=',')
# elev_df = elev_df.set_index(['i', 'j'])


# merge dfs
centroid_df = pd.read_csv('Result/NB_area_wgs84_centroids_{}.csv'.format(dggs_res), usecols=(['i','j','lon_c','lat_c']))
elev_df = pd.read_csv('Result/NB_elev_{}.csv'.format(dggs_res), sep=',')
merge_df = pd.merge(left = elev_df, right = centroid_df, how="inner", on=['i','j'])

# filter out those fall out of study area
extent = gpd.GeoDataFrame.from_file('Data/NB_area_subbasin_wgs84.shp')
merge_gdf = gpd.GeoDataFrame(merge_df, geometry=gpd.points_from_xy(merge_df.lon_c, merge_df.lat_c))
merge_gdf = merge_gdf[merge_gdf.geometry.within(extent.geometry[sub_id-1])]

# convert to pd df
elev_df = pd.DataFrame(merge_gdf)
elev_df = elev_df.drop(columns=['lon_c','lat_c','geometry'])
elev_df = elev_df.set_index(['i', 'j'])
elev_df = elev_df.sort_index()

del centroid_df,merge_gdf,merge_df
gc.collect()

## depression filling; flat area included
start_time = time.time()

# initialize queues 
Q = elev_df.index.values.tolist() # all cell indices

edge_cell_df_p = functools.partial(dbfc.edge_cell_df,res=dggs_res,allcell=Q)
n_cores = int(os.environ.get('SLURM_CPUS_PER_TASK',default=1))
elev_df_split = np.array_split(elev_df, n_cores)
pool = mp.Pool(processes = n_cores)
elev_df_temp = pd.concat(pool.map(edge_cell_df_p, elev_df_split))
pool.close()
pool.join()

# elev_df_temp = dbfc.edge_cell_df(elev_df,dggs_res,Q)
P = elev_df_temp.index.values.tolist()

# for i in Q:
#     if dbfc.edge_cell(dggs_res,i,Q):
#         head = i
#         break
    
# P = dbfc.edge_cell_chain(dggs_res,head,Q)

# temp_p_neigh = []

# for i in P:
#     temp_p_neigh = temp_p_neigh + dbfc.neighbor_coords(dggs_res,i)[1:]
#     temp_p_neigh = list(set(temp_p_neigh))
    
# temp_p_neigh = [i for i in temp_p_neigh if i not in P and i in Q]

# for i in temp_p_neigh:
#     if dbfc.edge_cell(dggs_res,i,Q):
#         P.append(i)

U = [i for i in Q if (i not in P)] # unfinished queue
A = [] # plain queue

print ("Queue initialization processing time: %s seconds" % (time.time() - start_time))

# fill depression in ascending order based on elevation
start_time = time.time()

P_df = elev_df[elev_df.index.isin(P)]

while len(U) > 0:
    
    P_df = P_df.sort_values(by=['model_elev'])
    ij = P_df[:1].index.values[0]
    elev = P_df[:1].model_elev.values[0]
    C = dbfc.neighbor_coords(dggs_res,ij)[1:]
    C = [i for i in C if (i not in P) and (i in U)]
    P.remove(ij)
    P_df = P_df.drop([ij])
    
    for c in C:
        U.remove(c)
        c_elev = elev_df['model_elev'].loc[c]
        if c_elev != -32767:
            if c_elev <= elev:
                elev_df.at[(c),'model_elev'] = elev + 0.01 * cell_spacing
                A.append(c)
            else:
                P.append(c)
                P_df = P_df.append(elev_df.loc[c])
        
        while len(A) > 0:
            for a in A:
                a_elev = elev_df['model_elev'].loc[a]
                D = dbfc.neighbor_coords(dggs_res,a)[1:]
                D = [i for i in D if (i not in P) and (i in U) and (i not in C)]
                for d in D:
                    U.remove(d)
                    d_elev = elev_df['model_elev'].loc[d]
                    if d_elev <= a_elev and d not in A:
                        elev_df.at[(d),'model_elev'] = a_elev + 0.01 * cell_spacing
                        A.append(d)
                    else:
                        P.append(d)
                        P_df = P_df.append(elev_df.loc[d])
                A.remove(a)

print ("Depression filling processing time: %s seconds" % (time.time() - start_time))

output_csv_path = 'Result/NB_NoneDepress{}_{}.csv'.format(sub_id,dggs_res)
elev_df.to_csv(output_csv_path, index=True)

## generate flow directions with MDG algorithm
# compute slope and aspect
slope_aspect_df_p = functools.partial(dtfc.slope_aspect_df,elev_df=elev_df,method='MDG',res=dggs_res,cell_spacing=cell_spacing)
n_cores = int(os.environ.get('SLURM_CPUS_PER_TASK',default=1))
elev_df_split = np.array_split(elev_df, n_cores)
pool = mp.Pool(processes = n_cores)
elev_df = pd.concat(pool.map(slope_aspect_df_p, elev_df_split))
pool.close()
pool.join()
# elev_df = slope_aspect_df_p(elev_df)

# populate flow direction
flow_direction_restricted_df_p = functools.partial(dhfc.flow_direction_restricted_df,res=dggs_res)
elev_df_split = np.array_split(elev_df, n_cores)
pool = mp.Pool(processes = n_cores)
elev_df = pd.concat(pool.map(flow_direction_restricted_df_p, elev_df_split))
pool.close()
pool.join()
# elev_df = dhfc.flow_direction_restricted_df(elev_df,dggs_res)

## compute flow accumulation and hydro indices
# initialize inflow and upslope
elev_df['inflow'] = 0
elev_df['upslope'] = 1

# calculate inflow count
inflow_count_restricted_df_p = functools.partial(dhfc.inflow_count_restricted_df,res=dggs_res,elev_df=elev_df)
elev_df_split = np.array_split(elev_df, n_cores)
pool = mp.Pool(processes = n_cores)
elev_df = pd.concat(pool.map(inflow_count_restricted_df_p, elev_df_split))
pool.close()
pool.join()
# elev_df = dhfc.inflow_count_restricted_df(elev_df,dggs_res,elev_df)

# recursively determine upslope cell counts
while True:
    elev_df_temp = elev_df[elev_df['inflow'] == 0].dropna(subset=['direction_code'])
    if len(elev_df_temp) > 0:
        for ij in elev_df_temp.index.values:
            dhfc.upslope_restricted(ij,dggs_res,elev_df)
    elif all(i < 0 for i in elev_df['inflow']):
        break
    else:
        elev_df['inflow'] =  elev_df['inflow'] - 1


# calculate upslope area and contributing area
elev_df['upslope_area'] = elev_df['upslope'] * cell_area
elev_df['contri_area'] = elev_df['upslope_area'] / cell_spacing

# save csv
elev_df = elev_df.drop(columns=['inflow'])
output_csv_path = 'Result/NB_flowdirec{}_{}.csv'.format(sub_id,dggs_res)
elev_df.to_csv(output_csv_path, index=True)

## match values for sample points
input_csv_path = 'Data/NB_sample_cent_quanti3_{}.csv'.format(dggs_res)
sdf = pd.read_csv(input_csv_path, sep=',')
sdf = sdf.set_index(['i', 'j'])

# drop duplicated columns
elev_df = elev_df.drop(columns=['Cell_address','model_elev','gradient_deg','aspect_deg'])
join_df = sdf.join(elev_df,how = 'left')
join_df = dhfc.TWI_df(join_df)
join_df = dhfc.SPI_df(join_df)

join_df = join_df.rename(columns={"direction_code": "fldir", "upslope_area": "flacc"})
join_df = join_df.drop(columns=['upslope','contri_area'])

# save csv
output_csv_path = 'Data/sub_basin/NB_sample_cent_quanti4_{}_{}.csv'.format(sub_id,dggs_res)
join_df.to_csv(output_csv_path, index=True)


