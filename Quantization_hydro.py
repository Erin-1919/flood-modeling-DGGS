import warnings, sys, time
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
cell_area = look_up.loc[dggs_res,'cell_area'] * 1000000

# read csv
input_csv_path = 'Result/NB_elev_{}.csv'.format(dggs_res)
elev_df = pd.read_csv(input_csv_path, sep=',')
elev_df = elev_df.set_index(['i', 'j'])

## depression filling; flat area included
# initialize queues 
Q = elev_df.index.values.tolist() # all cell indices
P = dbfc.edge_cell_ls(dggs_res,elev_df) # priority queue
U = [i for i in Q if (i not in P)] # unfinished queue
A = [] # plain queue

start_time = time.time()

while len(U) > 0:
    Q_df = elev_df[elev_df.index.isin(P)]
    Q_df = Q_df[Q_df.model_elev == Q_df.model_elev.min()]
    ij = Q_df[:1].index.values[0]
    elev = Q_df[:1].model_elev.values[0]
    C = dbfc.neighbor_coords(dggs_res,ij)[1:]
    C = [i for i in C if (i not in P) and (i in U)]
       
    for c in C:
        c_elev = elev_df['model_elev'].loc[c]
        if c_elev != -32767:
            if c_elev <= elev:
                elev_df.at[(c),'model_elev'] = elev + 0.01 * cell_spacing
                A.append(c)
            else:
                P.append(c)
        try:
            U.remove(c)
        except:
            pass
        
        while len(A) > 0:
            for a in A:
                a_elev = elev_df['model_elev'].loc[a]
                D = dbfc.neighbor_coords(dggs_res,a)[1:]
                D = [i for i in D if (i not in P) and (i in U)]
                for d in D:
                    d_elev = elev_df['model_elev'].loc[d]
                    if d_elev <= a_elev and d not in A:
                        elev_df.at[(d),'model_elev'] = a_elev + 0.01 * cell_spacing
                        A.append(d)
                    else:
                        P.append(d)
                    try:
                        U.remove(d)
                    except:
                        pass
                A.remove(a)
    P.remove(ij)

print ("Processing time: %s seconds" % (time.time() - start_time))

## generate flow directions with MDG algorithm
# compute slope and aspect
elev_df = dtfc.slope_aspect_df(elev_df,elev_df,'MDG',dggs_res,cell_spacing)

# populate flow direction
elev_df = dhfc.flow_direction_restricted_df(elev_df,dggs_res)

## compute flow accumulation and hydro indices
# initialize inflow and upslope
elev_df['inflow'] = 0
elev_df['upslope'] = 1

# calculate inflow count
elev_df = dhfc.inflow_count_restricted_df(elev_df,dggs_res,elev_df)

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
output_csv_path = 'Result/NB_flowdirec_{}.csv'.format(dggs_res)
elev_df.to_csv(output_csv_path, index=True)

## match values for sample points
input_csv_path = 'Data/NB_sample_cent_quanti3_{}.csv'.format(dggs_res)
sdf = pd.read_csv(input_csv_path, sep=',')
sdf = sdf.set_index(['i', 'j'])

# drop duplicated columns
elev_df = elev_df.drop(columns=['Cell_address','model_elev','gradient_deg','aspect_deg'])
join_df = sdf.join(elev_df,how = 'left')
join_df = dhfc.TRI_df(join_df)
join_df = dhfc.TPI_df(join_df)

# save csv
output_csv_path = 'Data/NB_sample_cent_quanti4_{}.csv'.format(dggs_res)
join_df.to_csv(output_csv_path, index=True)


