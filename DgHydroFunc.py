import numpy as np
import pandas as pd
import math
import DgBaseFunc


def flow_direction_restricted(asp,res):
    if res%2 == 1:
        if 0 <= asp < 360:
            direc_code = asp // 60 + 1
        else:
            direc_code = asp // 60
    elif res%2 == 0:
        if 0 <= asp < 330:
            direc_code = (asp + 30) // 60 + 1
        else:
            direc_code = (asp + 30) // 60
    return direc_code

def flow_direction_restricted_df(dataframe,res):
    ''' calculate flow direction among normal area, leave pits and flat area as NaN '''
    dataframe['aspect_deg'] = [i if i != -1 else np.nan for i in dataframe['aspect_deg']]
    dataframe['direction_code'] = np.nan
    dataframe['direction_code'] = [flow_direction_restricted(i,res) if not np.isnan(i) else np.nan for i in dataframe['aspect_deg']]
    return dataframe

def pits_and_flat_df(dataframe,res,elev_df):
    ''' determine pits and flat area, i.e., center elevation lower or equal to neighbors
    return a dataframe with ascending elevations '''
    pending_i = []
    pending_j = []
    pending_elev = []
    for coords in dataframe.index.values:
        i,j = coords[0],coords[1]
        elev_neighbor = DgBaseFunc.neighbor_navig(res,coords,elev_df)
        if DgBaseFunc.edge_cell_exist(elev_neighbor):
            pass
        else:
            neighbor = elev_neighbor[1:]
            if all(n >= elev_neighbor[0] for n in neighbor):
                pending_i.append(i)
                pending_j.append(j)
                pending_elev.append(elev_neighbor[0])
    pending_df = pd.DataFrame(list(zip(pending_i,pending_j,pending_elev)), columns =['i','j','elev'])
    pending_df = pending_df.set_index(['i', 'j'])
    return pending_df

def flow_to_cell(coords,res,elev_df):
    ''' determine the coords of the cell that a target cell will flow into '''
    try:
        direc = elev_df['direction_code'].loc[coords]
        if not np.isnan(direc):
            if direc == -1:
                return -1
            else:
                direct = int(direc)
                neighbor_flowto_coords = DgBaseFunc.neighbor_coords(res,coords)[direct]
                if neighbor_flowto_coords in list(elev_df.index.values):
                    return neighbor_flowto_coords
    except KeyError:
        pass

def upslope_restricted(coords,res,elev_df):
    ''' the count of the upslope count grid cell the zero valued cell 
    flows into is increased by adding the zero cells's upslope area;
    if the flow-into cell has negative inflow count then set it to 0;
    if not, the inflow grid cells involved, including the zero valued cell, 
    all have their values decreased by 1 '''
    direct = int(elev_df['direction_code'].loc[coords])
    upslope = int(elev_df['upslope'].loc[coords])
    neighbor_flowto = DgBaseFunc.neighbor_coords(res,coords)
    try:
        elev_df.at[neighbor_flowto[direct],'upslope'] += upslope
        if elev_df.at[neighbor_flowto[direct],'inflow'] < 0:
            if not np.isnan(elev_df.at[neighbor_flowto[direct],'direction_code']):
                elev_df.at[neighbor_flowto[direct],'inflow'] = 0
        else:
            elev_df.at[neighbor_flowto[direct],'inflow'] -= 1
    except KeyError:
        pass
    elev_df.at[coords,'inflow'] -= 1
    
def check_closed_loop(coords,res,elev_df):
    ''' check if there are cells forming a closed loop; 
    return the cell coord and flow path if there are any;
    tracing stops when encounter an edge cell or a pit/flat '''
    flow_path = [coords]
    flowto_coords = flow_to_cell(coords,res,elev_df)
    while True:
        if flowto_coords is None or flowto_coords == -1:
            return None,None
            break
        elif flowto_coords in flow_path:
            return flowto_coords,flow_path
            break
        else:
            flow_path.append(flowto_coords)
            flowto_coords = flow_to_cell(flowto_coords,res,elev_df)
            
def check_closed_loop_df(dataframe,res,elev_df):
    ''' check if there are any closed loops in a dataframe '''
    close_loop_df = pd.DataFrame(columns = ['flow_path'])
    for coords in dataframe.index.values:
        flowto_coords,flow_path = check_closed_loop(coords,res,elev_df)
        if flowto_coords is not None:
            cell_index = flow_path.index(flowto_coords)
            flow_path = flow_path[cell_index:]
            flow_elev = [elev_df['model_elev'].loc[coord] for coord in flow_path]
            flow_dict = dict(zip(flow_path, flow_elev))
            flow_dict = dict(sorted(flow_dict.items(), key = lambda item: (item[1], item[0])))
            close_loop_df = close_loop_df.append({'flow_path': flow_dict}, ignore_index=True)
    return close_loop_df

def inflow_count_restricted(coords,res,elev_df):
    ''' count inflow cells, ranging from zero to six meaning none to all of the neighbors '''
    neighbor_direc = DgBaseFunc.neighbor_direc_navig(res,coords,elev_df)
    count = 0
    direction_ls = [4,5,6,1,2,3]
    for i in range(1,7):
        if neighbor_direc[i] == direction_ls[i-1]:
            count += 1
    return count

def inflow_count_restricted_df(dataframe,res,elev_df):
    ''' count inflow cell numbers over a dataframe '''
    dataframe['inflow'] = [inflow_count_restricted(ij,res,elev_df) for ij in dataframe.index.values]
    return dataframe

def TWI_calcu(alpha,beta):
    ''' Topographic wetness index (TWI) '''
    if beta == 0:
        TWI_value = np.nan
    else:
        beta_rad = math.radians(beta)
        TWI_value = math.log(alpha/math.tan(beta_rad))
    return TWI_value
    
def SPI_calcu(alpha,beta):
    ''' Stream power index (SPI) '''
    beta_rad = math.radians(beta)
    SPI_value = alpha * math.tan(beta_rad)
    return SPI_value

def TWI_df(dataframe):
    dataframe['tWi'] = np.nan
    dataframe['twi'] = [TWI_calcu(a,b) for a,b in zip(dataframe['contri_area'],dataframe['gradient_deg'])]
    return dataframe

def SPI_df(dataframe):
    dataframe['spi'] = np.nan
    dataframe['spi'] = [SPI_calcu(a,b) for a,b in zip(dataframe['contri_area'],dataframe['gradient_deg'])]
    return dataframe
