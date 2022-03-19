import rasterio
import numpy as np
from scipy import interpolate
from pyproj import Proj, transform
import DgBaseFunc as dbfc

def catch(func, handle = lambda e : e, *args, **kwargs):
    ''' Handle the try except in a general function'''
    try:
        return func(*args, **kwargs)
    except:
        return np.nan
    
def reproject_coords(x,y,source_epsg='epsg:4326',target_epsg='epsg:3979'):
    '''
    Reproject a coord pair to a new CRS.
    ---------------------------------------
    coords        |    original (x,y) coords
    source_epsg   |    EPSG code for the source CRS (default 4326)
    target_epsg   |    EPSG code for the target CRS (default 3979)
    '''
    inProj = Proj(init=source_epsg)
    outProj = Proj(init=target_epsg)
    new_x,new_y = transform(inProj,outProj,x,y)
    return [new_x,new_y]

def find_neighbor(x,y,raster):
    ''' 
    Find neighbors for interpolation --
    determine the DEM source
    find out the 4 neighbors geographic coords
    extract the elevations at these 4 coords
    convert 4 coords to array index then back to 4 coords of grid mesh centers
    '''
    x_index,y_index = rasterio.transform.rowcol(raster.transform, x,y)
    xc,yc = rasterio.transform.xy(raster.transform, x_index, y_index)
    if x > xc and y > yc:
        x_index_array = [x_index-1,x_index-1,x_index,x_index]
        y_index_array = [y_index,y_index+1,y_index,y_index+1]
    elif x > xc and y < yc:
        x_index_array = [x_index,x_index,x_index+1,x_index+1]
        y_index_array = [y_index,y_index+1,y_index,y_index+1]
    elif x < xc and y > yc:
        x_index_array = [x_index-1,x_index-1,x_index,x_index]
        y_index_array = [y_index-1,y_index,y_index-1,y_index]
    elif x < xc and y < yc:
        x_index_array = [x_index,x_index,x_index+1,x_index+1]
        y_index_array = [y_index-1,y_index,y_index-1,y_index]
    x_array,y_array = rasterio.transform.xy(raster.transform, x_index_array, y_index_array)
    coords = [(lon,lat) for lon, lat in zip(x_array,y_array)]
    z_array = [i[0] for i in raster.sample(coords)]
    return x_array, y_array, z_array

def bilinear_interp(x,y,raster,interp = 'linear'):
    ''' 
    Bilinear interpolation -- 
    if an error is raised then return -32767 as its final value
    if the point or any of its neighbors has no data then return -32767 
    '''
    try:
        x_array, y_array, z_array = find_neighbor(x,y,raster)
        if raster.nodata in z_array:
            return -32767
        else:
            interp = interpolate.interp2d(x_array, y_array, z_array, kind=interp)
            v = interp(x,y)[0]
            return v
    except:
        return -32767

def IDW_interp(z,dist,power=2):
    """
    Inverse Distance Weighted interpolation
    ---------------------------------------
    z         |    numpy array of known values
    dist      |    numpy array of gridded dist
    power     |    power on distance
    """
    # calculate the inverse distances with a small e to avoid infinities
    idist = 1.0 / (dist + 1e-12)**power
    # calculate the weighted average
    idw_values = np.sum(z[None,:] * idist, axis=1) / np.sum(idist, axis=1)
    return idw_values

def nearest_interp_df(df,tif,var,x_col,y_col):
    ''' Extract value from tif '''
    TIF = rasterio.open(tif)
    coords = [(lon,lat) for lon, lat in zip(df[x_col], df[y_col])]
    df[var] = [catch(lambda: i[0]) for i in TIF.sample(coords)]
    # df[var] = [x[0] if x.mask.ndim == 1 and not x.mask[0] else np.nan for x in rasterio.sample.sample_gen(TIF, coords, masked=True)]
    return df

def bilinear_interp_df(df,tif,var,x_col,y_col):
    ''' Extract value from tif '''
    TIF = rasterio.open(tif)
    df[var] = [bilinear_interp(lon,lat,TIF) for lon, lat in zip(df[x_col], df[y_col])]
    return df

def IDW_interp_df(df,cdf,normal,month,var,res):
    ''' IDW interpolation '''
    cdf = cdf[cdf['MONTH'] == month]
    cdf = cdf[cdf['E_NORMAL_E'] == normal]
    cdf_coords = [(i,j) for i,j in zip(cdf.i,cdf.j)]
    sdf_coords = [(i,j) for i,j in zip(df.i,df.j)]
    station_value = cdf['VALUE'].to_numpy()
    ring_arr = np.array([[dbfc.hex_dist(ij1,ij2,res) for ij2 in cdf_coords] for ij1 in sdf_coords])
    df[var] = IDW_interp(station_value,ring_arr)
    return df

def reproject_coords_df(df,x_col,y_col,x_new,y_new):
    df[[x_new,y_new]] = [reproject_coords(x,y) for x,y in zip(df[x_col], df[y_col])]
    return df

if __name__=='__main__':
    pass

