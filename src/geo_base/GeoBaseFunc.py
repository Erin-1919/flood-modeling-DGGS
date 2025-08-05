"""
Geographic Base Functions for Flood Modeling DGGS

This module provides core geographic functions for spatial data processing,
coordinate transformations, and interpolation methods used in the flood modeling framework.

Author: Erin Li
"""

import rasterio
import numpy as np
from scipy import interpolate
from pyproj import Proj, transform
import warnings
from typing import Tuple, List, Union, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import local modules
try:
    import DgBaseFunc as dbfc
except ImportError:
    logger.warning("DgBaseFunc module not found. Some functions may not work properly.")

def catch(func, handle=lambda e: e, *args, **kwargs):
    """
    Handle exceptions in a general function with optional custom error handling.
    
    Args:
        func: Function to execute
        handle: Error handler function (default: return exception)
        *args: Arguments to pass to func
        **kwargs: Keyword arguments to pass to func
        
    Returns:
        Result of func(*args, **kwargs) or handle(exception)
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.debug(f"Exception caught in {func.__name__}: {e}")
        return handle(e)
    
def reproject_coords(x: float, y: float, source_epsg: str = 'epsg:4326', 
                    target_epsg: str = 'epsg:3979') -> List[float]:
    """
    Reproject a coordinate pair to a new Coordinate Reference System (CRS).
    
    Args:
        x: X coordinate (longitude)
        y: Y coordinate (latitude)
        source_epsg: EPSG code for the source CRS (default: 4326 - WGS84)
        target_epsg: EPSG code for the target CRS (default: 3979 - NAD83)
        
    Returns:
        List containing [new_x, new_y] coordinates in target CRS
        
    Raises:
        ValueError: If coordinates are invalid
        RuntimeError: If projection transformation fails
    """
    try:
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            raise ValueError("Coordinates must be numeric values")
            
        inProj = Proj(init=source_epsg)
        outProj = Proj(init=target_epsg)
        new_x, new_y = transform(inProj, outProj, x, y)
        
        return [new_x, new_y]
    except Exception as e:
        logger.error(f"Coordinate reprojection failed: {e}")
        raise RuntimeError(f"Failed to reproject coordinates: {e}")

def find_neighbor(x: float, y: float, raster) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """
    Find neighboring grid cells for interpolation.
    
    Determines the 4 neighboring grid cells around a given coordinate point
    and extracts their elevation values for interpolation.
    
    Args:
        x: X coordinate (longitude)
        y: Y coordinate (latitude)
        raster: Rasterio dataset object
        
    Returns:
        Tuple containing:
        - x_array: Array of X coordinates for neighboring cells
        - y_array: Array of Y coordinates for neighboring cells
        - z_array: List of elevation values for neighboring cells
        
    Raises:
        ValueError: If coordinates are outside raster bounds
        RuntimeError: If raster sampling fails
    """
    try:
        # Convert coordinates to raster indices
        x_index, y_index = rasterio.transform.rowcol(raster.transform, x, y)
        
        # Get center coordinates of the grid cell
        xc, yc = rasterio.transform.xy(raster.transform, x_index, y_index)
        
        # Determine which quadrant the point falls into and get appropriate neighbors
        if x > xc and y > yc:
            x_index_array = [x_index-1, x_index-1, x_index, x_index]
            y_index_array = [y_index, y_index+1, y_index, y_index+1]
        elif x > xc and y < yc:
            x_index_array = [x_index, x_index, x_index+1, x_index+1]
            y_index_array = [y_index, y_index+1, y_index, y_index+1]
        elif x < xc and y > yc:
            x_index_array = [x_index-1, x_index-1, x_index, x_index]
            y_index_array = [y_index-1, y_index, y_index-1, y_index]
        elif x < xc and y < yc:
            x_index_array = [x_index, x_index, x_index+1, x_index+1]
            y_index_array = [y_index-1, y_index, y_index-1, y_index]
        else:
            # Point is exactly on grid center
            x_index_array = [x_index-1, x_index-1, x_index, x_index]
            y_index_array = [y_index-1, y_index, y_index-1, y_index]
        
        # Convert indices back to coordinates
        x_array, y_array = rasterio.transform.xy(raster.transform, x_index_array, y_index_array)
        
        # Sample elevation values at neighbor coordinates
        coords = [(lon, lat) for lon, lat in zip(x_array, y_array)]
        z_array = [i[0] for i in raster.sample(coords)]
        
        return x_array, y_array, z_array
        
    except Exception as e:
        logger.error(f"Failed to find neighbors for coordinates ({x}, {y}): {e}")
        raise RuntimeError(f"Neighbor finding failed: {e}")

def bilinear_interp(x: float, y: float, raster, interp: str = 'linear') -> float:
    """
    Perform bilinear interpolation on raster data.
    
    Interpolates elevation values using bilinear interpolation from neighboring
    grid cells. Returns -32767 for no-data values or interpolation failures.
    
    Args:
        x: X coordinate (longitude)
        y: Y coordinate (latitude)
        raster: Rasterio dataset object
        interp: Interpolation method ('linear', 'cubic', 'nearest')
        
    Returns:
        Interpolated elevation value or -32767 for no-data
        
    Note:
        Returns -32767 if:
        - Interpolation fails
        - Point or neighbors have no-data values
        - Coordinates are outside raster bounds
    """
    try:
        # Find neighboring cells
        x_array, y_array, z_array = find_neighbor(x, y, raster)
        
        # Check for no-data values
        if raster.nodata in z_array or any(np.isnan(z) for z in z_array):
            logger.debug(f"No-data value found at coordinates ({x}, {y})")
            return -32767
        
        # Perform interpolation
        interp_func = interpolate.interp2d(x_array, y_array, z_array, kind=interp)
        interpolated_value = interp_func(x, y)[0]
        
        # Validate result
        if np.isnan(interpolated_value) or np.isinf(interpolated_value):
            logger.debug(f"Invalid interpolation result at ({x}, {y}): {interpolated_value}")
            return -32767
            
        return float(interpolated_value)
        
    except Exception as e:
        logger.debug(f"Bilinear interpolation failed at ({x}, {y}): {e}")
        return -32767

def IDW_interp(z: np.ndarray, dist: np.ndarray, power: float = 2.0) -> np.ndarray:
    """
    Perform Inverse Distance Weighted (IDW) interpolation.
    
    Calculates weighted average of known values based on inverse distance
    to interpolation points.
    
    Args:
        z: Numpy array of known values at reference points
        dist: Numpy array of distances from interpolation points to reference points
        power: Power parameter for distance weighting (default: 2.0)
        
    Returns:
        Numpy array of interpolated values
        
    Raises:
        ValueError: If arrays have incompatible shapes or invalid values
        RuntimeError: If interpolation fails due to numerical issues
    """
    try:
        # Validate inputs
        if z.size == 0 or dist.size == 0:
            raise ValueError("Input arrays cannot be empty")
            
        if power <= 0:
            raise ValueError("Power parameter must be positive")
            
        # Add small epsilon to avoid division by zero
        epsilon = 1e-12
        idist = 1.0 / (dist + epsilon)**power
        
        # Calculate weighted average
        weighted_sum = np.sum(z[None, :] * idist, axis=1)
        weight_sum = np.sum(idist, axis=1)
        
        # Avoid division by zero
        idw_values = np.where(weight_sum > 0, weighted_sum / weight_sum, np.nan)
        
        return idw_values
        
    except Exception as e:
        logger.error(f"IDW interpolation failed: {e}")
        raise RuntimeError(f"IDW interpolation failed: {e}")

def nearest_interp_df(df, tif: str, var: str, x_col: str, y_col: str):
    """
    Extract values from raster using nearest neighbor interpolation.
    
    Samples raster values at specified coordinates using nearest neighbor
    interpolation and adds results to the dataframe.
    
    Args:
        df: Pandas DataFrame containing coordinates
        tif: Path to raster file (GeoTIFF)
        var: Column name for extracted values
        x_col: Column name containing X coordinates
        y_col: Column name containing Y coordinates
        
    Returns:
        DataFrame with new column containing interpolated values
        
    Raises:
        FileNotFoundError: If raster file doesn't exist
        ValueError: If coordinate columns don't exist
        RuntimeError: If raster sampling fails
    """
    try:
        # Validate inputs
        if not all(col in df.columns for col in [x_col, y_col]):
            raise ValueError(f"Coordinate columns {x_col}, {y_col} not found in DataFrame")
            
        # Open raster file
        with rasterio.open(tif) as TIF:
            # Prepare coordinates
            coords = [(lon, lat) for lon, lat in zip(df[x_col], df[y_col])]
            
            # Sample raster values
            sampled_values = []
            for coord in coords:
                try:
                    value = list(TIF.sample([coord]))[0][0]
                    sampled_values.append(value)
                except Exception as e:
                    logger.debug(f"Sampling failed at {coord}: {e}")
                    sampled_values.append(np.nan)
            
            # Add results to dataframe
            df[var] = sampled_values
            
        return df
        
    except Exception as e:
        logger.error(f"Nearest neighbor interpolation failed: {e}")
        raise RuntimeError(f"Raster sampling failed: {e}")

def bilinear_interp_df(df, tif: str, var: str, x_col: str, y_col: str):
    """
    Extract values from raster using bilinear interpolation.
    
    Samples raster values at specified coordinates using bilinear interpolation
    and adds results to the dataframe.
    
    Args:
        df: Pandas DataFrame containing coordinates
        tif: Path to raster file (GeoTIFF)
        var: Column name for extracted values
        x_col: Column name containing X coordinates
        y_col: Column name containing Y coordinates
        
    Returns:
        DataFrame with new column containing interpolated values
        
    Raises:
        FileNotFoundError: If raster file doesn't exist
        ValueError: If coordinate columns don't exist
        RuntimeError: If interpolation fails
    """
    try:
        # Validate inputs
        if not all(col in df.columns for col in [x_col, y_col]):
            raise ValueError(f"Coordinate columns {x_col}, {y_col} not found in DataFrame")
            
        # Open raster file
        with rasterio.open(tif) as TIF:
            # Perform bilinear interpolation for each coordinate
            interpolated_values = []
            for lon, lat in zip(df[x_col], df[y_col]):
                try:
                    value = bilinear_interp(lon, lat, TIF)
                    interpolated_values.append(value)
                except Exception as e:
                    logger.debug(f"Bilinear interpolation failed at ({lon}, {lat}): {e}")
                    interpolated_values.append(-32767)
            
            # Add results to dataframe
            df[var] = interpolated_values
            
        return df
        
    except Exception as e:
        logger.error(f"Bilinear interpolation failed: {e}")
        raise RuntimeError(f"Raster interpolation failed: {e}")

def IDW_interp_df(df, cdf, normal: str, month: int, var: str, res: int):
    """
    Perform Inverse Distance Weighted (IDW) interpolation on climate data.
    
    Interpolates climate station data to grid points using IDW method
    based on hexagonal distances in DGGS.
    
    Args:
        df: Target DataFrame with grid coordinates (i, j)
        cdf: Climate data DataFrame with station data
        normal: Climate normal identifier
        month: Month number (1-12)
        var: Variable name for output column
        res: DGGS resolution
        
    Returns:
        DataFrame with interpolated climate values
        
    Raises:
        ValueError: If required columns are missing or data is invalid
        RuntimeError: If interpolation fails
    """
    try:
        # Validate inputs
        required_cols = ['MONTH', 'E_NORMAL_E', 'VALUE', 'i', 'j']
        if not all(col in cdf.columns for col in required_cols):
            raise ValueError(f"Climate data missing required columns: {required_cols}")
            
        if not all(col in df.columns for col in ['i', 'j']):
            raise ValueError("Target DataFrame missing coordinate columns: i, j")
            
        # Filter climate data for specific month and normal
        filtered_cdf = cdf[(cdf['MONTH'] == month) & (cdf['E_NORMAL_E'] == normal)]
        
        if filtered_cdf.empty:
            logger.warning(f"No climate data found for month {month} and normal {normal}")
            df[var] = np.nan
            return df
        
        # Prepare coordinates and values
        cdf_coords = [(i, j) for i, j in zip(filtered_cdf.i, filtered_cdf.j)]
        sdf_coords = [(i, j) for i, j in zip(df.i, df.j)]
        station_values = filtered_cdf['VALUE'].to_numpy()
        
        # Calculate hexagonal distances
        ring_arr = np.array([[dbfc.hex_dist(ij1, ij2, res) for ij2 in cdf_coords] 
                           for ij1 in sdf_coords])
        
        # Perform IDW interpolation
        interpolated_values = IDW_interp(station_values, ring_arr)
        df[var] = interpolated_values
        
        return df
        
    except Exception as e:
        logger.error(f"IDW interpolation failed: {e}")
        raise RuntimeError(f"Climate data interpolation failed: {e}")

def reproject_coords_df(df, x_col: str, y_col: str, x_new: str, y_new: str):
    """
    Reproject coordinate columns in a DataFrame to a new CRS.
    
    Converts coordinates from source CRS to target CRS and adds
    new columns with reprojected coordinates.
    
    Args:
        df: Pandas DataFrame containing coordinates
        x_col: Column name containing X coordinates
        y_col: Column name containing Y coordinates
        x_new: New column name for reprojected X coordinates
        y_new: New column name for reprojected Y coordinates
        
    Returns:
        DataFrame with new reprojected coordinate columns
        
    Raises:
        ValueError: If coordinate columns don't exist
        RuntimeError: If reprojection fails
    """
    try:
        # Validate inputs
        if not all(col in df.columns for col in [x_col, y_col]):
            raise ValueError(f"Coordinate columns {x_col}, {y_col} not found in DataFrame")
            
        # Reproject coordinates
        reprojected_coords = []
        for x, y in zip(df[x_col], df[y_col]):
            try:
                new_coords = reproject_coords(x, y)
                reprojected_coords.append(new_coords)
            except Exception as e:
                logger.debug(f"Reprojection failed for ({x}, {y}): {e}")
                reprojected_coords.append([np.nan, np.nan])
        
        # Add new columns
        df[x_new] = [coord[0] for coord in reprojected_coords]
        df[y_new] = [coord[1] for coord in reprojected_coords]
        
        return df
        
    except Exception as e:
        logger.error(f"Coordinate reprojection failed: {e}")
        raise RuntimeError(f"DataFrame reprojection failed: {e}")

if __name__ == '__main__':
    """
    Example usage of GeoBaseFunc module.
    """
    import pandas as pd
    
    # Example: Create sample data
    sample_data = pd.DataFrame({
        'lon': [-75.0, -74.5, -74.0],
        'lat': [45.0, 45.5, 46.0]
    })
    
    print("GeoBaseFunc module loaded successfully!")
    print("Available functions:")
    print("- reproject_coords(): Reproject coordinate pairs")
    print("- bilinear_interp(): Bilinear interpolation")
    print("- IDW_interp(): Inverse Distance Weighted interpolation")
    print("- nearest_interp_df(): Nearest neighbor interpolation on DataFrame")
    print("- bilinear_interp_df(): Bilinear interpolation on DataFrame")
    print("- IDW_interp_df(): IDW interpolation on climate data")
    print("- reproject_coords_df(): Reproject coordinates in DataFrame")

