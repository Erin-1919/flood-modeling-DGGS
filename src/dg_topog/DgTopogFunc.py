"""
DGGS Topographic Functions for Flood Modeling

This module provides topographic analysis functions for Discrete Global Grid System (DGGS)
operations, including slope calculation using multiple algorithms, aspect computation,
curvature analysis, roughness estimation, and topographic indices (TRI, TPI).

The module implements various slope calculation methods adapted for hexagonal DGGS cells:
- MAG: Maximum Absolute Gradient
- MDG: Maximum Downward Gradient  
- MDN: Multiple Downhill Neighbors
- FDA: Finite-Difference Algorithm
- BFP: Best Fit Plane

Author: Erin Li
"""

import math
import numpy as np
import logging
from typing import List, Tuple, Union, Optional, Dict, Any
import warnings

# Import base functions
import DgBaseFunc as dbfc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
NO_DATA_VALUE = -32767
FLAT_ASPECT = -1
DEFAULT_RINGS = 1
DEFAULT_ALTITUDE = 45
DEFAULT_AZIMUTH = 315


def slope_MAG(coords: Tuple[int, int], df: pd.DataFrame, res: int, cell_spacing: float) -> Tuple[float, float]:
    """
    Calculate slope using Maximum Absolute Gradient method.
    
    Determines the absolute maximum differences between the center cell and its six neighbors.
    If edge cell, assigns NaN values. If flat, slope = 0 and aspect = -1.
    If multiple equal gradients found, chooses the first neighbor encountered clockwise from north.
    
    Args:
        coords: Cell coordinates (i, j)
        df: DataFrame containing elevation data
        res: DGGS resolution level
        cell_spacing: Cell spacing in degrees
        
    Returns:
        Tuple of (slope_angle, aspect_angle) in degrees
        
    Raises:
        ValueError: If coordinates are invalid
        RuntimeError: If calculation fails
    """
    try:
        if not isinstance(coords, tuple) or len(coords) != 2:
            raise ValueError("Coordinates must be a tuple of (i, j)")
            
        elev_neighbor = dbfc.neighbor_navig(res, coords, df)
        
        if dbfc.edge_cell_exist(elev_neighbor):
            return np.nan, np.nan
        else:
            # Calculate gradients
            gradient = [e - elev_neighbor[0] for e in elev_neighbor]
            gradient_abs = [abs(e) for e in gradient]
            gradient_max = max(gradient_abs)
            
            if gradient_max == 0:
                gradient_mag = 0
                aspect_mag = FLAT_ASPECT
            else:
                gradient_rad_mag = math.atan(gradient_max / cell_spacing)
                gradient_mag = math.degrees(gradient_rad_mag)
                aspect_index = gradient_abs.index(gradient_max)
                
                if gradient[aspect_index] > 0:
                    aspect_mag = dbfc.aspect_restricted_oppo(res)[aspect_index - 1]
                elif gradient[aspect_index] < 0:
                    aspect_mag = dbfc.aspect_restricted(res)[aspect_index - 1]
                    
            return gradient_mag, aspect_mag
            
    except Exception as e:
        logger.error(f"Slope MAG calculation failed: {e}")
        raise RuntimeError(f"Failed to calculate slope MAG: {e}")


def slope_MDG(coords: Tuple[int, int], df: pd.DataFrame, res: int, cell_spacing: float) -> Tuple[float, float]:
    """
    Calculate slope using Maximum Downward Gradient method.
    
    Requires pit-filling beforehand without saving the altered elevations.
    If edge cell, assigns NaN values. If flat, slope = 0 and aspect = -1.
    If multiple equal gradients found, chooses the first neighbor encountered clockwise from north.
    
    Args:
        coords: Cell coordinates (i, j)
        df: DataFrame containing elevation data
        res: DGGS resolution level
        cell_spacing: Cell spacing in degrees
        
    Returns:
        Tuple of (slope_angle, aspect_angle) in degrees
        
    Raises:
        ValueError: If coordinates are invalid
        RuntimeError: If calculation fails
    """
    try:
        if not isinstance(coords, tuple) or len(coords) != 2:
            raise ValueError("Coordinates must be a tuple of (i, j)")
            
        elev_neighbor = dbfc.neighbor_navig(res, coords, df)
        
        if dbfc.edge_cell_exist(elev_neighbor):
            return np.nan, np.nan
        else:
            elev_neighbor_ = elev_neighbor[1:]
            if all(i > elev_neighbor[0] for i in elev_neighbor_):
                elev_neighbor[0] = min(elev_neighbor_)
                
            # Calculate gradients
            gradient = [e - elev_neighbor[0] for e in elev_neighbor]
            gradient_min = min(gradient)
            
            if gradient_min >= 0:
                gradient_mdg = 0
                aspect_mdg = FLAT_ASPECT
            else:
                gradient_rad_mdg = math.atan(abs(gradient_min) / cell_spacing)
                gradient_mdg = math.degrees(gradient_rad_mdg)
                aspect_index = gradient.index(gradient_min)
                aspect_mdg = dbfc.aspect_restricted(res)[aspect_index - 1]
                
            return gradient_mdg, aspect_mdg
            
    except Exception as e:
        logger.error(f"Slope MDG calculation failed: {e}")
        raise RuntimeError(f"Failed to calculate slope MDG: {e}")


def slope_MDN(coords: Tuple[int, int], df: pd.DataFrame, res: int, cell_spacing: float) -> Tuple[float, float]:
    """
    Calculate slope using Multiple Downhill Neighbors method.
    
    Distributes flow from a pixel amongst all of its lower elevation neighbor pixels.
    Requires pit-filling beforehand without saving the altered elevations.
    If edge cell, assigns NaN values. If flat, slope = 0 and aspect = -1.
    
    Args:
        coords: Cell coordinates (i, j)
        df: DataFrame containing elevation data
        res: DGGS resolution level
        cell_spacing: Cell spacing in degrees
        
    Returns:
        Tuple of (slope_angle, aspect_angle) in degrees
        
    Raises:
        ValueError: If coordinates are invalid
        RuntimeError: If calculation fails
    """
    try:
        if not isinstance(coords, tuple) or len(coords) != 2:
            raise ValueError("Coordinates must be a tuple of (i, j)")
            
        elev_neighbor = dbfc.neighbor_navig(res, coords, df)
        
        if dbfc.edge_cell_exist(elev_neighbor):
            return np.nan, np.nan
        else:
            elev_neighbor_ = elev_neighbor[1:]
            if all(i > elev_neighbor[0] for i in elev_neighbor_):
                elev_neighbor[0] = min(elev_neighbor_)
                
            # Calculate gradients
            gradient = [e - elev_neighbor[0] for e in elev_neighbor]
            
            if min(gradient) >= 0:
                gradient_mdn = 0
                aspect_mdn = FLAT_ASPECT
            else:
                gradient_rad_mdn_ls = [math.atan(abs(g) / cell_spacing) for g in gradient if g < 0]
                gradient_rad_mdn = sum(gradient_rad_mdn_ls) / len(gradient_rad_mdn_ls)
                gradient_mdn = math.degrees(gradient_rad_mdn)
                
                # Find downhill cells
                cell_ls = []
                for i, g in enumerate(gradient):
                    if g < 0:
                        cell_ls.append(i)
                        
                if len(cell_ls) == 1:
                    aspect_index = cell_ls[0]
                    aspect_mdn = dbfc.aspect_restricted(res)[aspect_index - 1]
                elif len(cell_ls) == 2 and abs(cell_ls[0] - cell_ls[1]) == 3:
                    aspect_index_1, aspect_index_2 = cell_ls[0], cell_ls[1]
                    gradient_1, gradient_2 = gradient[aspect_index_1], gradient[aspect_index_2]
                    aspect_index = aspect_index_1 if gradient_1 <= gradient_2 else aspect_index_2
                    aspect_mdn = dbfc.aspect_restricted(res)[aspect_index - 1]
                else:
                    avg_norm = dbfc.mean_norm_vector(res, elev_neighbor, *cell_ls)
                    aspect_mdn = dbfc.aspect_unrestricted(avg_norm[0], avg_norm[1])
                    aspect_mdn = math.degrees(aspect_mdn)
                    
                if aspect_mdn < 0:
                    aspect_mdn += 360
                    
            return gradient_mdn, aspect_mdn
            
    except Exception as e:
        logger.error(f"Slope MDN calculation failed: {e}")
        raise RuntimeError(f"Failed to calculate slope MDN: {e}")


def slope_FDA(coords: Tuple[int, int], df: pd.DataFrame, res: int, cell_spacing: float) -> Tuple[float, float]:
    """
    Calculate slope using Finite-Difference Algorithm.
    
    Projects the non-normalized gradient in three directions to orthogonal x, y axes.
    If edge cell, assigns NaN values. If flat, slope = 0 and aspect = -1.
    
    Args:
        coords: Cell coordinates (i, j)
        df: DataFrame containing elevation data
        res: DGGS resolution level
        cell_spacing: Cell spacing in degrees
        
    Returns:
        Tuple of (slope_angle, aspect_angle) in degrees
        
    Raises:
        ValueError: If coordinates are invalid
        RuntimeError: If calculation fails
    """
    try:
        if not isinstance(coords, tuple) or len(coords) != 2:
            raise ValueError("Coordinates must be a tuple of (i, j)")
            
        elev_neighbor = dbfc.neighbor_navig(res, coords, df)
        
        if dbfc.edge_cell_exist(elev_neighbor):
            return np.nan, np.nan
        else:
            dzx, dzy = dbfc.first_derivative(res, elev_neighbor)
            
            if dzx == 0 and dzy == 0:
                gradient_fda = 0
                aspect_fda = FLAT_ASPECT
            else:
                gradient_rad_fda = math.atan(math.sqrt(dzx**2 + dzy**2) / cell_spacing)
                gradient_fda = math.degrees(gradient_rad_fda)
                aspect_fda = dbfc.aspect_unrestricted(dzx, dzy)
                aspect_fda = math.degrees(aspect_fda)
                
            if aspect_fda < 0:
                aspect_fda += 360
                
            return gradient_fda, aspect_fda
            
    except Exception as e:
        logger.error(f"Slope FDA calculation failed: {e}")
        raise RuntimeError(f"Failed to calculate slope FDA: {e}")


def slope_BFP(coords: Tuple[int, int], df: pd.DataFrame, res: int, cell_spacing: float) -> Tuple[float, float]:
    """
    Calculate slope using Best Fit Plane method.
    
    Fits a surface to seven centroids by multiple linear regression models,
    using least squares to minimize the sum of distances from the surface to the cells.
    If edge cell, assigns NaN values. If flat, slope = 0 and aspect = -1.
    
    Args:
        coords: Cell coordinates (i, j)
        df: DataFrame containing elevation data
        res: DGGS resolution level
        cell_spacing: Cell spacing in degrees
        
    Returns:
        Tuple of (slope_angle, aspect_angle) in degrees
        
    Raises:
        ValueError: If coordinates are invalid
        RuntimeError: If calculation fails
    """
    try:
        if not isinstance(coords, tuple) or len(coords) != 2:
            raise ValueError("Coordinates must be a tuple of (i, j)")
            
        elev_neighbor = dbfc.neighbor_navig(res, coords, df)
        
        if dbfc.edge_cell_exist(elev_neighbor):
            return np.nan, np.nan
        else:
            norm_vec = np.array(dbfc.fit_plane_norm_vector(res, elev_neighbor))
            norm_vec_mag = np.linalg.norm(norm_vec)
            unit_norm_vec = np.array([i / norm_vec_mag for i in norm_vec])
            ref_vec = np.array([0, 0, -1])
            ref_vec_proj = ref_vec - (np.dot(ref_vec, unit_norm_vec) * unit_norm_vec)
            
            try:
                gradient_rad_bfp = math.atan(
                    (ref_vec_proj[2]**2 / math.sqrt(ref_vec_proj[0]**2 + ref_vec_proj[1]**2)) / cell_spacing
                )
                gradient_bfp = math.degrees(gradient_rad_bfp)
                aspect_bfp = dbfc.aspect_unrestricted(ref_vec_proj[0], ref_vec_proj[1])
                aspect_bfp = math.degrees(aspect_bfp)
            except:
                gradient_bfp = 0
                aspect_bfp = FLAT_ASPECT
                
            if aspect_bfp < 0:
                aspect_bfp += 360
                
            return gradient_bfp, aspect_bfp
            
    except Exception as e:
        logger.error(f"Slope BFP calculation failed: {e}")
        raise RuntimeError(f"Failed to calculate slope BFP: {e}")


def curvature(coords: Tuple[int, int], df: pd.DataFrame, res: int, cell_spacing: float) -> float:
    """
    Calculate curvature as the rate of change of landform.
    
    Curvature is the second derivative of DEM and first derivative of slope.
    Represents the rate of change of slope.
    
    Args:
        coords: Cell coordinates (i, j)
        df: DataFrame containing elevation data
        res: DGGS resolution level
        cell_spacing: Cell spacing in degrees
        
    Returns:
        Curvature value or NaN if calculation not possible
        
    Raises:
        ValueError: If coordinates are invalid
        RuntimeError: If calculation fails
    """
    try:
        if not isinstance(coords, tuple) or len(coords) != 2:
            raise ValueError("Coordinates must be a tuple of (i, j)")
            
        elev_neighbor = dbfc.neighbor_navig(res, coords, df)
        
        if dbfc.edge_cell_exist(elev_neighbor):
            return np.nan
        else:
            dzx2, dzy2 = dbfc.second_derivative(res, elev_neighbor)
            curv = math.sqrt(dzx2**2 + dzy2**2) / cell_spacing**2
            return curv
            
    except Exception as e:
        logger.error(f"Curvature calculation failed: {e}")
        raise RuntimeError(f"Failed to calculate curvature: {e}")


def hillshade(coords: Tuple[int, int], df: pd.DataFrame, res: int, 
              altitude: float = DEFAULT_ALTITUDE, azimuth: float = DEFAULT_AZIMUTH) -> float:
    """
    Calculate hillshade for 3D surface representation.
    
    A hillshade is a grayscale 3D representation of the surface, with the sun's
    relative position taken into account for shading. Uses altitude and azimuth
    properties to specify the sun's position.
    
    Args:
        coords: Cell coordinates (i, j)
        df: DataFrame containing elevation data
        res: DGGS resolution level
        altitude: Slope/angle of illumination source above horizon (0-90 degrees)
        azimuth: Angular direction of sun, measured from north clockwise (0-360 degrees)
        
    Returns:
        Hillshade value (0-255)
        
    Raises:
        ValueError: If coordinates are invalid
        RuntimeError: If calculation fails
    """
    try:
        if not isinstance(coords, tuple) or len(coords) != 2:
            raise ValueError("Coordinates must be a tuple of (i, j)")
            
        slope_deg, aspect_deg = slope_FDA(coords, df, res, 1.0)  # Use default cell spacing
        slope_rad = math.radians(slope_deg)
        aspect_rad = math.radians(aspect_deg)
        
        if slope_rad == 0:
            hs = 255.0
        else:
            zenith_deg = 90.0 - altitude
            zenith_rad = math.radians(zenith_deg)
            azimuth_math = 360.0 - azimuth + 90.0
            
            if azimuth_math >= 360.0:
                azimuth_math = azimuth_math - 360.0
                
            azimuth_rad = math.radians(azimuth_math)
            hs = 255.0 * (
                (math.cos(zenith_rad) * math.cos(slope_rad)) + 
                (math.sin(zenith_rad) * math.sin(slope_rad) * math.cos(azimuth_rad - aspect_rad))
            )
            
        return hs
        
    except Exception as e:
        logger.error(f"Hillshade calculation failed: {e}")
        raise RuntimeError(f"Failed to calculate hillshade: {e}")


def roughness(coords: Tuple[int, int], df: pd.DataFrame, res: int, 
              vertical_res: int, rings: int = DEFAULT_RINGS) -> float:
    """
    Calculate terrain roughness as the absolute difference between max and min in neighborhood.
    
    Removes voids in neighbors and calculates terrain roughness as the absolute
    difference between maximum and minimum elevation in the neighborhood.
    
    Args:
        coords: Cell coordinates (i, j)
        df: DataFrame containing elevation data
        res: DGGS resolution level
        vertical_res: Vertical resolution for rounding
        rings: Number of rings around center cell to include
        
    Returns:
        Roughness value or NaN if calculation not possible
        
    Raises:
        ValueError: If coordinates are invalid
        RuntimeError: If calculation fails
    """
    try:
        if not isinstance(coords, tuple) or len(coords) != 2:
            raise ValueError("Coordinates must be a tuple of (i, j)")
            
        elev_neighbor = dbfc.neighbor_navig_by_rings(res, coords, df, rings)
        elev_neighbor = [x for x in elev_neighbor if x != NO_DATA_VALUE and not np.isnan(x)]
        
        if len(elev_neighbor) == 0:
            return np.nan
        if len(elev_neighbor) == 1:
            return 0
        else:
            elev_max = round(max(elev_neighbor), vertical_res)
            elev_min = round(min(elev_neighbor), vertical_res)
            elev_range = round(elev_max - elev_min, vertical_res)
            return elev_range
            
    except Exception as e:
        logger.error(f"Roughness calculation failed: {e}")
        raise RuntimeError(f"Failed to calculate roughness: {e}")


def TRI_calcu(coords: Tuple[int, int], res: int, elev_df: pd.DataFrame) -> float:
    """
    Calculate Terrain Roughness Index (TRI).
    
    TRI represents the square root of sum of squared differences between a central
    cell and its adjacent cells.
    
    Args:
        coords: Cell coordinates (i, j)
        res: DGGS resolution level
        elev_df: DataFrame containing elevation data
        
    Returns:
        TRI value or NaN if calculation not possible
        
    Raises:
        ValueError: If coordinates are invalid
        RuntimeError: If calculation fails
    """
    try:
        if not isinstance(coords, tuple) or len(coords) != 2:
            raise ValueError("Coordinates must be a tuple of (i, j)")
            
        elev_neighbor = dbfc.neighbor_navig(res, coords, elev_df)
        
        if dbfc.edge_cell_exist(elev_neighbor):
            return np.nan
        else:
            TRI_ls = [(n - elev_neighbor[0]) ** 2 for n in elev_neighbor[1:]]
            TRI_value = math.sqrt(sum(TRI_ls))
            return TRI_value
            
    except Exception as e:
        logger.error(f"TRI calculation failed: {e}")
        raise RuntimeError(f"Failed to calculate TRI: {e}")


def TPI_calcu(coords: Tuple[int, int], res: int, elev_df: pd.DataFrame) -> float:
    """
    Calculate Topographic Position Index (TPI).
    
    TPI represents the difference between a central cell and the mean of its
    surrounding cells.
    
    Args:
        coords: Cell coordinates (i, j)
        res: DGGS resolution level
        elev_df: DataFrame containing elevation data
        
    Returns:
        TPI value or NaN if calculation not possible
        
    Raises:
        ValueError: If coordinates are invalid
        RuntimeError: If calculation fails
    """
    try:
        if not isinstance(coords, tuple) or len(coords) != 2:
            raise ValueError("Coordinates must be a tuple of (i, j)")
            
        elev_neighbor = dbfc.neighbor_navig(res, coords, elev_df)
        
        if dbfc.edge_cell_exist(elev_neighbor):
            return np.nan
        else:
            TPI_ls = elev_neighbor[1:]
            TPI_value = elev_neighbor[0] - (sum(TPI_ls) / len(TPI_ls))
            return TPI_value
            
    except Exception as e:
        logger.error(f"TPI calculation failed: {e}")
        raise RuntimeError(f"Failed to calculate TPI: {e}")


def slope_aspect_df(dataframe: pd.DataFrame, elev_df: pd.DataFrame, method: str, 
                   res: int, cell_spacing: float) -> pd.DataFrame:
    """
    Calculate slope and aspect by specified method.
    
    Args:
        dataframe: DataFrame to store results
        elev_df: DataFrame containing elevation data
        method: Slope calculation method ('MAG', 'MDG', 'MDN', 'FDA', 'BFP')
        res: DGGS resolution level
        cell_spacing: Cell spacing in degrees
        
    Returns:
        DataFrame with added 'slp' and 'asp' columns
        
    Raises:
        ValueError: If method is invalid or input data is invalid
        RuntimeError: If processing fails
    """
    try:
        if dataframe.empty or elev_df.empty:
            raise ValueError("Input DataFrames cannot be empty")
            
        valid_methods = ['MAG', 'MDG', 'MDN', 'FDA', 'BFP']
        if method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")
            
        df = dataframe.copy()
        df['slp'] = df['asp'] = np.nan
        
        if method == 'MAG':
            df[['slp', 'asp']] = [slope_MAG(ij, elev_df, res, cell_spacing) for ij in df.index.values]
        elif method == 'MDG':
            df[['slp', 'asp']] = [slope_MDG(ij, elev_df, res, cell_spacing) for ij in df.index.values]
        elif method == 'MDN':
            df[['slp', 'asp']] = [slope_MDN(ij, elev_df, res, cell_spacing) for ij in df.index.values]
        elif method == 'FDA':
            df[['slp', 'asp']] = [slope_FDA(ij, elev_df, res, cell_spacing) for ij in df.index.values]
        elif method == 'BFP':
            df[['slp', 'asp']] = [slope_BFP(ij, elev_df, res, cell_spacing) for ij in df.index.values]
            
        return df
        
    except Exception as e:
        logger.error(f"Slope and aspect DataFrame processing failed: {e}")
        raise RuntimeError(f"Failed to process slope and aspect DataFrame: {e}")


def curvature_df(dataframe: pd.DataFrame, elev_df: pd.DataFrame, res: int, cell_spacing: float) -> pd.DataFrame:
    """
    Calculate curvature for all cells in a DataFrame.
    
    Args:
        dataframe: DataFrame to store results
        elev_df: DataFrame containing elevation data
        res: DGGS resolution level
        cell_spacing: Cell spacing in degrees
        
    Returns:
        DataFrame with added 'curv' column
        
    Raises:
        ValueError: If input data is invalid
        RuntimeError: If processing fails
    """
    try:
        if dataframe.empty or elev_df.empty:
            raise ValueError("Input DataFrames cannot be empty")
            
        df = dataframe.copy()
        df['curv'] = np.nan
        df['curv'] = [curvature(ij, elev_df, res, cell_spacing) for ij in df.index.values]
        
        return df
        
    except Exception as e:
        logger.error(f"Curvature DataFrame processing failed: {e}")
        raise RuntimeError(f"Failed to process curvature DataFrame: {e}")


def hillshade_df(dataframe: pd.DataFrame, elev_df: pd.DataFrame, res: int) -> pd.DataFrame:
    """
    Calculate hillshade for all cells in a DataFrame.
    
    Args:
        dataframe: DataFrame to store results
        elev_df: DataFrame containing elevation data
        res: DGGS resolution level
        
    Returns:
        DataFrame with added 'hs' column
        
    Raises:
        ValueError: If input data is invalid
        RuntimeError: If processing fails
    """
    try:
        if dataframe.empty or elev_df.empty:
            raise ValueError("Input DataFrames cannot be empty")
            
        df = dataframe.copy()
        df['hs'] = np.nan
        df['hs'] = [hillshade(ij, res, elev_df) for ij in df.index.values]
        
        return df
        
    except Exception as e:
        logger.error(f"Hillshade DataFrame processing failed: {e}")
        raise RuntimeError(f"Failed to process hillshade DataFrame: {e}")


def roughness_df(dataframe: pd.DataFrame, elev_df: pd.DataFrame, res: int, 
                vertical_res: int, rings: int = DEFAULT_RINGS) -> pd.DataFrame:
    """
    Calculate roughness for all cells in a DataFrame.
    
    Args:
        dataframe: DataFrame to store results
        elev_df: DataFrame containing elevation data
        res: DGGS resolution level
        vertical_res: Vertical resolution for rounding
        rings: Number of rings around center cell to include
        
    Returns:
        DataFrame with added 'rgh' column
        
    Raises:
        ValueError: If input data is invalid
        RuntimeError: If processing fails
    """
    try:
        if dataframe.empty or elev_df.empty:
            raise ValueError("Input DataFrames cannot be empty")
            
        df = dataframe.copy()
        df['rgh'] = np.nan
        df['rgh'] = [roughness(ij, elev_df, res, vertical_res, rings) for ij in df.index.values]
        
        return df
        
    except Exception as e:
        logger.error(f"Roughness DataFrame processing failed: {e}")
        raise RuntimeError(f"Failed to process roughness DataFrame: {e}")


def TRI_df(dataframe: pd.DataFrame, res: int, elev_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Terrain Roughness Index for all cells in a DataFrame.
    
    Args:
        dataframe: DataFrame to store results
        elev_df: DataFrame containing elevation data
        res: DGGS resolution level
        
    Returns:
        DataFrame with added 'tri' column
        
    Raises:
        ValueError: If input data is invalid
        RuntimeError: If processing fails
    """
    try:
        if dataframe.empty or elev_df.empty:
            raise ValueError("Input DataFrames cannot be empty")
            
        df = dataframe.copy()
        df['tri'] = np.nan
        df['tri'] = [TRI_calcu(ij, res, elev_df) for ij in df.index.values]
        
        return df
        
    except Exception as e:
        logger.error(f"TRI DataFrame processing failed: {e}")
        raise RuntimeError(f"Failed to process TRI DataFrame: {e}")


def TPI_df(dataframe: pd.DataFrame, res: int, elev_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Topographic Position Index for all cells in a DataFrame.
    
    Args:
        dataframe: DataFrame to store results
        elev_df: DataFrame containing elevation data
        res: DGGS resolution level
        
    Returns:
        DataFrame with added 'tpi' column
        
    Raises:
        ValueError: If input data is invalid
        RuntimeError: If processing fails
    """
    try:
        if dataframe.empty or elev_df.empty:
            raise ValueError("Input DataFrames cannot be empty")
            
        df = dataframe.copy()
        df['tpi'] = np.nan
        df['tpi'] = [TPI_calcu(ij, res, elev_df) for ij in df.index.values]
        
        return df
        
    except Exception as e:
        logger.error(f"TPI DataFrame processing failed: {e}")
        raise RuntimeError(f"Failed to process TPI DataFrame: {e}")


if __name__ == '__main__':
    """
    Example usage of topographic functions.
    
    This section demonstrates how to use the key functions in this module.
    """
    # Example: Calculate slope using different methods
    coords = (100, 200)
    resolution = 23
    cell_spacing = 0.0001
    
    # Create sample elevation DataFrame
    import pandas as pd
    sample_elev_df = pd.DataFrame({
        'elev': [100, 105, 98, 102, 99, 103, 101]
    }, index=[(100, 200), (101, 200), (99, 200), (100, 201), (100, 199), (101, 201), (99, 199)])
    
    # Calculate slope using FDA method
    slope_fda, aspect_fda = slope_FDA(coords, sample_elev_df, resolution, cell_spacing)
    print(f"Slope (FDA): {slope_fda:.2f}°, Aspect: {aspect_fda:.2f}°")
    
    # Calculate curvature
    curv_value = curvature(coords, sample_elev_df, resolution, cell_spacing)
    print(f"Curvature: {curv_value:.4f}")
    
    # Calculate TRI and TPI
    tri_value = TRI_calcu(coords, resolution, sample_elev_df)
    tpi_value = TPI_calcu(coords, resolution, sample_elev_df)
    print(f"TRI: {tri_value:.4f}")
    print(f"TPI: {tpi_value:.4f}")