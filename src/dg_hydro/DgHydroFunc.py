"""
DGGS Hydrological Functions for Flood Modeling

This module provides hydrological analysis functions for Discrete Global Grid System (DGGS)
operations, including flow direction calculation, flow accumulation, topographic wetness
index (TWI), stream power index (SPI), and hydrological parameter computation.

The module implements hydrological algorithms adapted for hexagonal DGGS cells,
including pit detection, flow direction assignment, and accumulation calculations.

Author: Erin Li
"""

import numpy as np
import pandas as pd
import math
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


def flow_direction_restricted(asp: float, res: int) -> int:
    """
    Calculate restricted flow direction based on aspect and resolution.
    
    Determines the flow direction code (1-6) for hexagonal DGGS cells based on
    the aspect angle and resolution. The direction is restricted to 6 discrete
    directions corresponding to the 6 neighbors of a hexagonal cell.
    
    Args:
        asp: Aspect angle in degrees (0-360)
        res: DGGS resolution level
        
    Returns:
        Direction code (1-6) representing the flow direction
        
    Raises:
        ValueError: If aspect angle is outside valid range
    """
    try:
        if not isinstance(asp, (int, float)) or not isinstance(res, int):
            raise ValueError("Aspect must be numeric and resolution must be integer")
            
        if res < 1:
            raise ValueError("Resolution must be positive integer")
            
        if res % 2 == 1:
            if 0 <= asp < 360:
                direc_code = asp // 60 + 1
            else:
                direc_code = asp // 60
        elif res % 2 == 0:
            if 0 <= asp < 330:
                direc_code = (asp + 30) // 60 + 1
            else:
                direc_code = (asp + 30) // 60
        else:
            raise ValueError("Invalid resolution value")
            
        return int(direc_code)
        
    except Exception as e:
        logger.error(f"Flow direction calculation failed: {e}")
        raise RuntimeError(f"Failed to calculate flow direction: {e}")


def flow_direction_restricted_df(dataframe: pd.DataFrame, res: int) -> pd.DataFrame:
    """
    Calculate flow direction for all cells in a DataFrame.
    
    Processes aspect values and assigns flow direction codes to each cell.
    Handles NaN values and edge cases appropriately.
    
    Args:
        dataframe: DataFrame containing 'asp' column with aspect values
        res: DGGS resolution level
        
    Returns:
        DataFrame with added 'direction_code' column
        
    Raises:
        ValueError: If required columns are missing
        RuntimeError: If processing fails
    """
    try:
        if 'asp' not in dataframe.columns:
            raise ValueError("DataFrame must contain 'asp' column")
            
        # Create a copy to avoid modifying original
        df = dataframe.copy()
        
        # Handle NaN values in aspect column
        df['asp'] = [i if i != NO_DATA_VALUE else np.nan for i in df['asp']]
        df['direction_code'] = np.nan
        
        # Calculate flow direction for each cell
        df['direction_code'] = [
            flow_direction_restricted(i, res) if not np.isnan(i) else np.nan 
            for i in df['asp']
        ]
        
        return df
        
    except Exception as e:
        logger.error(f"Flow direction DataFrame processing failed: {e}")
        raise RuntimeError(f"Failed to process flow direction DataFrame: {e}")


def pits_and_flat_df(dataframe: pd.DataFrame, res: int, elev_df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify pits and flat areas in the elevation data.
    
    Determines cells where center elevation is lower than or equal to all neighbors,
    indicating pits or flat areas. Returns a DataFrame with ascending elevations.
    
    Args:
        dataframe: DataFrame containing cell coordinates
        res: DGGS resolution level
        elev_df: DataFrame containing elevation data
        
    Returns:
        DataFrame with pit/flat cell information (i, j, elev columns)
        
    Raises:
        ValueError: If input data is invalid
        RuntimeError: If processing fails
    """
    try:
        if dataframe.empty or elev_df.empty:
            raise ValueError("Input DataFrames cannot be empty")
            
        pending_i = []
        pending_j = []
        pending_elev = []
        
        for coords in dataframe.index.values:
            i, j = coords[0], coords[1]
            elev_neighbor = dbfc.neighbor_navig(res, coords, elev_df)
            
            if dbfc.edge_cell_exist(elev_neighbor):
                continue
            else:
                neighbor = elev_neighbor[1:]
                if all(n >= elev_neighbor[0] for n in neighbor):
                    pending_i.append(i)
                    pending_j.append(j)
                    pending_elev.append(elev_neighbor[0])
        
        # Create DataFrame with pit/flat information
        pending_df = pd.DataFrame(
            list(zip(pending_i, pending_j, pending_elev)), 
            columns=['i', 'j', 'elev']
        )
        pending_df = pending_df.set_index(['i', 'j'])
        
        return pending_df
        
    except Exception as e:
        logger.error(f"Pit and flat area detection failed: {e}")
        raise RuntimeError(f"Failed to detect pits and flat areas: {e}")


def flow_to_cell(coords: Tuple[int, int], res: int, elev_df: pd.DataFrame) -> Optional[Tuple[int, int]]:
    """
    Determine the coordinates of the cell that a target cell will flow into.
    
    Args:
        coords: Target cell coordinates (i, j)
        res: DGGS resolution level
        elev_df: DataFrame containing elevation and direction data
        
    Returns:
        Coordinates of the cell that receives flow, or None if no valid flow
        
    Raises:
        ValueError: If coordinates are invalid
        RuntimeError: If flow calculation fails
    """
    try:
        if not isinstance(coords, tuple) or len(coords) != 2:
            raise ValueError("Coordinates must be a tuple of (i, j)")
            
        direc = elev_df['direction_code'].loc[coords]
        
        if not np.isnan(direc):
            if direc == -1:
                return -1
            else:
                direct = int(direc)
                neighbor_flowto_coords = dbfc.neighbor_coords(res, coords)[direct]
                
                if neighbor_flowto_coords in list(elev_df.index.values):
                    return neighbor_flowto_coords
                    
        return None
        
    except KeyError:
        logger.warning(f"Coordinates {coords} not found in elevation DataFrame")
        return None
    except Exception as e:
        logger.error(f"Flow to cell calculation failed: {e}")
        raise RuntimeError(f"Failed to calculate flow to cell: {e}")


def upslope_restricted(coords: Tuple[int, int], res: int, elev_df: pd.DataFrame) -> None:
    """
    Update upslope area calculations for flow accumulation.
    
    Modifies the elevation DataFrame in-place to update upslope area calculations
    based on flow direction and current upslope values.
    
    Args:
        coords: Cell coordinates (i, j)
        res: DGGS resolution level
        elev_df: DataFrame containing elevation, direction, and upslope data
        
    Raises:
        ValueError: If coordinates are invalid
        RuntimeError: If upslope calculation fails
    """
    try:
        if not isinstance(coords, tuple) or len(coords) != 2:
            raise ValueError("Coordinates must be a tuple of (i, j)")
            
        direct = int(elev_df['direction_code'].loc[coords])
        upslope = int(elev_df['upslope'].loc[coords])
        neighbor_flowto = dbfc.neighbor_coords(res, coords)
        
        try:
            elev_df.at[neighbor_flowto[direct], 'upslope'] += upslope
            
            if elev_df.at[neighbor_flowto[direct], 'inflow'] < 0:
                if not np.isnan(elev_df.at[neighbor_flowto[direct], 'direction_code']):
                    elev_df.at[neighbor_flowto[direct], 'inflow'] = 0
            else:
                elev_df.at[neighbor_flowto[direct], 'inflow'] -= 1
                
        except KeyError:
            logger.warning(f"Neighbor coordinates not found in elevation DataFrame")
            
        elev_df.at[coords, 'inflow'] -= 1
        
    except Exception as e:
        logger.error(f"Upslope calculation failed: {e}")
        raise RuntimeError(f"Failed to calculate upslope: {e}")


def check_closed_loop(coords: Tuple[int, int], res: int, elev_df: pd.DataFrame) -> Tuple[Optional[Tuple[int, int]], Optional[List[Tuple[int, int]]]]:
    """
    Check if there are cells forming a closed loop in flow direction.
    
    Traces the flow path from a starting cell to detect closed loops.
    Stops when encountering an edge cell or pit/flat area.
    
    Args:
        coords: Starting cell coordinates (i, j)
        res: DGGS resolution level
        elev_df: DataFrame containing elevation and direction data
        
    Returns:
        Tuple of (loop_start_coords, flow_path) or (None, None) if no loop
        
    Raises:
        ValueError: If coordinates are invalid
        RuntimeError: If loop detection fails
    """
    try:
        if not isinstance(coords, tuple) or len(coords) != 2:
            raise ValueError("Coordinates must be a tuple of (i, j)")
            
        flow_path = [coords]
        flowto_coords = flow_to_cell(coords, res, elev_df)
        
        while True:
            if flowto_coords is None or flowto_coords == -1:
                return None, None
            elif flowto_coords in flow_path:
                return flowto_coords, flow_path
            else:
                flow_path.append(flowto_coords)
                flowto_coords = flow_to_cell(flowto_coords, res, elev_df)
                
    except Exception as e:
        logger.error(f"Closed loop detection failed: {e}")
        raise RuntimeError(f"Failed to detect closed loop: {e}")


def check_closed_loop_df(dataframe: pd.DataFrame, res: int, elev_df: pd.DataFrame) -> pd.DataFrame:
    """
    Check for closed loops in flow direction across entire DataFrame.
    
    Args:
        dataframe: DataFrame containing cell coordinates
        res: DGGS resolution level
        elev_df: DataFrame containing elevation and direction data
        
    Returns:
        DataFrame with detected closed loop information
        
    Raises:
        ValueError: If input data is invalid
        RuntimeError: If processing fails
    """
    try:
        if dataframe.empty:
            raise ValueError("Input DataFrame cannot be empty")
            
        close_loop_df = pd.DataFrame(columns=['flow_path'])
        
        for coords in dataframe.index.values:
            flowto_coords, flow_path = check_closed_loop(coords, res, elev_df)
            
            if flowto_coords is not None:
                cell_index = flow_path.index(flowto_coords)
                flow_path = flow_path[cell_index:]
                flow_elev = [elev_df['model_elev'].loc[coord] for coord in flow_path]
                flow_dict = dict(zip(flow_path, flow_elev))
                flow_dict = dict(sorted(flow_dict.items(), key=lambda item: (item[1], item[0])))
                
                close_loop_df = close_loop_df.append({'flow_path': flow_dict}, ignore_index=True)
        
        return close_loop_df
        
    except Exception as e:
        logger.error(f"Closed loop DataFrame processing failed: {e}")
        raise RuntimeError(f"Failed to process closed loop DataFrame: {e}")


def inflow_count_restricted(coords: Tuple[int, int], res: int, elev_df: pd.DataFrame) -> int:
    """
    Count inflow cells for a given cell.
    
    Counts the number of neighboring cells that flow into the target cell.
    Range is 0-6, representing none to all neighbors flowing in.
    
    Args:
        coords: Target cell coordinates (i, j)
        res: DGGS resolution level
        elev_df: DataFrame containing elevation and direction data
        
    Returns:
        Number of inflow cells (0-6)
        
    Raises:
        ValueError: If coordinates are invalid
        RuntimeError: If inflow calculation fails
    """
    try:
        if not isinstance(coords, tuple) or len(coords) != 2:
            raise ValueError("Coordinates must be a tuple of (i, j)")
            
        neighbor_direc = dbfc.neighbor_direc_navig(res, coords, elev_df)
        count = 0
        direction_ls = [4, 5, 6, 1, 2, 3]
        
        for i in range(1, 7):
            if neighbor_direc[i] == direction_ls[i-1]:
                count += 1
                
        return count
        
    except Exception as e:
        logger.error(f"Inflow count calculation failed: {e}")
        raise RuntimeError(f"Failed to calculate inflow count: {e}")


def inflow_count_restricted_df(dataframe: pd.DataFrame, res: int, elev_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate inflow cell counts for all cells in a DataFrame.
    
    Args:
        dataframe: DataFrame containing cell coordinates
        res: DGGS resolution level
        elev_df: DataFrame containing elevation and direction data
        
    Returns:
        DataFrame with added 'inflow' column
        
    Raises:
        ValueError: If input data is invalid
        RuntimeError: If processing fails
    """
    try:
        if dataframe.empty:
            raise ValueError("Input DataFrame cannot be empty")
            
        df = dataframe.copy()
        df['inflow'] = [inflow_count_restricted(ij, res, elev_df) for ij in df.index.values]
        
        return df
        
    except Exception as e:
        logger.error(f"Inflow count DataFrame processing failed: {e}")
        raise RuntimeError(f"Failed to process inflow count DataFrame: {e}")


def TWI_calcu(alpha: float, beta: float) -> float:
    """
    Calculate Topographic Wetness Index (TWI).
    
    TWI = ln(α/tan(β)) where α is the upslope contributing area and β is the slope angle.
    
    Args:
        alpha: Upslope contributing area
        beta: Slope angle in degrees
        
    Returns:
        TWI value or NaN if calculation not possible
        
    Raises:
        ValueError: If inputs are invalid
        RuntimeError: If calculation fails
    """
    try:
        if not isinstance(alpha, (int, float)) or not isinstance(beta, (int, float)):
            raise ValueError("Alpha and beta must be numeric values")
            
        if beta == 0:
            return np.nan
        else:
            beta_rad = math.radians(beta)
            TWI_value = math.log(alpha / math.tan(beta_rad))
            return TWI_value
            
    except Exception as e:
        logger.error(f"TWI calculation failed: {e}")
        return np.nan


def SPI_calcu(alpha: float, beta: float) -> float:
    """
    Calculate Stream Power Index (SPI).
    
    SPI = α * tan(β) where α is the upslope contributing area and β is the slope angle.
    
    Args:
        alpha: Upslope contributing area
        beta: Slope angle in degrees
        
    Returns:
        SPI value
        
    Raises:
        ValueError: If inputs are invalid
        RuntimeError: If calculation fails
    """
    try:
        if not isinstance(alpha, (int, float)) or not isinstance(beta, (int, float)):
            raise ValueError("Alpha and beta must be numeric values")
            
        beta_rad = math.radians(beta)
        SPI_value = alpha * math.tan(beta_rad)
        return SPI_value
        
    except Exception as e:
        logger.error(f"SPI calculation failed: {e}")
        raise RuntimeError(f"Failed to calculate SPI: {e}")


def TWI_df(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Topographic Wetness Index for all cells in a DataFrame.
    
    Args:
        dataframe: DataFrame containing 'contri_area' and 'slp' columns
        
    Returns:
        DataFrame with added 'twi' column
        
    Raises:
        ValueError: If required columns are missing
        RuntimeError: If processing fails
    """
    try:
        if 'contri_area' not in dataframe.columns or 'slp' not in dataframe.columns:
            raise ValueError("DataFrame must contain 'contri_area' and 'slp' columns")
            
        df = dataframe.copy()
        df['twi'] = np.nan
        df['twi'] = [TWI_calcu(a, b) for a, b in zip(df['contri_area'], df['slp'])]
        
        return df
        
    except Exception as e:
        logger.error(f"TWI DataFrame processing failed: {e}")
        raise RuntimeError(f"Failed to process TWI DataFrame: {e}")


def SPI_df(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Stream Power Index for all cells in a DataFrame.
    
    Args:
        dataframe: DataFrame containing 'contri_area' and 'slp' columns
        
    Returns:
        DataFrame with added 'spi' column
        
    Raises:
        ValueError: If required columns are missing
        RuntimeError: If processing fails
    """
    try:
        if 'contri_area' not in dataframe.columns or 'slp' not in dataframe.columns:
            raise ValueError("DataFrame must contain 'contri_area' and 'slp' columns")
            
        df = dataframe.copy()
        df['spi'] = np.nan
        df['spi'] = [SPI_calcu(a, b) for a, b in zip(df['contri_area'], df['slp'])]
        
        return df
        
    except Exception as e:
        logger.error(f"SPI DataFrame processing failed: {e}")
        raise RuntimeError(f"Failed to process SPI DataFrame: {e}")


if __name__ == '__main__':
    """
    Example usage of hydrological functions.
    
    This section demonstrates how to use the key functions in this module.
    """
    # Example: Calculate flow direction for a single cell
    aspect_angle = 45.0
    resolution = 23
    direction = flow_direction_restricted(aspect_angle, resolution)
    print(f"Flow direction for aspect {aspect_angle}° at resolution {resolution}: {direction}")
    
    # Example: Calculate TWI and SPI
    alpha = 1000.0  # Contributing area
    beta = 15.0     # Slope angle in degrees
    
    twi_value = TWI_calcu(alpha, beta)
    spi_value = SPI_calcu(alpha, beta)
    
    print(f"TWI: {twi_value:.4f}")
    print(f"SPI: {spi_value:.4f}")