"""
Hydrological Quantization Script for DGGS Flood Modeling

This script performs hydrological quantization for Discrete Global Grid System (DGGS)
by calculating flow direction, flow accumulation, and hydrological indices.

The script implements depression filling, flow direction calculation, and
hydrological parameter computation for flood modeling applications.

Author: Erin Li
"""

import warnings
import sys
import time
import functools
import os
import gc
import logging
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import pandas as pd
import geopandas as gpd
import numpy as np
import multiprocess as mp

# Import local modules
import DgBaseFunc as dbfc
import DgTopogFunc as dtfc
import DgHydroFunc as dhfc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.simplefilter('error', RuntimeWarning)


class HydrologicalQuantizer:
    """
    A class to handle hydrological quantization for DGGS flood modeling.
    
    This class provides methods to calculate flow direction, flow accumulation,
    and hydrological indices using depression filling and flow routing algorithms.
    """
    
    def __init__(self, resolution: int, sub_basin_id: int, data_dir: str = "Data", 
                 result_dir: str = "Result"):
        """
        Initialize the hydrological quantizer.
        
        Args:
            resolution: DGGS resolution level
            sub_basin_id: Sub-basin ID
            data_dir: Directory containing input data files
            result_dir: Directory for output results
        """
        self.resolution = resolution
        self.sub_basin_id = sub_basin_id
        self.data_dir = Path(data_dir)
        self.result_dir = Path(result_dir)
        
        # Validate inputs
        if not isinstance(resolution, int) or resolution < 16 or resolution > 29:
            raise ValueError("Resolution must be an integer between 16 and 29")
            
        if not isinstance(sub_basin_id, int) or sub_basin_id < 1:
            raise ValueError("Sub-basin ID must be a positive integer")
            
        # Create output directory if it doesn't exist
        self.result_dir.mkdir(exist_ok=True)
        
        # Load lookup table for cell properties
        self.lookup_table = dbfc.look_up_table()
        self.cell_spacing = self.lookup_table.loc[resolution, 'cell_spacing'] * 1000
        self.vertical_res = self.lookup_table.loc[resolution, 'verti_res']
        self.cell_area = self.lookup_table.loc[resolution, 'cell_area'] * 1000000
        
        logger.info(f"Initialized HydrologicalQuantizer for resolution {resolution}, sub-basin {sub_basin_id}")
        
    def load_elevation_data(self) -> pd.DataFrame:
        """
        Load and merge elevation data with centroid information.
        
        Returns:
            DataFrame containing elevation data with spatial information
            
        Raises:
            FileNotFoundError: If required files don't exist
        """
        try:
            # Load centroid data
            centroid_path = self.result_dir / f"NB_area_wgs84_centroids_{self.resolution}.csv"
            if not centroid_path.exists():
                raise FileNotFoundError(f"Centroid file not found: {centroid_path}")
                
            centroid_df = pd.read_csv(centroid_path, usecols=['i', 'j', 'lon_c', 'lat_c'])
            
            # Load elevation data
            elev_path = self.result_dir / f"NB_elev_{self.resolution}.csv"
            if not elev_path.exists():
                raise FileNotFoundError(f"Elevation file not found: {elev_path}")
                
            elev_df = pd.read_csv(elev_path, sep=',')
            
            # Merge dataframes
            merge_df = pd.merge(left=elev_df, right=centroid_df, how="inner", on=['i', 'j'])
            
            logger.info(f"Loaded elevation data with {len(merge_df)} records")
            return merge_df
            
        except Exception as e:
            logger.error(f"Failed to load elevation data: {e}")
            raise
            
    def filter_by_sub_basin(self, merge_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter elevation data to include only cells within the specified sub-basin.
        
        Args:
            merge_df: DataFrame containing elevation and spatial data
            
        Returns:
            Filtered DataFrame
            
        Raises:
            FileNotFoundError: If sub-basin shapefile doesn't exist
        """
        try:
            # Load sub-basin extent
            extent_path = self.data_dir / "NB_area_subbasin_wgs84.shp"
            if not extent_path.exists():
                raise FileNotFoundError(f"Sub-basin shapefile not found: {extent_path}")
                
            extent = gpd.GeoDataFrame.from_file(extent_path)
            
            # Create GeoDataFrame and filter
            merge_gdf = gpd.GeoDataFrame(
                merge_df, 
                geometry=gpd.points_from_xy(merge_df.lon_c, merge_df.lat_c)
            )
            
            # Filter by sub-basin (adjust index for 0-based indexing)
            filtered_gdf = merge_gdf[merge_gdf.geometry.within(extent.geometry[self.sub_basin_id - 1])]
            
            # Convert back to DataFrame
            elev_df = pd.DataFrame(filtered_gdf)
            elev_df = elev_df.drop(columns=['lon_c', 'lat_c', 'geometry'])
            elev_df = elev_df.set_index(['i', 'j'])
            elev_df = elev_df.sort_index()
            
            logger.info(f"Filtered to {len(elev_df)} cells in sub-basin {self.sub_basin_id}")
            return elev_df
            
        except Exception as e:
            logger.error(f"Failed to filter by sub-basin: {e}")
            raise
            
    def initialize_edge_cells(self, elev_df: pd.DataFrame, n_cores: int = 1) -> Tuple[List, List]:
        """
        Initialize edge cells and create processing queues.
        
        Args:
            elev_df: DataFrame containing elevation data
            n_cores: Number of cores for parallel processing
            
        Returns:
            Tuple of (finished_queue, unfinished_queue)
        """
        try:
            logger.info("Initializing edge cells")
            start_time = time.time()
            
            # Initialize queues
            Q = elev_df.index.values.tolist()  # All cell indices
            
            # Process edge cells in parallel
            edge_cell_df_p = functools.partial(dbfc.edge_cell_df, res=self.resolution, allcell=Q)
            
            if n_cores > 1:
                elev_df_split = np.array_split(elev_df, n_cores)
                with mp.Pool(processes=n_cores) as pool:
                    elev_df_temp = pd.concat(pool.map(edge_cell_df_p, elev_df_split))
            else:
                elev_df_temp = edge_cell_df_p(elev_df)
            
            # Create finished and unfinished queues
            P = elev_df_temp.index.values.tolist()  # Finished queue
            U = [i for i in Q if (i not in P)]  # Unfinished queue
            A = []  # Plain queue
            
            processing_time = time.time() - start_time
            logger.info(f"Edge cell initialization completed in {processing_time:.2f} seconds")
            logger.info(f"Finished queue: {len(P)} cells, Unfinished queue: {len(U)} cells")
            
            return P, U, A
            
        except Exception as e:
            logger.error(f"Failed to initialize edge cells: {e}")
            raise
            
    def fill_depressions(self, elev_df: pd.DataFrame, P: List, U: List, A: List) -> pd.DataFrame:
        """
        Fill depressions using ascending elevation order.
        
        Args:
            elev_df: DataFrame containing elevation data
            P: Finished queue
            U: Unfinished queue
            A: Plain queue
            
        Returns:
            DataFrame with filled depressions
        """
        try:
            logger.info("Starting depression filling")
            start_time = time.time()
            
            P_df = elev_df[elev_df.index.isin(P)]
            
            while len(U) > 0:
                # Sort by elevation and process lowest cell
                P_df = P_df.sort_values(by=['model_elev'])
                ij = P_df[:1].index.values[0]
                elev = P_df[:1].model_elev.values[0]
                
                # Get neighbors
                C = dbfc.neighbor_coords(self.resolution, ij)[1:]
                C = [i for i in C if (i not in P) and (i in U)]
                
                # Process neighbors
                for neighbor in C:
                    if neighbor in U:
                        # Fill depression
                        elev_df.at[neighbor, 'model_elev'] = elev
                        
                        # Move from U to P
                        U.remove(neighbor)
                        P.append(neighbor)
                        P_df = elev_df[elev_df.index.isin(P)]
                        
                        # Update A if needed
                        if neighbor not in A:
                            A.append(neighbor)
                
                # Remove processed cell from P_df
                P_df = P_df.drop(ij)
                
                # Move processed cell to A if not already there
                if ij not in A:
                    A.append(ij)
                    
            processing_time = time.time() - start_time
            logger.info(f"Depression filling completed in {processing_time:.2f} seconds")
            
            return elev_df
            
        except Exception as e:
            logger.error(f"Failed to fill depressions: {e}")
            raise
            
    def calculate_flow_direction(self, elev_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate flow direction for all cells.
        
        Args:
            elev_df: DataFrame containing elevation data
            
        Returns:
            DataFrame with flow direction information
        """
        try:
            logger.info("Calculating flow direction")
            
            # Calculate slope and aspect
            elev_df = dtfc.slope_aspect_df(elev_df, elev_df, 'FDA', self.resolution, self.cell_spacing)
            
            # Calculate flow direction
            elev_df = dhfc.flow_direction_restricted_df(elev_df, self.resolution)
            
            logger.info("Flow direction calculation completed")
            return elev_df
            
        except Exception as e:
            logger.error(f"Failed to calculate flow direction: {e}")
            raise
            
    def calculate_flow_accumulation(self, elev_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate flow accumulation for all cells.
        
        Args:
            elev_df: DataFrame containing elevation and flow direction data
            
        Returns:
            DataFrame with flow accumulation information
        """
        try:
            logger.info("Calculating flow accumulation")
            
            # Initialize upslope area
            elev_df['upslope'] = 1
            
            # Calculate inflow count
            elev_df = dhfc.inflow_count_restricted_df(elev_df, self.resolution, elev_df)
            
            # Process flow accumulation
            while elev_df['inflow'].min() == 0:
                # Find cells with zero inflow
                zero_inflow = elev_df[elev_df['inflow'] == 0].index.tolist()
                
                for coords in zero_inflow:
                    if coords in elev_df.index:
                        dhfc.upslope_restricted(coords, self.resolution, elev_df)
            
            # Calculate contributing area
            elev_df['contri_area'] = elev_df['upslope'] * self.cell_area
            
            logger.info("Flow accumulation calculation completed")
            return elev_df
            
        except Exception as e:
            logger.error(f"Failed to calculate flow accumulation: {e}")
            raise
            
    def calculate_hydrological_indices(self, elev_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate hydrological indices (TWI, SPI).
        
        Args:
            elev_df: DataFrame containing elevation, slope, and flow accumulation data
            
        Returns:
            DataFrame with hydrological indices
        """
        try:
            logger.info("Calculating hydrological indices")
            
            # Calculate TWI and SPI
            elev_df = dhfc.TWI_df(elev_df)
            elev_df = dhfc.SPI_df(elev_df)
            
            logger.info("Hydrological indices calculation completed")
            return elev_df
            
        except Exception as e:
            logger.error(f"Failed to calculate hydrological indices: {e}")
            raise
            
    def save_results(self, elev_df: pd.DataFrame) -> None:
        """
        Save hydrological quantization results to CSV file.
        
        Args:
            elev_df: DataFrame containing results
        """
        try:
            output_path = self.result_dir / f"NB_hydro_{self.resolution}_{self.sub_basin_id}.csv"
            
            logger.info(f"Saving results to {output_path}")
            elev_df.to_csv(output_path)
            
            logger.info(f"Saved {len(elev_df)} hydrological records")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            raise
            
    def run_quantization(self, n_cores: int = 1) -> None:
        """
        Run the complete hydrological quantization process.
        
        Args:
            n_cores: Number of cores for parallel processing
        """
        start_time = time.time()
        
        try:
            logger.info(f"Starting hydrological quantization for resolution {self.resolution}, sub-basin {self.sub_basin_id}")
            
            # Load and prepare data
            merge_df = self.load_elevation_data()
            elev_df = self.filter_by_sub_basin(merge_df)
            
            # Initialize edge cells
            P, U, A = self.initialize_edge_cells(elev_df, n_cores)
            
            # Fill depressions
            elev_df = self.fill_depressions(elev_df, P, U, A)
            
            # Calculate flow direction
            elev_df = self.calculate_flow_direction(elev_df)
            
            # Calculate flow accumulation
            elev_df = self.calculate_flow_accumulation(elev_df)
            
            # Calculate hydrological indices
            elev_df = self.calculate_hydrological_indices(elev_df)
            
            # Save results
            self.save_results(elev_df)
            
            processing_time = time.time() - start_time
            logger.info(f"Hydrological quantization completed in {processing_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Hydrological quantization failed: {e}")
            raise


def main():
    """
    Main function to run hydrological quantization.
    
    Command line usage:
        python Quantization_hydro.py <resolution> <sub_basin_id> [--n_cores <cores>]
    """
    parser = argparse.ArgumentParser(
        description="Hydrological quantization for DGGS flood modeling"
    )
    parser.add_argument(
        "resolution", 
        type=int, 
        help="DGGS resolution level (16-29)"
    )
    parser.add_argument(
        "sub_basin_id", 
        type=int, 
        help="Sub-basin ID"
    )
    parser.add_argument(
        "--n_cores", 
        type=int, 
        default=1,
        help="Number of cores for parallel processing"
    )
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="Data",
        help="Directory containing input data files"
    )
    parser.add_argument(
        "--result_dir", 
        type=str, 
        default="Result",
        help="Directory for output results"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize quantizer
        quantizer = HydrologicalQuantizer(
            resolution=args.resolution,
            sub_basin_id=args.sub_basin_id,
            data_dir=args.data_dir,
            result_dir=args.result_dir
        )
        
        # Run quantization
        quantizer.run_quantization(n_cores=args.n_cores)
        
    except Exception as e:
        logger.error(f"Hydrological quantization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()


