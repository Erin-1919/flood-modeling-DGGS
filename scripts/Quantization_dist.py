"""
Distance Quantization Script for DGGS Flood Modeling

This script performs distance quantization for Discrete Global Grid System (DGGS)
by calculating hexagonal distances to the nearest hydrological network (NHN).

The script extracts NHN values from raster data, identifies cells with NHN presence,
and calculates hexagonal distances from all cells to the nearest NHN features.

Author: Erin Li
"""

import warnings
import sys
import time
import os
import functools
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
import GeoBaseFunc as gbfc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.simplefilter('error', RuntimeWarning)


class DistanceQuantizer:
    """
    A class to handle distance quantization for DGGS flood modeling.
    
    This class provides methods to calculate hexagonal distances to the nearest
    hydrological network (NHN) features for flood modeling applications.
    """
    
    def __init__(self, resolution: int, data_dir: str = "Data", result_dir: str = "Result"):
        """
        Initialize the distance quantizer.
        
        Args:
            resolution: DGGS resolution level
            data_dir: Directory containing input data files
            result_dir: Directory for output results
        """
        self.resolution = resolution
        self.data_dir = Path(data_dir)
        self.result_dir = Path(result_dir)
        
        # Validate resolution
        if not isinstance(resolution, int) or resolution < 16 or resolution > 29:
            raise ValueError("Resolution must be an integer between 16 and 29")
            
        # Create output directory if it doesn't exist
        self.result_dir.mkdir(exist_ok=True)
        
        # Load lookup table for cell properties
        self.lookup_table = dbfc.look_up_table()
        self.cell_spacing = self.lookup_table.loc[resolution, 'cell_spacing'] * 1000
        self.vertical_res = self.lookup_table.loc[resolution, 'verti_res']
        
        logger.info(f"Initialized DistanceQuantizer for resolution {resolution}")
        
    def load_centroid_data(self) -> gpd.GeoDataFrame:
        """
        Load centroid data from CSV file.
        
        Returns:
            GeoDataFrame containing centroid data with geometry
            
        Raises:
            FileNotFoundError: If centroid file doesn't exist
        """
        try:
            input_csv_path = self.result_dir / f"NB_area_wgs84_centroids_{self.resolution}.csv"
            
            if not input_csv_path.exists():
                raise FileNotFoundError(f"Centroid file not found: {input_csv_path}")
                
            logger.info(f"Loading centroid data from {input_csv_path}")
            
            centroid_df = pd.read_csv(input_csv_path, sep=',')
            centroid_gdf = gpd.GeoDataFrame(
                centroid_df, 
                geometry=gpd.points_from_xy(centroid_df.lon_c, centroid_df.lat_c)
            )
            centroid_gdf = centroid_gdf.set_index(['i', 'j'])
            
            logger.info(f"Loaded {len(centroid_gdf)} centroids")
            return centroid_gdf
            
        except Exception as e:
            logger.error(f"Failed to load centroid data: {e}")
            raise
            
    def load_sample_data(self) -> pd.DataFrame:
        """
        Load sample data from CSV file.
        
        Returns:
            DataFrame containing sample data
            
        Raises:
            FileNotFoundError: If sample file doesn't exist
        """
        try:
            input_csv_path = self.data_dir / f"NB_sample_cent_quanti2_{self.resolution}.csv"
            
            if not input_csv_path.exists():
                raise FileNotFoundError(f"Sample data file not found: {input_csv_path}")
                
            logger.info(f"Loading sample data from {input_csv_path}")
            
            sdf = pd.read_csv(input_csv_path, sep=',')
            sdf = sdf.set_index(['i', 'j'])
            sdf = sdf.sort_index()
            
            logger.info(f"Loaded {len(sdf)} sample records")
            return sdf
            
        except Exception as e:
            logger.error(f"Failed to load sample data: {e}")
            raise
            
    def filter_by_study_area(self, centroid_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Filter centroids to only include those within the study area.
        
        Args:
            centroid_gdf: GeoDataFrame containing centroid data
            
        Returns:
            Filtered GeoDataFrame
            
        Raises:
            FileNotFoundError: If study area shapefile doesn't exist
        """
        try:
            extent_path = self.data_dir / "NB_area_wgs84.shp"
            
            if not extent_path.exists():
                raise FileNotFoundError(f"Study area shapefile not found: {extent_path}")
                
            logger.info("Loading study area extent")
            extent = gpd.GeoDataFrame.from_file(extent_path)
            
            # Filter centroids within study area
            filtered_gdf = centroid_gdf[centroid_gdf.geometry.within(extent.geometry[0])]
            
            logger.info(f"Filtered to {len(filtered_gdf)} centroids within study area")
            return filtered_gdf
            
        except Exception as e:
            logger.error(f"Failed to filter by study area: {e}")
            raise
            
    def extract_nhn_values(self, centroid_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Extract NHN (Natural Hydrological Network) values from raster data.
        
        Args:
            centroid_gdf: GeoDataFrame containing centroid data
            
        Returns:
            GeoDataFrame with NHN values
            
        Raises:
            FileNotFoundError: If NHN raster files don't exist
        """
        try:
            logger.info("Extracting NHN values from raster data")
            
            # Extract values by nearest interpolation - rasterized NHN
            nhn1_path = self.data_dir / "out_nlflow_1_wgs84.tif"
            nhn2_path = self.data_dir / "out_waterbody_2_wgs84.tif"
            
            if not nhn1_path.exists():
                raise FileNotFoundError(f"NHN flow raster not found: {nhn1_path}")
            if not nhn2_path.exists():
                raise FileNotFoundError(f"NHN waterbody raster not found: {nhn2_path}")
                
            centroid_gdf = gbfc.nearest_interp_df(
                centroid_gdf, str(nhn1_path), 'nhn1', 'lon_c', 'lat_c'
            )
            centroid_gdf = gbfc.nearest_interp_df(
                centroid_gdf, str(nhn2_path), 'nhn2', 'lon_c', 'lat_c'
            )
            
            # Populate column value indicating existence of NHN
            centroid_gdf['nhn3'] = np.where(
                (centroid_gdf['nhn1'] != -32767) | (centroid_gdf['nhn2'] != -32767), 
                1, 0
            )
            
            # Filter to only cells with NHN presence
            centroid_gdf = centroid_gdf[centroid_gdf['nhn3'] == 1]
            
            logger.info(f"Extracted NHN values for {len(centroid_gdf)} cells")
            return centroid_gdf
            
        except Exception as e:
            logger.error(f"Failed to extract NHN values: {e}")
            raise
            
    def calculate_hex_distance(self, sdf: pd.DataFrame, coords_target: List, 
                             n_cores: int = 1) -> pd.DataFrame:
        """
        Calculate hexagonal distance to the closest NHN for each cell.
        
        Args:
            sdf: DataFrame containing sample data
            coords_target: List of target coordinates with NHN presence
            n_cores: Number of cores for parallel processing
            
        Returns:
            DataFrame with hexagonal distance values
        """
        try:
            logger.info("Calculating hexagonal distances to NHN")
            
            # Create partial function for parallel processing
            hex_dist_comb_df_p = functools.partial(
                dbfc.hex_dist_comb_df,
                coords_target=coords_target,
                var='nhn',
                res=self.resolution
            )
            
            if n_cores > 1:
                # Parallel processing
                logger.info(f"Using parallel processing with {n_cores} cores")
                sdf_split = np.array_split(sdf, n_cores)
                
                with mp.Pool(processes=n_cores) as pool:
                    results = pool.map(hex_dist_comb_df_p, sdf_split)
                    sdf = pd.concat(results)
            else:
                # Sequential processing
                logger.info("Using sequential processing")
                sdf = hex_dist_comb_df_p(sdf)
            
            logger.info("Hexagonal distance calculation completed")
            return sdf
            
        except Exception as e:
            logger.error(f"Failed to calculate hexagonal distances: {e}")
            raise
            
    def save_results(self, sdf: pd.DataFrame) -> None:
        """
        Save distance quantization results to CSV file.
        
        Args:
            sdf: DataFrame containing results
        """
        try:
            output_path = self.data_dir / f"NB_sample_cent_quanti3_{self.resolution}.csv"
            
            logger.info(f"Saving results to {output_path}")
            sdf.to_csv(output_path, index=True)
            
            logger.info(f"Saved {len(sdf)} distance records")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            raise
            
    def run_quantization(self, n_cores: int = 1) -> None:
        """
        Run the complete distance quantization process.
        
        Args:
            n_cores: Number of cores for parallel processing
        """
        start_time = time.time()
        
        try:
            logger.info(f"Starting distance quantization for resolution {self.resolution}")
            
            # Load centroid data
            centroid_gdf = self.load_centroid_data()
            
            # Load sample data
            sdf = self.load_sample_data()
            
            # Filter by study area
            centroid_gdf = self.filter_by_study_area(centroid_gdf)
            
            # Extract NHN values
            centroid_gdf = self.extract_nhn_values(centroid_gdf)
            
            # Get target coordinates
            coords_target = centroid_gdf.index.values.tolist()
            
            # Calculate hexagonal distances
            sdf = self.calculate_hex_distance(sdf, coords_target, n_cores)
            
            # Save results
            self.save_results(sdf)
            
            processing_time = time.time() - start_time
            logger.info(f"Distance quantization completed in {processing_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Distance quantization failed: {e}")
            raise


def main():
    """
    Main function to run distance quantization.
    
    Command line usage:
        python Quantization_dist.py <resolution> [--n_cores <cores>]
    """
    parser = argparse.ArgumentParser(
        description="Distance quantization for DGGS flood modeling"
    )
    parser.add_argument(
        "resolution", 
        type=int, 
        help="DGGS resolution level (16-29)"
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
        quantizer = DistanceQuantizer(
            resolution=args.resolution,
            data_dir=args.data_dir,
            result_dir=args.result_dir
        )
        
        # Run quantization
        quantizer.run_quantization(n_cores=args.n_cores)
        
    except Exception as e:
        logger.error(f"Distance quantization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

