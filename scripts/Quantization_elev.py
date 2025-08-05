"""
Elevation Quantization Script for DGGS Flood Modeling

This script performs elevation quantization for Discrete Global Grid System (DGGS)
by interpolating elevation values from raster data to DGGS cell centroids.

The script reads centroid data, filters by study area extent, interpolates
elevation values using bilinear interpolation, and saves the results.

Author: Erin Li
"""

import functools
import pandas as pd
import geopandas as gpd
import multiprocess as mp
import warnings
import sys
import time
import logging
import argparse
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np

# Import local modules
import GeoBaseFunc as gbfc
import DgBaseFunc as dbfc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.simplefilter('error', RuntimeWarning)


class ElevationQuantizer:
    """
    A class to handle elevation quantization for DGGS flood modeling.
    
    This class provides methods to quantize elevation data from raster format
    to DGGS cell centroids using bilinear interpolation.
    """
    
    def __init__(self, resolution: int, data_dir: str = "Data", result_dir: str = "Result"):
        """
        Initialize the elevation quantizer.
        
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
        self.vertical_res = self.lookup_table.loc[resolution, 'verti_res']
        
        logger.info(f"Initialized ElevationQuantizer for resolution {resolution}")
        
    def load_centroid_data(self) -> gpd.GeoDataFrame:
        """
        Load centroid data from CSV file.
        
        Returns:
            GeoDataFrame containing centroid data with geometry
            
        Raises:
            FileNotFoundError: If centroid file doesn't exist
            ValueError: If data is invalid
        """
        try:
            input_csv_path = self.result_dir / f"NB_area_wgs84_centroids_{self.resolution}.csv"
            
            if not input_csv_path.exists():
                raise FileNotFoundError(f"Centroid file not found: {input_csv_path}")
                
            logger.info(f"Loading centroid data from {input_csv_path}")
            
            centroid_df = pd.read_csv(input_csv_path, sep=',')
            
            # Validate required columns
            required_cols = ['lon_c', 'lat_c']
            missing_cols = [col for col in required_cols if col not in centroid_df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
                
            # Create GeoDataFrame
            centroid_gdf = gpd.GeoDataFrame(
                centroid_df, 
                geometry=gpd.points_from_xy(centroid_df.lon_c, centroid_df.lat_c)
            )
            
            logger.info(f"Loaded {len(centroid_gdf)} centroids")
            return centroid_gdf
            
        except Exception as e:
            logger.error(f"Failed to load centroid data: {e}")
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
            
    def interpolate_elevation(self, centroid_gdf: gpd.GeoDataFrame, 
                            use_parallel: bool = False, n_cores: Optional[int] = None) -> gpd.GeoDataFrame:
        """
        Interpolate elevation values using bilinear interpolation.
        
        Args:
            centroid_gdf: GeoDataFrame containing centroid data
            use_parallel: Whether to use parallel processing
            n_cores: Number of cores for parallel processing
            
        Returns:
            GeoDataFrame with interpolated elevation values
        """
        try:
            dtm_path = self.data_dir / "NB_DTM_O_30_wgs84.tif"
            
            if not dtm_path.exists():
                raise FileNotFoundError(f"DTM file not found: {dtm_path}")
                
            logger.info("Starting elevation interpolation")
            
            if use_parallel and n_cores:
                # Parallel processing
                logger.info(f"Using parallel processing with {n_cores} cores")
                centroid_df_split = np.array_split(centroid_gdf, n_cores)
                
                with mp.Pool(processes=n_cores) as pool:
                    bilinear_interp_df_p = functools.partial(
                        gbfc.bilinear_interp_df,
                        tif=str(dtm_path),
                        var='model_elev',
                        x_col='lon_c',
                        y_col='lat_c'
                    )
                    results = pool.map(bilinear_interp_df_p, centroid_df_split)
                    centroid_gdf = pd.concat(results)
            else:
                # Sequential processing
                logger.info("Using sequential processing")
                bilinear_interp_df_p = functools.partial(
                    gbfc.bilinear_interp_df,
                    tif=str(dtm_path),
                    var='model_elev',
                    x_col='lon_c',
                    y_col='lat_c'
                )
                centroid_gdf = bilinear_interp_df_p(centroid_gdf)
            
            # Round elevation values to specified precision
            centroid_gdf['model_elev'] = [
                round(elev, self.vertical_res) for elev in centroid_gdf['model_elev']
            ]
            
            logger.info("Elevation interpolation completed")
            return centroid_gdf
            
        except Exception as e:
            logger.error(f"Failed to interpolate elevation: {e}")
            raise
            
    def save_results(self, centroid_gdf: gpd.GeoDataFrame) -> None:
        """
        Save quantized elevation results to CSV file.
        
        Args:
            centroid_gdf: GeoDataFrame containing results
        """
        try:
            # Remove unnecessary columns
            output_df = centroid_gdf.drop(columns=['lon_c', 'lat_c', 'geometry'])
            
            output_path = self.result_dir / f"NB_elev_{self.resolution}.csv"
            
            logger.info(f"Saving results to {output_path}")
            output_df.to_csv(output_path, index=False)
            
            logger.info(f"Saved {len(output_df)} elevation records")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            raise
            
    def run_quantization(self, use_parallel: bool = False, n_cores: Optional[int] = None) -> None:
        """
        Run the complete elevation quantization process.
        
        Args:
            use_parallel: Whether to use parallel processing
            n_cores: Number of cores for parallel processing
        """
        start_time = time.time()
        
        try:
            logger.info(f"Starting elevation quantization for resolution {self.resolution}")
            
            # Load centroid data
            centroid_gdf = self.load_centroid_data()
            
            # Filter by study area
            centroid_gdf = self.filter_by_study_area(centroid_gdf)
            
            # Interpolate elevation
            centroid_gdf = self.interpolate_elevation(centroid_gdf, use_parallel, n_cores)
            
            # Save results
            self.save_results(centroid_gdf)
            
            processing_time = time.time() - start_time
            logger.info(f"Elevation quantization completed in {processing_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Elevation quantization failed: {e}")
            raise


def main():
    """
    Main function to run elevation quantization.
    
    Command line usage:
        python Quantization_elev.py <resolution> [--parallel] [--n_cores <cores>]
    """
    parser = argparse.ArgumentParser(
        description="Elevation quantization for DGGS flood modeling"
    )
    parser.add_argument(
        "resolution", 
        type=int, 
        help="DGGS resolution level (16-29)"
    )
    parser.add_argument(
        "--parallel", 
        action="store_true", 
        help="Use parallel processing"
    )
    parser.add_argument(
        "--n_cores", 
        type=int, 
        default=None,
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
        quantizer = ElevationQuantizer(
            resolution=args.resolution,
            data_dir=args.data_dir,
            result_dir=args.result_dir
        )
        
        # Run quantization
        quantizer.run_quantization(
            use_parallel=args.parallel,
            n_cores=args.n_cores
        )
        
    except Exception as e:
        logger.error(f"Elevation quantization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
