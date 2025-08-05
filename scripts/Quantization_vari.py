"""
Variable Quantization Script for DGGS Flood Modeling

This script performs variable quantization for Discrete Global Grid System (DGGS)
by interpolating various environmental variables including geology, water level,
soil texture, land cover, NDVI, MSI, and climate variables.

The script handles coordinate reprojection and uses different interpolation
methods (nearest neighbor for categorical data, bilinear for continuous data).

Author: Erin Li
"""

import logging
import argparse
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np

# Import local modules
import GeoBaseFunc as gbfc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VariableQuantizer:
    """
    A class to handle variable quantization for DGGS flood modeling.
    
    This class provides methods to interpolate various environmental variables
    including geology, water level, soil texture, land cover, NDVI, MSI,
    and climate variables for flood modeling applications.
    """
    
    def __init__(self, resolution: int, data_dir: str = "../Data", 
                 data_clip_dir: str = "../Data_clip_tif"):
        """
        Initialize the variable quantizer.
        
        Args:
            resolution: DGGS resolution level
            data_dir: Directory containing input data files
            data_clip_dir: Directory containing clipped raster data
        """
        self.resolution = resolution
        self.data_dir = Path(data_dir)
        self.data_clip_dir = Path(data_clip_dir)
        
        # Validate resolution
        if not isinstance(resolution, int) or resolution < 16 or resolution > 29:
            raise ValueError("Resolution must be an integer between 16 and 29")
            
        # Create output directory if it doesn't exist
        self.data_dir.mkdir(exist_ok=True)
        
        logger.info(f"Initialized VariableQuantizer for resolution {resolution}")
        
    def load_sample_data(self) -> pd.DataFrame:
        """
        Load sample data from CSV file.
        
        Returns:
            DataFrame containing sample data
            
        Raises:
            FileNotFoundError: If sample file doesn't exist
        """
        try:
            input_csv_path = self.data_dir / f"NB_sample_cent_{self.resolution}.csv"
            
            if not input_csv_path.exists():
                raise FileNotFoundError(f"Sample data file not found: {input_csv_path}")
                
            logger.info(f"Loading sample data from {input_csv_path}")
            
            sdf = pd.read_csv(
                input_csv_path, 
                sep=',',
                usecols=['Cell_address', 'lon_c', 'lat_c', 'i', 'j', 'grid_code']
            )
            
            logger.info(f"Loaded {len(sdf)} sample records")
            return sdf
            
        except Exception as e:
            logger.error(f"Failed to load sample data: {e}")
            raise
            
    def load_climate_data(self) -> pd.DataFrame:
        """
        Load climate station data from CSV file.
        
        Returns:
            DataFrame containing climate data
            
        Raises:
            FileNotFoundError: If climate file doesn't exist
        """
        try:
            input_csv_path = self.data_dir / f"NB_climate_cent_{self.resolution}.csv"
            
            if not input_csv_path.exists():
                raise FileNotFoundError(f"Climate data file not found: {input_csv_path}")
                
            logger.info(f"Loading climate data from {input_csv_path}")
            
            cdf = pd.read_csv(
                input_csv_path, 
                sep=',',
                usecols=['E_NORMAL_E', 'MONTH', 'lon_c', 'lat_c', 'i', 'j', 'VALUE']
            )
            
            logger.info(f"Loaded {len(cdf)} climate records")
            return cdf
            
        except Exception as e:
            logger.error(f"Failed to load climate data: {e}")
            raise
            
    def reproject_coordinates(self, sdf: pd.DataFrame) -> pd.DataFrame:
        """
        Reproject coordinates from WGS84 to Canada Atlas Lambert.
        
        Args:
            sdf: DataFrame containing sample data
            
        Returns:
            DataFrame with reprojected coordinates
        """
        try:
            logger.info("Reprojecting coordinates from WGS84 to Canada Atlas Lambert")
            
            # Reproject coordinates
            sdf[['lon_c_p', 'lat_c_p']] = [
                gbfc.reproject_coords(x, y) for x, y in zip(sdf.lon_c, sdf.lat_c)
            ]
            
            logger.info("Coordinate reprojection completed")
            return sdf
            
        except Exception as e:
            logger.error(f"Failed to reproject coordinates: {e}")
            raise
            
    def interpolate_geomorphology_variables(self, sdf: pd.DataFrame) -> pd.DataFrame:
        """
        Interpolate geomorphology variables using nearest neighbor interpolation.
        
        Args:
            sdf: DataFrame containing sample data
            
        Returns:
            DataFrame with interpolated geomorphology variables
            
        Raises:
            FileNotFoundError: If raster files don't exist
        """
        try:
            logger.info("Interpolating geomorphology variables")
            
            # Define raster file paths
            raster_files = {
                'geo': 'out_geology.tif',
                'wl': 'NB_WL_O_30.tif',
                'sol': 'out_BIO_ECODIST_SOIL_TEXTURE.tif',
                'lc': 'out_lc.tif',
                'ndvi': 'NB_NDVI_O_30.tif',
                'msi': 'out_Terra_MODIS_2015_MSI_Probability_100_v4_20180122.tif'
            }
            
            # Interpolate each variable
            for var_name, raster_file in raster_files.items():
                raster_path = self.data_clip_dir / raster_file
                
                if not raster_path.exists():
                    raise FileNotFoundError(f"Raster file not found: {raster_path}")
                    
                logger.info(f"Interpolating {var_name} from {raster_file}")
                
                if var_name in ['ndvi', 'msi']:
                    # Use bilinear interpolation for continuous variables
                    sdf = gbfc.bilinear_interp_df(
                        sdf, str(raster_path), var_name, 'lon_c_p', 'lat_c_p'
                    )
                else:
                    # Use nearest neighbor interpolation for categorical variables
                    sdf = gbfc.nearest_interp_df(
                        sdf, str(raster_path), var_name, 'lon_c_p', 'lat_c_p'
                    )
            
            logger.info("Geomorphology variable interpolation completed")
            return sdf
            
        except Exception as e:
            logger.error(f"Failed to interpolate geomorphology variables: {e}")
            raise
            
    def populate_derived_variables(self, sdf: pd.DataFrame) -> pd.DataFrame:
        """
        Populate derived variables based on interpolated data.
        
        Args:
            sdf: DataFrame containing interpolated variables
            
        Returns:
            DataFrame with derived variables
        """
        try:
            logger.info("Populating derived variables")
            
            # Populate impervious area (ia) - land cover type 17
            sdf['ia'] = np.where(sdf['lc'] == 17, 1, 0)
            
            # Populate forest cover percentage (fcp) - land cover types 1,2,5,6,8
            forest_types = [1, 2, 5, 6, 8]
            sdf['fcp'] = np.where(sdf['lc'].isin(forest_types), 1, 0)
            
            # Populate water level (wl) - convert to binary
            sdf['wl'] = np.where(sdf['wl'] > 0, 1, 0)
            
            logger.info("Derived variable population completed")
            return sdf
            
        except Exception as e:
            logger.error(f"Failed to populate derived variables: {e}")
            raise
            
    def interpolate_climate_variables(self, sdf: pd.DataFrame, cdf: pd.DataFrame) -> pd.DataFrame:
        """
        Interpolate climate variables using IDW interpolation.
        
        Args:
            sdf: DataFrame containing sample data
            cdf: DataFrame containing climate station data
            
        Returns:
            DataFrame with interpolated climate variables
        """
        try:
            logger.info("Interpolating climate variables")
            
            # Define climate variables and their parameters
            climate_variables = {
                'precip': ('Total precipitation mm', 13),
                'tavg': ('Mean daily temperature deg C', 13),
                'r10': ('Days with rainfall GE 10 mm', 13),
                'r25': ('Days with rainfall GE 25 mm', 13),
                'tm10': ('Days with daily min temperature LT -10 deg C', 13),
                'sd50': ('Days with snow depth GE 50 cm', 13),
                'ts': ('Total snowfall cm', 13)
            }
            
            # Interpolate each climate variable
            for var_name, (description, month) in climate_variables.items():
                logger.info(f"Interpolating {var_name}: {description}")
                sdf = gbfc.IDW_interp_df(sdf, cdf, description, month, var_name, self.resolution)
            
            # Interpolate spring variables (months 3, 4, 5)
            spring_months = [3, 4, 5]
            spring_vars = ['spr1', 'spr2', 'spr3']
            
            for month, var_name in zip(spring_months, spring_vars):
                logger.info(f"Interpolating {var_name} for month {month}")
                sdf = gbfc.IDW_interp_df(
                    sdf, cdf, 'Days with daily min temperature GT 0 deg C', 
                    month, var_name, self.resolution
                )
            
            # Calculate total spring days
            sdf['spr'] = sdf['spr1'] + sdf['spr2'] + sdf['spr3']
            
            logger.info("Climate variable interpolation completed")
            return sdf
            
        except Exception as e:
            logger.error(f"Failed to interpolate climate variables: {e}")
            raise
            
    def clean_and_save_results(self, sdf: pd.DataFrame) -> None:
        """
        Clean up temporary columns and save results.
        
        Args:
            sdf: DataFrame containing all variables
        """
        try:
            logger.info("Cleaning up and saving results")
            
            # Remove temporary columns
            temp_columns = ['lon_c_p', 'lat_c_p', 'spr1', 'spr2', 'spr3']
            sdf = sdf.drop(columns=temp_columns)
            
            # Save results
            output_path = self.data_dir / f"NB_sample_cent_quanti_{self.resolution}.csv"
            
            logger.info(f"Saving results to {output_path}")
            sdf.to_csv(output_path, index=False)
            
            logger.info(f"Saved {len(sdf)} variable records")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            raise
            
    def run_quantization(self) -> None:
        """
        Run the complete variable quantization process.
        """
        try:
            logger.info(f"Starting variable quantization for resolution {self.resolution}")
            
            # Load data
            sdf = self.load_sample_data()
            cdf = self.load_climate_data()
            
            # Reproject coordinates
            sdf = self.reproject_coordinates(sdf)
            
            # Interpolate geomorphology variables
            sdf = self.interpolate_geomorphology_variables(sdf)
            
            # Populate derived variables
            sdf = self.populate_derived_variables(sdf)
            
            # Interpolate climate variables
            sdf = self.interpolate_climate_variables(sdf, cdf)
            
            # Clean up and save results
            self.clean_and_save_results(sdf)
            
            logger.info("Variable quantization completed successfully")
            
        except Exception as e:
            logger.error(f"Variable quantization failed: {e}")
            raise


def main():
    """
    Main function to run variable quantization.
    
    Command line usage:
        python Quantization_vari.py <resolution>
    """
    parser = argparse.ArgumentParser(
        description="Variable quantization for DGGS flood modeling"
    )
    parser.add_argument(
        "resolution", 
        type=int, 
        help="DGGS resolution level (16-29)"
    )
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="../Data",
        help="Directory containing input data files"
    )
    parser.add_argument(
        "--data_clip_dir", 
        type=str, 
        default="../Data_clip_tif",
        help="Directory containing clipped raster data"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize quantizer
        quantizer = VariableQuantizer(
            resolution=args.resolution,
            data_dir=args.data_dir,
            data_clip_dir=args.data_clip_dir
        )
        
        # Run quantization
        quantizer.run_quantization()
        
    except Exception as e:
        logger.error(f"Variable quantization failed: {e}")
        raise


if __name__ == "__main__":
    main()
