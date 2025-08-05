"""
Topographic Quantization Script for DGGS Flood Modeling

This script performs topographic quantization for Discrete Global Grid System (DGGS)
by computing topographic parameters including slope, aspect, curvature, roughness,
TRI, and TPI for sample points.

The script interpolates elevation data and calculates various topographic indices
for flood modeling applications.

Author: Erin Li
"""

import warnings
import sys
import logging
import argparse
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd

# Import local modules
import DgBaseFunc as dbfc
import DgTopogFunc as dtfc
import GeoBaseFunc as gbfc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.simplefilter('error', RuntimeWarning)


class TopographicQuantizer:
    """
    A class to handle topographic quantization for DGGS flood modeling.
    
    This class provides methods to compute topographic parameters including
    slope, aspect, curvature, roughness, TRI, and TPI for sample points.
    """
    
    def __init__(self, resolution: int, data_dir: str = "Data", result_dir: str = "Result"):
        """
        Initialize the topographic quantizer.
        
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
        self.data_dir.mkdir(exist_ok=True)
        
        # Load lookup table for cell properties
        self.lookup_table = dbfc.look_up_table()
        self.cell_spacing = self.lookup_table.loc[resolution, 'cell_spacing'] * 1000
        self.vertical_res = self.lookup_table.loc[resolution, 'verti_res']
        
        logger.info(f"Initialized TopographicQuantizer for resolution {resolution}")
        
    def load_elevation_data(self) -> pd.DataFrame:
        """
        Load elevation data from CSV file.
        
        Returns:
            DataFrame containing elevation data
            
        Raises:
            FileNotFoundError: If elevation file doesn't exist
        """
        try:
            input_csv_path = self.result_dir / f"NB_elev_{self.resolution}.csv"
            
            if not input_csv_path.exists():
                raise FileNotFoundError(f"Elevation file not found: {input_csv_path}")
                
            logger.info(f"Loading elevation data from {input_csv_path}")
            
            elev_df = pd.read_csv(input_csv_path, sep=',')
            elev_df = elev_df.set_index(['i', 'j'])
            
            logger.info(f"Loaded {len(elev_df)} elevation records")
            return elev_df
            
        except Exception as e:
            logger.error(f"Failed to load elevation data: {e}")
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
            input_csv_path = self.data_dir / f"NB_sample_cent_quanti_{self.resolution}.csv"
            
            if not input_csv_path.exists():
                raise FileNotFoundError(f"Sample data file not found: {input_csv_path}")
                
            logger.info(f"Loading sample data from {input_csv_path}")
            
            sdf = pd.read_csv(input_csv_path, sep=',')
            sdf = sdf.set_index(['i', 'j'])
            
            logger.info(f"Loaded {len(sdf)} sample records")
            return sdf
            
        except Exception as e:
            logger.error(f"Failed to load sample data: {e}")
            raise
            
    def interpolate_elevation(self, sdf: pd.DataFrame) -> pd.DataFrame:
        """
        Interpolate elevation values for sample points.
        
        Args:
            sdf: DataFrame containing sample data
            
        Returns:
            DataFrame with interpolated elevation values
            
        Raises:
            FileNotFoundError: If DTM file doesn't exist
        """
        try:
            logger.info("Interpolating elevation values for sample points")
            
            dtm_path = self.data_dir / "NB_DTM_O_30_wgs84.tif"
            
            if not dtm_path.exists():
                raise FileNotFoundError(f"DTM file not found: {dtm_path}")
                
            # Interpolate elevation using bilinear interpolation
            sdf = gbfc.bilinear_interp_df(
                sdf, str(dtm_path), 'model_elev', 'lon_c', 'lat_c'
            )
            
            # Round elevation values to specified precision
            sdf['model_elev'] = [round(elev, self.vertical_res) for elev in sdf['model_elev']]
            
            logger.info("Elevation interpolation completed")
            return sdf
            
        except Exception as e:
            logger.error(f"Failed to interpolate elevation: {e}")
            raise
            
    def compute_topographic_parameters(self, sdf: pd.DataFrame, elev_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute topographic parameters for sample points.
        
        Args:
            sdf: DataFrame containing sample data
            elev_df: DataFrame containing elevation data
            
        Returns:
            DataFrame with computed topographic parameters
        """
        try:
            logger.info("Computing topographic parameters")
            
            # Compute slope and aspect using FDA method
            logger.info("Computing slope and aspect")
            sdf = dtfc.slope_aspect_df(sdf, elev_df, 'FDA', self.resolution, self.cell_spacing)
            
            # Compute curvature
            logger.info("Computing curvature")
            sdf = dtfc.curvature_df(sdf, elev_df, self.resolution, self.cell_spacing)
            
            # Compute roughness
            logger.info("Computing roughness")
            sdf = dtfc.roughness_df(sdf, elev_df, self.resolution, self.vertical_res, 1)
            
            # Compute Terrain Roughness Index (TRI)
            logger.info("Computing TRI")
            sdf = dtfc.TRI_df(sdf, self.resolution, elev_df)
            
            # Compute Topographic Position Index (TPI)
            logger.info("Computing TPI")
            sdf = dtfc.TPI_df(sdf, self.resolution, elev_df)
            
            logger.info("Topographic parameter computation completed")
            return sdf
            
        except Exception as e:
            logger.error(f"Failed to compute topographic parameters: {e}")
            raise
            
    def rename_columns(self, sdf: pd.DataFrame) -> pd.DataFrame:
        """
        Rename columns for consistency.
        
        Args:
            sdf: DataFrame containing topographic data
            
        Returns:
            DataFrame with renamed columns
        """
        try:
            logger.info("Renaming columns")
            
            # Rename model_elev to dtm for consistency
            sdf = sdf.rename(columns={"model_elev": "dtm"})
            
            logger.info("Column renaming completed")
            return sdf
            
        except Exception as e:
            logger.error(f"Failed to rename columns: {e}")
            raise
            
    def save_results(self, sdf: pd.DataFrame) -> None:
        """
        Save topographic quantization results to CSV file.
        
        Args:
            sdf: DataFrame containing results
        """
        try:
            output_path = self.data_dir / f"NB_sample_cent_quanti2_{self.resolution}.csv"
            
            logger.info(f"Saving results to {output_path}")
            sdf.to_csv(output_path, index=True)
            
            logger.info(f"Saved {len(sdf)} topographic records")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            raise
            
    def run_quantization(self) -> None:
        """
        Run the complete topographic quantization process.
        """
        try:
            logger.info(f"Starting topographic quantization for resolution {self.resolution}")
            
            # Load elevation data
            elev_df = self.load_elevation_data()
            
            # Load sample data
            sdf = self.load_sample_data()
            
            # Interpolate elevation for sample points
            sdf = self.interpolate_elevation(sdf)
            
            # Compute topographic parameters
            sdf = self.compute_topographic_parameters(sdf, elev_df)
            
            # Rename columns for consistency
            sdf = self.rename_columns(sdf)
            
            # Save results
            self.save_results(sdf)
            
            logger.info("Topographic quantization completed successfully")
            
        except Exception as e:
            logger.error(f"Topographic quantization failed: {e}")
            raise


def main():
    """
    Main function to run topographic quantization.
    
    Command line usage:
        python Quantization_topog.py <resolution>
    """
    parser = argparse.ArgumentParser(
        description="Topographic quantization for DGGS flood modeling"
    )
    parser.add_argument(
        "resolution", 
        type=int, 
        help="DGGS resolution level (16-29)"
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
        quantizer = TopographicQuantizer(
            resolution=args.resolution,
            data_dir=args.data_dir,
            result_dir=args.result_dir
        )
        
        # Run quantization
        quantizer.run_quantization()
        
    except Exception as e:
        logger.error(f"Topographic quantization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

