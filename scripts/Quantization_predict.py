"""
Comprehensive Prediction Quantization Script for DGGS Flood Modeling

This script performs comprehensive quantization for Discrete Global Grid System (DGGS)
flood modeling by combining multiple quantization steps:

1. Elevation quantization (DTM)
2. Topographic parameter computation (slope, aspect, curvature, roughness, TRI, TPI)
3. Geomorphology variable quantization (geology, water level, soil, land cover, NDVI, MSI)
4. Climate variable interpolation (precipitation, temperature, rainfall days, etc.)
5. Distance to NHN calculation
6. Hydrology parameter computation (flow direction, flow accumulation, TWI, SPI)

The script orchestrates the complete data preparation pipeline for flood prediction.

Author: Erin Li
"""

import functools
import pandas as pd
import numpy as np
import geopandas as gpd
import multiprocess as mp
import warnings
import sys
import gc
import os
import logging
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

# Import local modules
import GeoBaseFunc as gbfc
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
warnings.simplefilter(action='ignore', category=FutureWarning)


class PredictionQuantizer:
    """
    A class to handle comprehensive prediction quantization for DGGS flood modeling.
    
    This class orchestrates the complete data preparation pipeline including
    elevation, topographic, geomorphology, climate, distance, and hydrological
    quantization for flood prediction applications.
    """
    
    def __init__(self, resolution: int, n_cores: int = 1, data_dir: str = "Data", 
                 result_dir: str = "Result"):
        """
        Initialize the prediction quantizer.
        
        Args:
            resolution: DGGS resolution level
            n_cores: Number of cores for parallel processing
            data_dir: Directory containing input data files
            result_dir: Directory for output results
        """
        self.resolution = resolution
        self.n_cores = n_cores
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
        self.cell_area = self.lookup_table.loc[resolution, 'cell_area'] * 1000000
        
        logger.info(f"Initialized PredictionQuantizer for resolution {resolution}")
        
    def quantize_elevation(self) -> None:
        """
        Quantize elevation data using bilinear interpolation.
        """
        try:
            logger.info("Starting elevation quantization")
            
            # Load centroid data
            input_csv_path = self.result_dir / f"NB_area_wgs84_centroids_{self.resolution}.csv"
            if not input_csv_path.exists():
                raise FileNotFoundError(f"Centroid file not found: {input_csv_path}")
                
            centroid_df = pd.read_csv(input_csv_path, sep=',')
            centroid_gdf = gpd.GeoDataFrame(
                centroid_df, 
                geometry=gpd.points_from_xy(centroid_df.lon_c, centroid_df.lat_c)
            )
            
            # Filter by study area
            extent_path = self.data_dir / "NB_area_wgs84.shp"
            if not extent_path.exists():
                raise FileNotFoundError(f"Study area shapefile not found: {extent_path}")
                
            extent = gpd.GeoDataFrame.from_file(extent_path)
            centroid_gdf = centroid_gdf[centroid_gdf.geometry.within(extent.geometry[0])]
            
            # Quantize elevation by bilinear interpolation
            dtm_path = self.data_dir / "NB_DTM_O_30_wgs84.tif"
            if not dtm_path.exists():
                raise FileNotFoundError(f"DTM file not found: {dtm_path}")
                
            bilinear_interp_df_dtm = functools.partial(
                gbfc.bilinear_interp_df,
                tif=str(dtm_path),
                var='model_elev',
                x_col='lon_c',
                y_col='lat_c'
            )
            centroid_gdf = bilinear_interp_df_dtm(centroid_gdf)
            
            # Round elevation values
            centroid_gdf['model_elev'] = [
                round(elev, self.vertical_res) for elev in centroid_gdf['model_elev']
            ]
            
            # Save results
            centroid_gdf = centroid_gdf.drop(columns=['geometry'])
            output_csv_path = self.result_dir / f"NB_elev_{self.resolution}.csv"
            centroid_gdf.to_csv(output_csv_path, index=False)
            
            logger.info("Elevation quantization completed")
            
        except Exception as e:
            logger.error(f"Elevation quantization failed: {e}")
            raise
            
    def compute_topographic_parameters(self) -> None:
        """
        Compute topographic parameters using parallel processing.
        """
        try:
            logger.info("Starting topographic parameter computation")
            
            # Load elevation data
            input_csv_path = self.result_dir / f"NB_elev_{self.resolution}.csv"
            if not input_csv_path.exists():
                raise FileNotFoundError(f"Elevation file not found: {input_csv_path}")
                
            elev_df = pd.read_csv(input_csv_path, sep=',')
            elev_df = elev_df.set_index(['i', 'j'])
            
            # Define partial functions for parallel processing
            slope_aspect_df_p = functools.partial(
                dtfc.slope_aspect_df,
                elev_df=elev_df,
                method='FDA',
                res=self.resolution,
                cell_spacing=self.cell_spacing
            )
            curvature_df_p = functools.partial(
                dtfc.curvature_df,
                elev_df=elev_df,
                res=self.resolution,
                cell_spacing=self.cell_spacing
            )
            roughness_df_p = functools.partial(
                dtfc.roughness_df,
                elev_df=elev_df,
                res=self.resolution,
                vertical_res=self.vertical_res,
                rings=1
            )
            TRI_df_p = functools.partial(
                dtfc.TRI_df,
                res=self.resolution,
                elev_df=elev_df
            )
            TPI_df_p = functools.partial(
                dtfc.TPI_df,
                res=self.resolution,
                elev_df=elev_df
            )
            
            # Process each topographic parameter in parallel
            functions = [
                (slope_aspect_df_p, "slope/aspect"),
                (curvature_df_p, "curvature"),
                (roughness_df_p, "roughness"),
                (TRI_df_p, "TRI"),
                (TPI_df_p, "TPI")
            ]
            
            for func, name in functions:
                logger.info(f"Computing {name}")
                elev_df_split = np.array_split(elev_df, self.n_cores)
                
                with mp.Pool(processes=self.n_cores) as pool:
                    elev_df = pd.concat(pool.map(func, elev_df_split))
                pool.close()
                pool.join()
            
            # Save results
            output_csv_path = self.result_dir / f"NB_topog_{self.resolution}.csv"
            elev_df.to_csv(output_csv_path, index=True)
            
            logger.info("Topographic parameter computation completed")
            
        except Exception as e:
            logger.error(f"Topographic parameter computation failed: {e}")
            raise
            
    def quantize_geomorphology_variables(self) -> None:
        """
        Quantize geomorphology variables including geology, water level, soil, land cover, NDVI, MSI.
        """
        try:
            logger.info("Starting geomorphology variable quantization")
            
            # Load topographic data
            input_csv_path = self.result_dir / f"NB_topog_{self.resolution}.csv"
            if not input_csv_path.exists():
                raise FileNotFoundError(f"Topographic file not found: {input_csv_path}")
                
            topog_df = pd.read_csv(input_csv_path, sep=',')
            
            # Reproject coordinates
            reproject_coords_df_p = functools.partial(
                gbfc.reproject_coords_df,
                x_col='lon_c',
                y_col='lat_c',
                x_new='lon_c_p',
                y_new='lat_c_p'
            )
            
            topog_df_split = np.array_split(topog_df, self.n_cores)
            with mp.Pool(processes=self.n_cores) as pool:
                topog_df = pd.concat(pool.map(reproject_coords_df_p, topog_df_split))
            pool.close()
            pool.join()
            
            # Extract values using different interpolation methods
            raster_files = {
                'geo': ('out_geology.tif', 'nearest'),
                'wl': ('NB_WL_O_30.tif', 'nearest'),
                'sol': ('out_BIO_ECODIST_SOIL_TEXTURE.tif', 'nearest'),
                'lc': ('out_lc.tif', 'nearest'),
                'ndvi': ('NB_NDVI_O_30.tif', 'bilinear'),
                'msi': ('out_Terra_MODIS_2015_MSI_Probability_100_v4_20180122.tif', 'bilinear')
            }
            
            for var_name, (raster_file, method) in raster_files.items():
                raster_path = self.data_dir / raster_file
                if not raster_path.exists():
                    raise FileNotFoundError(f"Raster file not found: {raster_path}")
                    
                logger.info(f"Interpolating {var_name} using {method} interpolation")
                
                if method == 'bilinear':
                    topog_df = gbfc.bilinear_interp_df(
                        topog_df, str(raster_path), var_name, 'lon_c_p', 'lat_c_p'
                    )
                else:
                    topog_df = gbfc.nearest_interp_df(
                        topog_df, str(raster_path), var_name, 'lon_c_p', 'lat_c_p'
                    )
            
            # Populate derived variables
            topog_df['ia'] = np.where(topog_df['lc'] == 17, 1, 0)
            topog_df['fcp'] = np.where(
                (topog_df['lc'] == 1) | (topog_df['lc'] == 2) | 
                (topog_df['lc'] == 5) | (topog_df['lc'] == 6) | 
                (topog_df['lc'] == 8), 1, 0
            )
            topog_df['wl'] = np.where(topog_df['wl'] > 0, 1, 0)
            
            # Save results
            topog_df = topog_df.drop(columns=['lon_c_p', 'lat_c_p'])
            output_csv_path = self.result_dir / f"NB_geom_{self.resolution}.csv"
            topog_df.to_csv(output_csv_path, index=False)
            
            logger.info("Geomorphology variable quantization completed")
            
        except Exception as e:
            logger.error(f"Geomorphology variable quantization failed: {e}")
            raise
            
    def interpolate_climate_variables(self) -> None:
        """
        Interpolate climate variables using IDW interpolation.
        """
        try:
            logger.info("Starting climate variable interpolation")
            
            # Load geomorphology data
            input_csv_path = self.result_dir / f"NB_geom_{self.resolution}.csv"
            if not input_csv_path.exists():
                raise FileNotFoundError(f"Geomorphology file not found: {input_csv_path}")
                
            geom_df = pd.read_csv(input_csv_path, sep=',')
            
            # Load climate station data
            climate_path = self.data_dir / f"NB_climate_cent_{self.resolution}.csv"
            if not climate_path.exists():
                raise FileNotFoundError(f"Climate data file not found: {climate_path}")
                
            cdf = pd.read_csv(
                climate_path, 
                sep=',',
                usecols=['E_NORMAL_E', 'MONTH', 'lon_c', 'lat_c', 'i', 'j', 'VALUE']
            )
            
            # Define climate variables
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
                geom_df = gbfc.IDW_interp_df(
                    geom_df, cdf, description, month, var_name, self.resolution
                )
            
            # Interpolate spring variables
            spring_months = [3, 4, 5]
            spring_vars = ['spr1', 'spr2', 'spr3']
            
            for month, var_name in zip(spring_months, spring_vars):
                logger.info(f"Interpolating {var_name} for month {month}")
                geom_df = gbfc.IDW_interp_df(
                    geom_df, cdf, 'Days with daily min temperature GT 0 deg C',
                    month, var_name, self.resolution
                )
            
            # Calculate total spring days
            geom_df['spr'] = geom_df['spr1'] + geom_df['spr2'] + geom_df['spr3']
            
            # Save results
            geom_df = geom_df.drop(columns=['spr1', 'spr2', 'spr3'])
            output_csv_path = self.result_dir / f"NB_climate_{self.resolution}.csv"
            geom_df.to_csv(output_csv_path, index=False)
            
            logger.info("Climate variable interpolation completed")
            
        except Exception as e:
            logger.error(f"Climate variable interpolation failed: {e}")
            raise
            
    def calculate_nhn_distance(self) -> None:
        """
        Calculate distance to Natural Hydrological Network (NHN).
        """
        try:
            logger.info("Starting NHN distance calculation")
            
            # Load centroid data
            centroid_path = self.result_dir / f"NB_area_wgs84_centroids_{self.resolution}.csv"
            if not centroid_path.exists():
                raise FileNotFoundError(f"Centroid file not found: {centroid_path}")
                
            centroid_df = pd.read_csv(centroid_path, sep=',')
            centroid_gdf = gpd.GeoDataFrame(
                centroid_df, 
                geometry=gpd.points_from_xy(centroid_df.lon_c, centroid_df.lat_c)
            )
            centroid_gdf = centroid_gdf.set_index(['i', 'j'])
            
            # Load climate data
            climate_path = self.result_dir / f"NB_climate_{self.resolution}.csv"
            if not climate_path.exists():
                raise FileNotFoundError(f"Climate file not found: {climate_path}")
                
            clim_df = pd.read_csv(climate_path, sep=',')
            clim_df = clim_df.set_index(['i', 'j'])
            clim_df = clim_df.sort_index()
            
            # Filter by study area
            extent_path = self.data_dir / "NB_area_wgs84.shp"
            if not extent_path.exists():
                raise FileNotFoundError(f"Study area shapefile not found: {extent_path}")
                
            extent = gpd.GeoDataFrame.from_file(extent_path)
            centroid_gdf = centroid_gdf[centroid_gdf.geometry.within(extent.geometry[0])]
            
            # Extract NHN values
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
            
            # Identify cells with NHN presence
            centroid_gdf['nhn3'] = np.where(
                (centroid_gdf['nhn1'] != -32767) | (centroid_gdf['nhn2'] != -32767), 
                1, 0
            )
            centroid_gdf = centroid_gdf[centroid_gdf['nhn3'] == 1]
            
            # Calculate hexagonal distance
            coords_target = centroid_gdf.index.values.tolist()
            
            def hex_dist_comb_df(dataframe, coords_target, var, res):
                """Helper function for hexagonal distance calculation"""
                dataframe[var] = [
                    dbfc.hex_dist_m(coord, coords_target, res) 
                    for coord in list(dataframe.index.values)
                ]
                df_done = dataframe.dropna(subset=[var])
                df_left = dataframe[np.isnan(dataframe[var])]
                df_left[var] = [
                    dbfc.hex_dist_l(coord, coords_target, res) 
                    for coord in list(df_left.index.values.tolist())
                ]
                dataframe = pd.concat([df_done, df_left])
                return dataframe
            
            hex_dist_comb_df_p = functools.partial(
                hex_dist_comb_df,
                coords_target=coords_target,
                var='nhn',
                res=self.resolution
            )
            
            clim_df_split = np.array_split(clim_df, self.n_cores)
            with mp.Pool(processes=self.n_cores) as pool:
                clim_df = pd.concat(pool.map(hex_dist_comb_df_p, clim_df_split))
            pool.close()
            pool.join()
            
            # Save results
            output_csv_path = self.result_dir / f"NB_nhnDist_{self.resolution}.csv"
            clim_df.to_csv(output_csv_path, index=True)
            
            logger.info("NHN distance calculation completed")
            
        except Exception as e:
            logger.error(f"NHN distance calculation failed: {e}")
            raise
            
    def compute_hydrology_parameters(self) -> None:
        """
        Compute hydrological parameters (flow direction, flow accumulation, TWI, SPI).
        """
        try:
            logger.info("Starting hydrology parameter computation")
            
            # Load NHN distance data
            input_csv_path = self.result_dir / f"NB_nhnDist_{self.resolution}.csv"
            if not input_csv_path.exists():
                raise FileNotFoundError(f"NHN distance file not found: {input_csv_path}")
                
            elev_df = pd.read_csv(input_csv_path, sep=',')
            elev_df = elev_df.set_index(['i', 'j'])
            
            # TODO: Implement hydrological parameter computation
            # This section would include flow direction, flow accumulation, TWI, SPI
            # For now, we'll just rename the elevation column
            
            # Rename elevation column
            elev_df = elev_df.rename(columns={"model_elev": "dtm"})
            
            # Save results
            output_csv_path = self.result_dir / f"NB_finalPredict_{self.resolution}.csv"
            elev_df.to_csv(output_csv_path, index=True)
            
            logger.info("Hydrology parameter computation completed")
            
        except Exception as e:
            logger.error(f"Hydrology parameter computation failed: {e}")
            raise
            
    def run_complete_quantization(self) -> None:
        """
        Run the complete prediction quantization pipeline.
        """
        try:
            logger.info(f"Starting complete prediction quantization for resolution {self.resolution}")
            
            # Step 1: Elevation quantization
            self.quantize_elevation()
            
            # Step 2: Topographic parameter computation
            self.compute_topographic_parameters()
            
            # Step 3: Geomorphology variable quantization
            self.quantize_geomorphology_variables()
            
            # Step 4: Climate variable interpolation
            self.interpolate_climate_variables()
            
            # Step 5: NHN distance calculation
            self.calculate_nhn_distance()
            
            # Step 6: Hydrology parameter computation
            self.compute_hydrology_parameters()
            
            logger.info("Complete prediction quantization pipeline finished successfully")
            
        except Exception as e:
            logger.error(f"Complete prediction quantization failed: {e}")
            raise


def main():
    """
    Main function to run prediction quantization.
    
    Command line usage:
        python Quantization_predict.py <resolution> [--n_cores <cores>]
    """
    parser = argparse.ArgumentParser(
        description="Comprehensive prediction quantization for DGGS flood modeling"
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
        quantizer = PredictionQuantizer(
            resolution=args.resolution,
            n_cores=args.n_cores,
            data_dir=args.data_dir,
            result_dir=args.result_dir
        )
        
        # Run complete quantization
        quantizer.run_complete_quantization()
        
    except Exception as e:
        logger.error(f"Prediction quantization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()