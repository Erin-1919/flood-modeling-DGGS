"""
Prediction Visualization Script for DGGS Flood Modeling

This script creates visualization plots for flood prediction results across
different resolutions and model configurations using datashader.

The script generates maps showing flood susceptibility predictions for
different DGGS resolutions (19, 21, 23) and model types (HG, HGPT, HG8M).

Author: Erin Li
"""

import datashader as ds
import matplotlib.pyplot as plt
import pandas as pd
from datashader.mpl_ext import dsshow
import gc
import logging
import argparse
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PredictionVisualizer:
    """
    A class to handle visualization of flood prediction results.
    
    This class provides methods to create maps showing flood susceptibility
    predictions for different resolutions and model configurations.
    """
    
    def __init__(self, result_dir: str = "../Result", output_dir: str = "../Result/img"):
        """
        Initialize the prediction visualizer.
        
        Args:
            result_dir: Directory containing prediction result files
            output_dir: Directory for output visualization files
        """
        self.result_dir = Path(result_dir)
        self.output_dir = Path(output_dir)
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Default configuration
        self.resolutions = [19, 21, 23]
        self.models = ['HG', 'HGPT', 'HG8M']
        self.plot_width = 300
        self.plot_height = 300
        self.colormap = 'Blues'
        
        logger.info("Initialized PredictionVisualizer")
        
    def load_prediction_data(self, resolution: int) -> pd.DataFrame:
        """
        Load prediction data for a specific resolution.
        
        Args:
            resolution: DGGS resolution level
            
        Returns:
            DataFrame containing prediction data
            
        Raises:
            FileNotFoundError: If prediction file doesn't exist
        """
        try:
            file_path = self.result_dir / f"NB_centPredict_{resolution}.csv"
            
            if not file_path.exists():
                raise FileNotFoundError(f"Prediction file not found: {file_path}")
                
            logger.info(f"Loading prediction data for resolution {resolution}")
            
            centroid_df = pd.read_csv(file_path, sep=',')
            
            # Validate required columns
            required_cols = ['lon_c', 'lat_c'] + self.models
            missing_cols = [col for col in required_cols if col not in centroid_df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
                
            logger.info(f"Loaded {len(centroid_df)} prediction records")
            return centroid_df
            
        except Exception as e:
            logger.error(f"Failed to load prediction data for resolution {resolution}: {e}")
            raise
            
    def create_prediction_map(self, data: pd.DataFrame, model: str, 
                            resolution: int, save_plot: bool = True) -> Optional[plt.Figure]:
        """
        Create a prediction map for a specific model and resolution.
        
        Args:
            data: DataFrame containing prediction data
            model: Model type (HG, HGPT, HG8M)
            resolution: DGGS resolution level
            save_plot: Whether to save the plot to file
            
        Returns:
            Matplotlib figure object or None if save_plot is True
        """
        try:
            logger.info(f"Creating prediction map for {model} at resolution {resolution}")
            
            # Create figure
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Create datashader plot
            artist = dsshow(
                data, 
                ds.Point('lon_c', 'lat_c'), 
                aggregator=ds.mean(model), 
                cmap=self.colormap, 
                plot_width=self.plot_width, 
                plot_height=self.plot_height, 
                ax=ax
            )
            
            # Customize plot appearance
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis("off")
            
            # Add title
            ax.set_title(f'Flood Prediction - {model} (Resolution {resolution})', 
                        fontsize=12, pad=20)
            
            # Add colorbar
            cbar = plt.colorbar(artist, ax=ax, shrink=0.8)
            cbar.set_label('Flood Susceptibility', fontsize=10)
            
            if save_plot:
                # Save the plot
                output_path = self.output_dir / f"NB_pred_{model}_{resolution}.png"
                plt.savefig(output_path, bbox_inches='tight', pad_inches=0.0, dpi=300)
                plt.close()
                logger.info(f"Saved prediction map to {output_path}")
                return None
            else:
                return fig
                
        except Exception as e:
            logger.error(f"Failed to create prediction map for {model} at resolution {resolution}: {e}")
            raise
            
    def generate_all_visualizations(self, resolutions: Optional[List[int]] = None, 
                                  models: Optional[List[str]] = None) -> None:
        """
        Generate visualization plots for all resolutions and models.
        
        Args:
            resolutions: List of resolutions to process (default: self.resolutions)
            models: List of models to process (default: self.models)
        """
        if resolutions is None:
            resolutions = self.resolutions
        if models is None:
            models = self.models
            
        try:
            logger.info("Starting visualization generation")
            
            for resolution in resolutions:
                try:
                    # Load prediction data for this resolution
                    prediction_data = self.load_prediction_data(resolution)
                    
                    for model in models:
                        try:
                            # Create and save prediction map
                            self.create_prediction_map(prediction_data, model, resolution)
                            
                        except Exception as e:
                            logger.error(f"Failed to create map for {model} at resolution {resolution}: {e}")
                            continue
                            
                except Exception as e:
                    logger.error(f"Failed to process resolution {resolution}: {e}")
                    continue
                    
            logger.info("Visualization generation completed")
            
        except Exception as e:
            logger.error(f"Visualization generation failed: {e}")
            raise
            
    def create_comparison_plot(self, resolution: int, save_plot: bool = True) -> Optional[plt.Figure]:
        """
        Create a comparison plot showing all models for a specific resolution.
        
        Args:
            resolution: DGGS resolution level
            save_plot: Whether to save the plot to file
            
        Returns:
            Matplotlib figure object or None if save_plot is True
        """
        try:
            logger.info(f"Creating comparison plot for resolution {resolution}")
            
            # Load prediction data
            prediction_data = self.load_prediction_data(resolution)
            
            # Create subplot grid
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle(f'Flood Prediction Comparison - Resolution {resolution}', fontsize=14)
            
            for i, model in enumerate(self.models):
                ax = axes[i]
                
                # Create datashader plot
                artist = dsshow(
                    prediction_data, 
                    ds.Point('lon_c', 'lat_c'), 
                    aggregator=ds.mean(model), 
                    cmap=self.colormap, 
                    plot_width=self.plot_width, 
                    plot_height=self.plot_height, 
                    ax=ax
                )
                
                # Customize plot appearance
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.axis("off")
                ax.set_title(f'{model}', fontsize=12)
                
                # Add colorbar
                cbar = plt.colorbar(artist, ax=ax, shrink=0.8)
                cbar.set_label('Flood Susceptibility', fontsize=10)
                
            plt.tight_layout()
            
            if save_plot:
                # Save the plot
                output_path = self.output_dir / f"NB_pred_comparison_{resolution}.png"
                plt.savefig(output_path, bbox_inches='tight', pad_inches=0.0, dpi=300)
                plt.close()
                logger.info(f"Saved comparison plot to {output_path}")
                return None
            else:
                return fig
                
        except Exception as e:
            logger.error(f"Failed to create comparison plot for resolution {resolution}: {e}")
            raise


def main():
    """
    Main function to run prediction visualization.
    
    Command line usage:
        python Visualization_pred.py [--result_dir <dir>] [--output_dir <dir>] [--comparison]
    """
    parser = argparse.ArgumentParser(
        description="Visualization of flood prediction results"
    )
    parser.add_argument(
        "--result_dir", 
        type=str, 
        default="../Result",
        help="Directory containing prediction result files"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="../Result/img",
        help="Directory for output visualization files"
    )
    parser.add_argument(
        "--comparison", 
        action="store_true", 
        help="Create comparison plots for each resolution"
    )
    parser.add_argument(
        "--resolutions", 
        type=int, 
        nargs='+',
        default=[19, 21, 23],
        help="List of resolutions to process"
    )
    parser.add_argument(
        "--models", 
        type=str, 
        nargs='+',
        default=['HG', 'HGPT', 'HG8M'],
        help="List of models to process"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize visualizer
        visualizer = PredictionVisualizer(
            result_dir=args.result_dir,
            output_dir=args.output_dir
        )
        
        # Update configuration
        visualizer.resolutions = args.resolutions
        visualizer.models = args.models
        
        # Generate visualizations
        visualizer.generate_all_visualizations()
        
        # Create comparison plots if requested
        if args.comparison:
            for resolution in args.resolutions:
                try:
                    visualizer.create_comparison_plot(resolution)
                except Exception as e:
                    logger.error(f"Failed to create comparison plot for resolution {resolution}: {e}")
                    continue
                    
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        raise


if __name__ == "__main__":
    main()