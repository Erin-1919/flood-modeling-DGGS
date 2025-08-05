# Multi-Scale Flood Mapping under Climate Change Scenarios in Hexagonal Discrete Global Grids

A comprehensive flood modeling framework that utilizes Discrete Global Grid System (DGGS) for multi-scale spatial analysis and machine learning-based flood prediction under climate change scenarios.

## Overview

This repository implements the methodology described in the paper "Multi-Scale Flood Mapping under Climate Change Scenarios in Hexagonal Discrete Global Grids" (DOI: 10.3390/ijgi11120627). The framework combines:

- **DGGS-based spatial quantization** for consistent global grid representation across multiple scales
- **Multi-scale topographic parameter computation** including slope, aspect, curvature, roughness, TRI, and TPI
- **Hydrological analysis** for flow accumulation and drainage patterns at various resolutions
- **Machine learning models** using VSURF and Random Forest for flood susceptibility prediction
- **Climate change scenario integration** for future flood risk assessment
- **Visualization tools** for multi-scale results interpretation

## Features

### Core Functionality
- **Multi-Scale Spatial Quantization**: Convert raster data to DGGS cells at multiple resolutions (16-29) for scale-dependent analysis
- **Topographic Analysis**: Compute slope, aspect, curvature, roughness, TRI, and TPI across different scales
- **Hydrological Processing**: Flow accumulation, drainage direction, and stream network analysis at various resolutions
- **Machine Learning**: VSURF-based Random Forest models for flood prediction with variable selection
- **Climate Change Integration**: Support for different climate scenarios and future projections
- **Parallel Processing**: Multi-core computation for large datasets
- **Visualization**: Interactive maps and statistical plots for multi-scale results

### Advanced Machine Learning Features
- **Unified Multi-Resolution Modeling**: Single script (`ML_flood_vsurf_unified.R`) handles multiple resolutions
- **Combined Sub-basin Analysis**: Comprehensive modeling across multiple sub-basins (`ML_flood_sub_comb.R`)
- **Variable Selection**: Automated feature selection using VSURF for optimal model performance
- **Performance Evaluation**: Comprehensive metrics including Accuracy, AUC, and F-Score
- **Error Handling**: Robust error handling and graceful degradation for large datasets

### Supported Resolutions
- DGGS resolutions 16-29 with corresponding cell sizes from 5km to 5m
- Automatic vertical resolution adjustment based on grid level
- Configurable cell spacing and area calculations

## Installation

### Prerequisites
- Python 3.8+
- R 4.0+ (for machine learning components)
- GDAL/OGR libraries

### Python Dependencies
```bash
pip install -r requirements.txt
```

### R Dependencies
```r
install.packages(c("VSURF", "ROCR", "dplyr"))
```

## Project Structure

```
flood-modeling-DGGS/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── setup.py                 # Package setup
├── config/                  # Configuration files
├── data/                    # Input data directory
├── results/                 # Output results
├── scripts/                 # Main execution scripts
│   ├── Quantization_*.py   # Spatial quantization scripts
│   ├── ML_flood_*.R        # Machine learning scripts
│   └── Visualization_*.py  # Visualization scripts
├── src/                     # Source code modules
│   ├── dg_base/            # DGGS base functions
│   │   └── DgBaseFunc.py   # Core DGGS operations
│   ├── dg_topog/           # Topographic functions
│   │   └── DgTopogFunc.py  # Topographic parameter computation
│   ├── dg_hydro/           # Hydrological functions
│   │   └── DgHydroFunc.py  # Hydrological analysis
│   └── geo_base/           # Geographic base functions
│       └── GeoBaseFunc.py  # Geographic operations
├── notebooks/               # Jupyter notebooks
└── tests/                  # Unit tests
```

## Usage

### 1. Data Preparation
Place your input data in the `data/` directory:
- Digital Terrain Model (DTM) in GeoTIFF format
- Study area boundary shapefile
- Training points with flood occurrence data

### 2. Spatial Quantization
```bash
# Elevation quantization
python scripts/Quantization_elev.py --resolution 23 --input data/dtm.tif --output results/elevation_23.csv

# Topographic parameter computation
python scripts/Quantization_topog.py --resolution 23 --input data/dtm.tif --output results/topography_23.csv

# Hydrological analysis
python scripts/Quantization_hydro.py --resolution 23 --input data/dtm.tif --output results/hydrology_23.csv

# Climate and environmental variables
python scripts/Quantization_vari.py --resolution 23 --input data/climate_data.csv --output results/variables_23.csv

# Distance to hydrological network
python scripts/Quantization_dist.py --resolution 23 --input data/nhn_data.csv --output results/distance_23.csv

# Complete prediction pipeline
python scripts/Quantization_predict.py --resolution 23 --input data/ --output results/prediction_23.csv
```

### 3. Machine Learning

#### Individual Resolution Analysis
```r
# For resolution 19
Rscript scripts/ML_flood_vsurf19.R

# For resolution 21
Rscript scripts/ML_flood_vsurf21.R

# For resolution 23
Rscript scripts/ML_flood_vsurf23.R
```

#### Unified Multi-Resolution Analysis
```r
# Analyze multiple resolutions in a single run
Rscript scripts/ML_flood_vsurf_unified.R
```

#### Combined Sub-basin Analysis
```r
# Comprehensive analysis across multiple sub-basins
Rscript scripts/ML_flood_sub_comb.R
```

### 4. Visualization
```bash
python scripts/Visualization_pred.py --input results/predictions.csv --output results/maps/
```

## Configuration

### Resolution Settings
- **Low Resolution (16-18)**: Regional analysis, coarse detail
- **Medium Resolution (19-22)**: Watershed analysis, moderate detail
- **High Resolution (23-29)**: Local analysis, fine detail

### Performance Optimization
- Adjust `n_cores` parameter for parallel processing
- Use appropriate resolution based on study area size
- Consider memory requirements for high-resolution analysis

## API Documentation

### Core Functions

#### DgBaseFunc
- `look_up_table()`: DGGS resolution lookup table
- `check_if_duplicates()`: Check for duplicate coordinates
- `common_member()`: Find common elements between arrays
- `catch()`: Error handling utility

#### DgTopogFunc
- `slope_aspect_df()`: Compute slope and aspect using multiple methods
- `curvature_df()`: Calculate curvature parameters
- `roughness_df()`: Estimate surface roughness and TRI
- `tpi_df()`: Calculate Topographic Position Index

#### DgHydroFunc
- `flow_direction_df()`: Calculate flow direction
- `flow_accumulation_df()`: Calculate flow accumulation
- `fill_depressions()`: Fill depressions in DEM
- `hydrological_indices()`: Calculate TWI and SPI

#### GeoBaseFunc
- `reproject_coords()`: Reproject coordinates between systems
- `find_neighbor()`: Find neighboring cells
- `bilinear_interp()`: Bilinear interpolation
- `IDW_interp()`: Inverse Distance Weighted interpolation

### Machine Learning Scripts

#### ML_flood_vsurf_unified.R
- **Purpose**: Unified analysis across multiple resolutions
- **Features**: Configurable resolution list, modular functions, comprehensive error handling
- **Output**: Variable selection results and performance metrics for each resolution

#### ML_flood_sub_comb.R
- **Purpose**: Combined analysis across multiple sub-basins
- **Features**: Robust data loading, error recovery, comprehensive reporting
- **Models**: HG (20 variables), HGPT (22 variables), HG8M (28 variables)

## Examples

### Basic Quantization
```python
from src.dg_base.DgBaseFunc import look_up_table
from scripts.Quantization_elev import ElevationQuantizer

# Initialize quantizer
quantizer = ElevationQuantizer(resolution=23)

# Quantize elevation data
quantizer.load_data('data/dtm.tif')
quantizer.filter_data()
quantizer.interpolate_elevation()
quantizer.save_results('results/elevation_23.csv')
```

### Machine Learning Pipeline
```r
# Load and prepare data
data <- read.csv("results/topography_23.csv")

# Train VSURF model
model <- VSURF(x = data[,2:21], y = data[,1], 
               ntree = 500, ncores = 8)

# Print variable selection results
print_variable_selection(model, "Model Name")

# Evaluate performance
results <- evaluate_model(model, test_data, test_features, "Model Name")
```

### Combined Sub-basin Analysis
```r
# The script automatically handles multiple sub-basins
# Configuration is at the top of ML_flood_sub_comb.R
SUB_BASIN_IDS <- c(1, 2, 3, 4, 5, 8, 9, 10, 11, 13, 14, 16)
RESOLUTION <- 19

# Run the analysis
source("scripts/ML_flood_sub_comb.R")
```

## Testing

Run the test suite:
```bash
python -m pytest tests/
```

## Citation

If you use this code in your research, please cite the original paper:

```bibtex
@article{flood_modeling_dggs_2022,
  title={Multi-Scale Flood Mapping under Climate Change Scenarios in Hexagonal Discrete Global Grids},
  author={Li, M.; McGrath, H.; Stefanakis, E.},
  journal={ISPRS International Journal of Geo-Information},
  volume={11},
  number={12},
  pages={627},
  year={2022},
  publisher={MDPI},
  doi={10.3390/ijgi11120627}
}
```

**Paper Abstract**: Among the most prevalent natural hazards, flooding has been threatening human lives and properties. Robust flood simulation is required for effective response and prevention. Machine learning is widely used in flood modeling due to its high performance and scalability. Nonetheless, data pre-processing of heterogeneous sources can be cumbersome, and traditional data processing and modeling have been limited to a single resolution. This study employed an Icosahedral Snyder Equal Area Aperture 3 Hexagonal Discrete Global Grid System (ISEA3H DGGS) as a scalable, standard spatial framework for computation, integration, and analysis of multi-source geospatial data. We managed to incorporate external machine learning algorithms with a DGGS-based data framework, and project future flood risks under multiple climate change scenarios for southern New Brunswick, Canada. A total of 32 explanatory factors including topographical, hydrological, geomorphic, meteorological, and anthropogenic were investigated. Results showed that low elevation and proximity to permanent waterbodies were primary factors of flooding events, and rising spring temperatures can increase flood risk. Flooding extent was predicted to occupy 135–203% of the 2019 flood area, one of the most recent major flooding events, by the year 2100. Our results assisted in understanding the potential impact of climate change on flood risk, and indicated the feasibility of DGGS as the standard data fabric for heterogeneous data integration and incorporated in multi-scale data mining.


