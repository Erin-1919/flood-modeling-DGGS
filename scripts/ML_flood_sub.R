#==============================================================================
# Sub-basin Flood Modeling using VSURF Random Forest
#==============================================================================
#
# This script performs flood susceptibility modeling using VSURF (Variable 
# Selection Using Random Forests) with Random Forest classification for
# different sub-basins and resolution levels.
#
# Features:
# - Variable selection using VSURF for different model configurations
# - Multiple sub-basin analysis (1,2,8,9,10,11,13,14)
# - Multiple resolution analysis (19,21,23)
# - Three model configurations: HG, HGPT, HG8M
# - Comprehensive variable mapping and selection reporting
#
# Author: Erin Li
#==============================================================================

# Load required libraries
library(VSURF)
library(ROCR)
library(dplyr)

# Set random seed for reproducibility
set.seed(500)

# Configuration
RESOLUTIONS <- c(19, 21, 23)
SUB_BASIN_IDS <- c(1, 2, 8, 9, 10, 11, 13, 14)

# Variable order definitions
COL_ORDER <- c("grid_code", "geo", "wl", "sol", "lc", "ndvi", "msi", "ia", "fcp", 
               "dtm", "slp", "asp", "curv", "rgh", "tri", "tpi", "nhn", "fldir", 
               "flacc", "twi", "spi", "precip", "tavg", "r10", "r25", "tm10", 
               "sd50", "ts", "spr")

VAR_ORDER <- c("geo", "wl", "sol", "lc", "ndvi", "msi", "ia", "fcp", "dtm", 
               "slp", "asp", "curv", "rgh", "tri", "tpi", "nhn", "fldir", 
               "flacc", "twi", "spi", "precip", "tavg", "r10", "r25", "tm10", 
               "sd50", "ts", "spr")

# Model configuration definitions
MODEL_CONFIGS <- list(
  HG = list(
    name = "Hydro-Geomorphological",
    var_indices = 2:21,
    description = "20 hydro-geomorphological variables"
  ),
  HGPT = list(
    name = "Hydro-Geomorphological + Precipitation/Temperature",
    var_indices = 2:23,
    description = "20 hydro-geomorphological variables + average annual Precipitation and Temperature"
  ),
  HG8M = list(
    name = "All Variables",
    var_indices = 2:29,
    description = "All variables (28 variables)"
  )
)


#' Map index to variable name
#' 
#' Converts numeric indices to corresponding variable names based on VAR_ORDER
#' 
#' @param index_vec Vector of numeric indices
#' @return Vector of variable names
#' @examples
#' index_to_var(c(1, 2, 3))
index_to_var <- function(index_vec) {
  if (!is.numeric(index_vec)) {
    stop("index_vec must be numeric")
  }
  
  if (any(index_vec < 1 | index_vec > length(VAR_ORDER))) {
    stop("Index values must be between 1 and ", length(VAR_ORDER))
  }
  
  var_name_vec <- VAR_ORDER[index_vec]
  return(var_name_vec)
}


#' Load and prepare sample point data
#' 
#' Loads sample point data for a specific sub-basin and resolution,
#' performs data cleaning and preparation
#' 
#' @param sub_id Sub-basin ID
#' @param dggs_res DGGS resolution level
#' @param data_dir Directory containing data files
#' @return Prepared data frame
#' @examples
#' spoints <- load_sample_data(1, 23, "../Data/sub_basin")
load_sample_data <- function(sub_id, dggs_res, data_dir = "../Data/sub_basin") {
  tryCatch({
    # Construct file path
    file_path <- file.path(data_dir, sprintf("NB_sample_cent_quanti4_%d_%d.csv", sub_id, dggs_res))
    
    # Check if file exists
    if (!file.exists(file_path)) {
      stop(sprintf("Sample data file not found: %s", file_path))
    }
    
    # Read sample points
    spoints <- read.csv(file_path, header = TRUE, fileEncoding = "UTF-8-BOM")
    
    # Remove unnecessary columns
    spoints <- subset(spoints, select = -c(i, j, Cell_address, lon_c, lat_c))
    
    # Filter out invalid data
    spoints <- filter(spoints, geo != -32767 & dtm != -32767)
    spoints <- spoints[complete.cases(spoints), ]
    
    # Convert categorical variables to factors
    categorical_cols <- c("grid_code", "geo", "wl", "sol", "lc", "ia", "fcp")
    spoints[categorical_cols] <- lapply(spoints[categorical_cols], factor)
    
    # Reorder columns
    spoints <- spoints[, COL_ORDER]
    
    cat(sprintf("Loaded %d sample points for sub-basin %d at resolution %d\n", 
                nrow(spoints), sub_id, dggs_res))
    
    return(spoints)
    
  }, error = function(e) {
    stop(sprintf("Failed to load sample data for sub-basin %d, resolution %d: %s", 
                 sub_id, dggs_res, e$message))
  })
}


#' Train VSURF model for a specific configuration
#' 
#' Trains a VSURF model with the specified variable configuration
#' 
#' @param data Sample point data
#' @param config_name Model configuration name (HG, HGPT, HG8M)
#' @param nmin Minimum number of variables for interpretation step
#' @param ntree Number of trees for Random Forest
#' @return Trained VSURF model
#' @examples
#' model <- train_vsurf_model(spoints, "HG", nmin = 3, ntree = 500)
train_vsurf_model <- function(data, config_name, nmin = 3, ntree = 500) {
  tryCatch({
    if (!config_name %in% names(MODEL_CONFIGS)) {
      stop(sprintf("Invalid configuration name: %s", config_name))
    }
    
    config <- MODEL_CONFIGS[[config_name]]
    var_indices <- config$var_indices
    
    cat(sprintf("Training %s model (%s)\n", config$name, config$description))
    cat(sprintf("Using variables %d to %d\n", min(var_indices), max(var_indices)))
    
    # Train VSURF model
    model <- VSURF(
      x = data[, var_indices], 
      y = data[, 1], 
      nmin = nmin, 
      ntree = ntree, 
      na.action = na.omit
    )
    
    return(model)
    
  }, error = function(e) {
    stop(sprintf("Failed to train %s model: %s", config_name, e$message))
  })
}


#' Print variable selection results
#' 
#' Prints the selected variables for interpretation and prediction steps
#' 
#' @param model Trained VSURF model
#' @param config_name Model configuration name
#' @examples
#' print_variable_selection(model, "HG")
print_variable_selection <- function(model, config_name) {
  cat(sprintf("\n=== Variable Selection Results for %s ===\n", config_name))
  
  # Interpretation step variables
  interp_vars <- index_to_var(model$varselect.interp)
  cat(sprintf("Interpretation step variables (%d): %s\n", 
              length(interp_vars), paste(interp_vars, collapse = ", ")))
  
  # Prediction step variables
  pred_vars <- index_to_var(model$varselect.pred)
  cat(sprintf("Prediction step variables (%d): %s\n", 
              length(pred_vars), paste(pred_vars, collapse = ", ")))
  
  cat("=" * 50, "\n")
}


#' Run analysis for a specific sub-basin and resolution
#' 
#' Performs complete analysis including data loading, model training,
#' and variable selection for all model configurations
#' 
#' @param sub_id Sub-basin ID
#' @param dggs_res DGGS resolution level
#' @examples
#' run_analysis(1, 23)
run_analysis <- function(sub_id, dggs_res) {
  cat(sprintf("\n" + "=" * 60 + "\n"))
  cat(sprintf("Starting analysis for Sub-basin ID: %d, Resolution: %d\n", sub_id, dggs_res))
  cat("=" * 60 + "\n")
  
  tryCatch({
    # Load and prepare data
    spoints <- load_sample_data(sub_id, dggs_res)
    
    # Train models for each configuration
    models <- list()
    
    for (config_name in names(MODEL_CONFIGS)) {
      cat(sprintf("\n--- Training %s model ---\n", config_name))
      
      # Train model
      model <- train_vsurf_model(spoints, config_name)
      models[[config_name]] <- model
      
      # Print variable selection results
      print_variable_selection(model, config_name)
    }
    
    cat(sprintf("\nAnalysis completed for Sub-basin %d, Resolution %d\n", sub_id, dggs_res))
    
    return(models)
    
  }, error = function(e) {
    cat(sprintf("Error in analysis for Sub-basin %d, Resolution %d: %s\n", 
                sub_id, dggs_res, e$message))
    return(NULL)
  })
}


#' Main analysis function
#' 
#' Runs the complete analysis across all sub-basins and resolutions
#' 
#' @examples
#' main_analysis()
main_analysis <- function() {
  cat("Starting sub-basin flood modeling analysis\n")
  cat(sprintf("Resolutions: %s\n", paste(RESOLUTIONS, collapse = ", ")))
  cat(sprintf("Sub-basins: %s\n", paste(SUB_BASIN_IDS, collapse = ", ")))
  cat(sprintf("Model configurations: %s\n", paste(names(MODEL_CONFIGS), collapse = ", ")))
  
  # Store results
  all_results <- list()
  
  # Run analysis for each resolution and sub-basin
  for (dggs_res in RESOLUTIONS) {
    for (sub_id in SUB_BASIN_IDS) {
      results <- run_analysis(sub_id, dggs_res)
      
      if (!is.null(results)) {
        all_results[[sprintf("sub_%d_res_%d", sub_id, dggs_res)]] <- results
      }
    }
  }
  
  cat("\n" + "=" * 60 + "\n")
  cat("Analysis completed for all sub-basins and resolutions\n")
  cat("=" * 60 + "\n")
  
  return(all_results)
}


#==============================================================================
# Example usage and evaluation functions (commented out)
#==============================================================================

# # Example: Evaluating and visualizing the performance
# evaluate_model_performance <- function(model, data, var_indices, model_name) {
#   # Make predictions
#   predictions <- predict(model, newdata = data[, var_indices], step = c("interp", "pred"))
#   
#   # Calculate AUC
#   pred_obj <- ROCR::prediction(as.numeric(predictions$pred), as.numeric(data$grid_code))
#   auc_value <- performance(pred_obj, "auc")@y.values[[1]]
#   
#   cat(sprintf("%s model AUC: %.4f\n", model_name, auc_value))
#   
#   return(auc_value)
# }

# # Example usage:
# # results <- main_analysis()
# # 
# # # Evaluate performance for a specific model
# # spoints <- load_sample_data(1, 23)
# # model <- train_vsurf_model(spoints, "HG")
# # auc <- evaluate_model_performance(model, spoints, 2:21, "HG")


#==============================================================================
# Run main analysis
#==============================================================================

# Run the complete analysis
results <- main_analysis()
