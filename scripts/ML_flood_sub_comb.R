#==============================================================================
# Combined Sub-basin Flood Modeling using VSURF Random Forest
#==============================================================================
#
# This script performs flood susceptibility modeling using VSURF (Variable 
# Selection Using Random Forests) with Random Forest classification by combining
# data from multiple sub-basins.
#
# Features:
# - Combines data from multiple sub-basins for comprehensive modeling
# - Variable selection using VSURF
# - Multiple model configurations (HG, HGPT, HG8M)
# - Performance evaluation with ROC analysis
# - Comprehensive error handling and logging
#
# Author: Erin Li
#==============================================================================

# Load required libraries
library(VSURF)
library(ROCR)
library(dplyr)

# Set random seed for reproducibility
set.seed(500)

#==============================================================================
# Configuration
#==============================================================================

# Model configuration
RESOLUTION <- 19
SUB_BASIN_IDS <- c(1, 2, 3, 4, 5, 8, 9, 10, 11, 13, 14, 16)
TRAIN_FRACTION <- 0.7
NTREES <- 500
NMIN <- 3
NCORES <- 4

# Variable order definitions
COL_ORDER <- c("grid_code", "geo", "wl", "sol", "lc", "ndvi", "msi", "ia", "fcp", 
               "dtm", "slp", "asp", "curv", "rgh", "tri", "tpi", "nhn", "fldir", "flacc", "twi", "spi",
               "precip", "tavg", "r10", "r25", "tm10", "sd50", "ts", "spr")

VAR_ORDER <- c("geo", "wl", "sol", "lc", "ndvi", "msi", "ia", "fcp", 
               "dtm", "slp", "asp", "curv", "rgh", "tri", "tpi", "nhn", "fldir", "flacc", "twi", "spi",
               "precip", "tavg", "r10", "r25", "tm10", "sd50", "ts", "spr")

#==============================================================================
# Utility Functions
#==============================================================================

#' Map index to variable name
#' 
#' @param index_vec Vector of indices
#' @return Vector of variable names
index_to_var <- function(index_vec) {
  var_name_vec <- c()
  for (i in index_vec) {
    var_name <- VAR_ORDER[i]
    var_name_vec <- append(var_name_vec, var_name)
  }
  return(var_name_vec)
}

#' Load and prepare data for a specific sub-basin
#' 
#' @param sub_id Sub-basin ID
#' @param resolution DGGS resolution level
#' @return Prepared data frame
load_sub_basin_data <- function(sub_id, resolution) {
  cat("Loading data for sub-basin", sub_id, "at resolution", resolution, "...\n")
  
  # Construct file path
  file_path <- sprintf("../Data/sub_basin/NB_sample_cent_quanti4_%d_%d.csv", sub_id, resolution)
  
  # Check if file exists
  if (!file.exists(file_path)) {
    stop(sprintf("Data file not found: %s", file_path))
  }
  
  tryCatch({
    # Read sample points
    spoints <- read.csv(file_path, header = TRUE, fileEncoding = "UTF-8-BOM")
    
    # Remove spatial coordinate columns
    spatial_cols <- c("i", "j", "Cell_address", "lon_c", "lat_c")
    spoints <- subset(spoints, select = -which(names(spoints) %in% spatial_cols))
    
    # Remove rows with no-data values
    spoints <- filter(spoints, geo != -32767 & dtm != -32767)
    spoints <- spoints[complete.cases(spoints), ]
    
    # Convert categorical variables to factors
    categorical_cols <- c("grid_code", "geo", "wl", "sol", "lc", "ia", "fcp")
    spoints[categorical_cols] <- lapply(spoints[categorical_cols], factor)
    
    # Reorder columns
    spoints <- spoints[, COL_ORDER]
    
    cat("Loaded", nrow(spoints), "records for sub-basin", sub_id, "\n")
    return(spoints)
    
  }, error = function(e) {
    stop(sprintf("Error loading data for sub-basin %d: %s", sub_id, e$message))
  })
}

#' Combine data from multiple sub-basins
#' 
#' @param sub_basin_ids Vector of sub-basin IDs
#' @param resolution DGGS resolution level
#' @return Combined data frame
combine_sub_basin_data <- function(sub_basin_ids, resolution) {
  cat("Combining data from", length(sub_basin_ids), "sub-basins...\n")
  
  # Initialize empty data frame with correct column structure
  combined_data <- setNames(data.frame(matrix(ncol = length(COL_ORDER), nrow = 0)), COL_ORDER)
  
  total_records <- 0
  
  for (sub_id in sub_basin_ids) {
    tryCatch({
      # Load data for this sub-basin
      sub_data <- load_sub_basin_data(sub_id, resolution)
      
      # Combine with existing data
      combined_data <- rbind(combined_data, sub_data)
      total_records <- total_records + nrow(sub_data)
      
      cat("Added sub-basin", sub_id, "with", nrow(sub_data), "records\n")
      
    }, error = function(e) {
      cat("Warning: Failed to load sub-basin", sub_id, ":", e$message, "\n")
      cat("Continuing with remaining sub-basins...\n")
    })
  }
  
  cat("Combined data contains", nrow(combined_data), "total records from", 
      length(sub_basin_ids), "sub-basins\n")
  
  return(combined_data)
}

#' Train VSURF model
#' 
#' @param x_data Training features
#' @param y_data Training labels
#' @param model_name Name of the model
#' @param n_vars Number of variables
#' @return Trained VSURF model
train_vsurf_model <- function(x_data, y_data, model_name, n_vars) {
  cat("Training", model_name, "model with", n_vars, "variables...\n")
  
  tryCatch({
    model <- VSURF(
      x = x_data, 
      y = y_data, 
      nmin = NMIN, 
      ntree = NTREES,
      parallel = TRUE, 
      ncores = NCORES, 
      clusterType = "PSOCK", 
      na.action = na.omit
    )
    
    cat("Model training completed for", model_name, "\n")
    cat("Mean performance:", model$mean.perf, "\n")
    
    return(model)
  }, error = function(e) {
    cat("Error training", model_name, "model:", e$message, "\n")
    stop(e)
  })
}

#' Print variable selection results
#' 
#' @param model Trained VSURF model
#' @param model_name Name of the model
print_variable_selection <- function(model, model_name) {
  cat("\n---", model_name, "Variable Selection ---\n")
  
  # Interpretation step variables
  interp_vars <- index_to_var(model$varselect.interp)
  cat("Interpretation step variables (", length(interp_vars), "):\n")
  cat(paste(interp_vars, collapse = ", "), "\n")
  
  # Prediction step variables
  pred_vars <- index_to_var(model$varselect.pred)
  cat("Prediction step variables (", length(pred_vars), "):\n")
  cat(paste(pred_vars, collapse = ", "), "\n")
}

#' Evaluate model performance
#' 
#' @param model Trained VSURF model
#' @param test_data Test data
#' @param test_features Test features
#' @param model_name Name of the model
#' @return List of performance metrics
evaluate_model <- function(model, test_data, test_features, model_name) {
  cat("\n---", model_name, "Performance Evaluation ---\n")
  
  tryCatch({
    # Make predictions
    predictions <- predict(model, newdata = test_features, step = c("interp", "pred"))
    
    # Create prediction object for ROC analysis
    pred_obj <- ROCR::prediction(
      as.numeric(predictions$pred), 
      as.numeric(test_data$grid_code)
    )
    
    # Calculate performance metrics
    accuracy <- performance(pred_obj, "acc")@y.values[[1]][2]
    auc <- performance(pred_obj, "auc")@y.values[[1]]
    f_score <- performance(pred_obj, "f")@y.values[[1]][2]
    
    # Print results
    cat("Accuracy:", round(accuracy, 4), "\n")
    cat("AUC:", round(auc, 4), "\n")
    cat("F-Score:", round(f_score, 4), "\n")
    
    return(list(
      predictions = predictions,
      accuracy = accuracy,
      auc = auc,
      f_score = f_score
    ))
  }, error = function(e) {
    cat("Error evaluating", model_name, "model:", e$message, "\n")
    return(NULL)
  })
}

#==============================================================================
# Main Analysis
#==============================================================================

cat("Starting combined sub-basin flood modeling analysis\n")
cat("Resolution:", RESOLUTION, "\n")
cat("Sub-basins:", paste(SUB_BASIN_IDS, collapse = ", "), "\n")
cat("Configuration:\n")
cat("- Training fraction:", TRAIN_FRACTION, "\n")
cat("- Number of trees:", NTREES, "\n")
cat("- Minimum variables:", NMIN, "\n")
cat("- Number of cores:", NCORES, "\n")

#==============================================================================
# Data Preparation
#==============================================================================

cat("\n" + "=" * 60 + "\n")
cat("Data Preparation Phase\n")
cat("=" * 60 + "\n")

# Combine data from all sub-basins
CombPoints <- combine_sub_basin_data(SUB_BASIN_IDS, RESOLUTION)

# Clean up individual sub-basin data
rm(list = ls(pattern = "^spoints"))

# Split data into training and testing sets
cat("\nSplitting data into training and testing sets...\n")
CombPoints.train <- CombPoints %>% sample_frac(TRAIN_FRACTION)
CombPoints.test <- setdiff(CombPoints, CombPoints.train)

cat("Training set size:", nrow(CombPoints.train), "\n")
cat("Testing set size:", nrow(CombPoints.test), "\n")

#==============================================================================
# Model Training
#==============================================================================

cat("\n" + "=" * 60 + "\n")
cat("Model Training Phase\n")
cat("=" * 60 + "\n")

# Model 1: Hydro-Geomorphological variables (20 variables)
cat("\n=== Model 1: Hydro-Geomorphological (HG) ===\n")
vnb.HG <- train_vsurf_model(
  x = CombPoints.train[, 2:21], 
  y = CombPoints.train[, 1], 
  model_name = "HG", 
  n_vars = 20
)

# Model 2: HG + Precipitation and Temperature (22 variables)
cat("\n=== Model 2: HG + Precipitation/Temperature (HGPT) ===\n")
vnb.HGPT <- train_vsurf_model(
  x = CombPoints.train[, 2:23], 
  y = CombPoints.train[, 1], 
  model_name = "HGPT", 
  n_vars = 22
)

# Model 3: All variables (28 variables)
cat("\n=== Model 3: All variables (HG8M) ===\n")
vnb.HG8M <- train_vsurf_model(
  x = CombPoints.train[, 2:29], 
  y = CombPoints.train[, 1], 
  model_name = "HG8M", 
  n_vars = 28
)

#==============================================================================
# Variable Selection Results
#==============================================================================

cat("\n" + "=" * 60 + "\n")
cat("Variable Selection Analysis\n")
cat("=" * 60 + "\n")

# Print variable selection results for all models
print_variable_selection(vnb.HG, "HG")
print_variable_selection(vnb.HGPT, "HGPT")
print_variable_selection(vnb.HG8M, "HG8M")

#==============================================================================
# Model Performance Evaluation
#==============================================================================

cat("\n" + "=" * 60 + "\n")
cat("Model Performance Evaluation\n")
cat("=" * 60 + "\n")

# Evaluate all models
hg_results <- evaluate_model(
  vnb.HG, 
  CombPoints.test, 
  CombPoints.test[, 2:21], 
  "HG"
)

hgpt_results <- evaluate_model(
  vnb.HGPT, 
  CombPoints.test, 
  CombPoints.test[, 2:23], 
  "HGPT"
)

hg8m_results <- evaluate_model(
  vnb.HG8M, 
  CombPoints.test, 
  CombPoints.test[, 2:29], 
  "HG8M"
)

#==============================================================================
# Summary and Cleanup
#==============================================================================

cat("\n" + "=" * 60 + "\n")
cat("Analysis Summary\n")
cat("=" * 60 + "\n")

cat("Combined sub-basin analysis completed successfully\n")
cat("Total records processed:", nrow(CombPoints), "\n")
cat("Training records:", nrow(CombPoints.train), "\n")
cat("Testing records:", nrow(CombPoints.test), "\n")
cat("Sub-basins included:", paste(SUB_BASIN_IDS, collapse = ", "), "\n")
cat("Resolution:", RESOLUTION, "\n")

# Save workspace
save.image(file = sprintf("combined_sub_basin_analysis_%d.RData", RESOLUTION))
cat("Workspace saved to:", sprintf("combined_sub_basin_analysis_%d.RData", RESOLUTION), "\n")

cat("\nAnalysis completed successfully!\n")
