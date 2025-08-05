#==============================================================================
# Unified Flood Modeling using VSURF Random Forest
#==============================================================================
#
# This script performs flood susceptibility modeling using VSURF (Variable 
# Selection Using Random Forests) with Random Forest classification for
# multiple DGGS resolutions (19, 21, 23).
#
# Features:
# - Variable selection using VSURF for multiple resolutions
# - Multiple model configurations (HG, HGPT, HG8M)
# - Performance evaluation with ROC analysis
# - Prediction on full study area
# - Comprehensive error handling and logging
# - Configurable parameters for different resolutions
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

# Global configuration
TRAIN_FRACTION <- 0.7
NTREES <- 500
NMIN <- 3
NCORES <- 8

# Resolution-specific configurations
RESOLUTION_CONFIGS <- list(
  "19" = list(
    data_file = "../Data/NB_sample_cent_quanti4_19.csv",
    prediction_file = "../Data/NB_finalPredict_19.csv",
    output_file = "../Result/NB_centPredict_19.csv",
    # Variable ranges for different models
    hg_vars = 2:21,      # 20 hydro-geomorphological variables
    hgpt_vars = 2:23,    # 20 HG + 2 climate variables
    hg8m_vars = 2:29,    # All variables
    # Column order for this resolution
    col_order = c("grid_code", "geo", "wl", "sol", "lc", "ndvi", "msi", "ia", "fcp", 
                  "dtm", "slp", "asp", "curv", "rgh", "tri", "tpi", "nhn", "fldir", "flacc", "twi", "spi",
                  "precip", "tavg", "r10", "r25", "tm10", "sd50", "ts", "spr"),
    var_order = c("geo", "wl", "sol", "lc", "ndvi", "msi", "ia", "fcp", 
                  "dtm", "slp", "asp", "curv", "rgh", "tri", "tpi", "nhn", "fldir", "flacc", "twi", "spi",
                  "precip", "tavg", "r10", "r25", "tm10", "sd50", "ts", "spr")
  ),
  "21" = list(
    data_file = "../Data/NB_sample_cent_quanti3_21.csv",
    prediction_file = "../Data/NB_finalPredict_21.csv",
    output_file = "../Result/NB_centPredict_21.csv",
    # Variable ranges for different models
    hg_vars = 2:17,      # 16 hydro-geomorphological variables
    hgpt_vars = 2:19,    # 16 HG + 2 climate variables
    hg8m_vars = 2:25,    # All variables
    # Column order for this resolution
    col_order = c("grid_code", "geo", "wl", "sol", "lc", "ndvi", "msi", "ia", "fcp", 
                  "dtm", "slp", "asp", "curv", "rgh", "tri", "tpi", "nhn",
                  "precip", "tavg", "r10", "r25", "tm10", "sd50", "ts", "spr"),
    var_order = c("geo", "wl", "sol", "lc", "ndvi", "msi", "ia", "fcp", 
                  "dtm", "slp", "asp", "curv", "rgh", "tri", "tpi", "nhn",
                  "precip", "tavg", "r10", "r25", "tm10", "sd50", "ts", "spr")
  ),
  "23" = list(
    data_file = "../Data/NB_sample_cent_quanti3_23.csv",
    prediction_file = "../Data/NB_finalPredict_23.csv",
    output_file = "../Result/NB_centPredict_23.csv",
    # Variable ranges for different models
    hg_vars = 2:17,      # 16 hydro-geomorphological variables
    hgpt_vars = 2:19,    # 16 HG + 2 climate variables
    hg8m_vars = 2:25,    # All variables
    # Column order for this resolution
    col_order = c("grid_code", "geo", "wl", "sol", "lc", "ndvi", "msi", "ia", "fcp", 
                  "dtm", "slp", "asp", "curv", "rgh", "tri", "tpi", "nhn",
                  "precip", "tavg", "r10", "r25", "tm10", "sd50", "ts", "spr"),
    var_order = c("geo", "wl", "sol", "lc", "ndvi", "msi", "ia", "fcp", 
                  "dtm", "slp", "asp", "curv", "rgh", "tri", "tpi", "nhn",
                  "precip", "tavg", "r10", "r25", "tm10", "sd50", "ts", "spr")
  )
)

#==============================================================================
# Utility Functions
#==============================================================================

#' Load and prepare data for a specific resolution
#' 
#' @param resolution Resolution level (19, 21, or 23)
#' @return List containing training and testing data
load_and_prepare_data <- function(resolution) {
  cat("Loading and preparing data for resolution", resolution, "...\n")
  
  # Get configuration for this resolution
  config <- RESOLUTION_CONFIGS[[as.character(resolution)]]
  if (is.null(config)) {
    stop(paste("Invalid resolution:", resolution))
  }
  
  # Check if data file exists
  if (!file.exists(config$data_file)) {
    stop(paste("Data file not found:", config$data_file))
  }
  
  # Read sample points
  spoints <- read.csv(config$data_file, header = TRUE, fileEncoding = "UTF-8-BOM")
  
  # Remove spatial coordinate columns
  spatial_cols <- c("i", "j", "Cell_address", "lon_c", "lat_c")
  spoints <- subset(spoints, select = -which(names(spoints) %in% spatial_cols))
  
  # Remove rows with no-data values (-32767)
  spoints <- spoints %>% filter_all(all_vars(!grepl(-32767, .)))
  
  # Remove rows with missing values
  spoints <- spoints[complete.cases(spoints), ]
  
  cat("Data shape after cleaning:", dim(spoints), "\n")
  
  # Convert categorical variables to factors
  categorical_cols <- c("grid_code", "geo", "wl", "sol", "lc", "ia", "fcp")
  spoints[categorical_cols] <- lapply(spoints[categorical_cols], factor)
  
  # Reorder columns
  spoints <- spoints[, config$col_order]
  
  # Split data into training and testing sets
  spoints.train <- spoints %>% sample_frac(TRAIN_FRACTION)
  spoints.test <- setdiff(spoints, spoints.train)
  
  cat("Training set size:", nrow(spoints.train), "\n")
  cat("Testing set size:", nrow(spoints.test), "\n")
  
  return(list(
    train = spoints.train,
    test = spoints.test,
    config = config
  ))
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

#' Map index to variable name
#' 
#' @param index_vec Vector of indices
#' @param var_order Vector of variable names
#' @return Vector of variable names
index_to_var <- function(index_vec, var_order) {
  var_name_vec <- c()
  for (i in index_vec) {
    var_name <- var_order[i]
    var_name_vec <- append(var_name_vec, var_name)
  }
  return(var_name_vec)
}

#' Print variable selection results
#' 
#' @param model Trained VSURF model
#' @param model_name Name of the model
#' @param var_order Vector of variable names
print_variable_selection <- function(model, model_name, var_order) {
  cat("\n---", model_name, "Variable Selection ---\n")
  
  # Interpretation step variables
  interp_vars <- index_to_var(model$varselect.interp, var_order)
  cat("Interpretation step variables (", length(interp_vars), "):\n")
  cat(paste(interp_vars, collapse = ", "), "\n")
  
  # Prediction step variables
  pred_vars <- index_to_var(model$varselect.pred, var_order)
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

#' Make predictions on full study area
#' 
#' @param models List of trained models
#' @param resolution Resolution level
#' @param config Configuration for this resolution
make_full_area_predictions <- function(models, resolution, config) {
  cat("Making predictions on full study area for resolution", resolution, "...\n")
  
  # Check if prediction file exists
  if (!file.exists(config$prediction_file)) {
    stop(paste("Prediction file not found:", config$prediction_file))
  }
  
  # Read all centroid points for full area prediction
  cent <- read.csv(config$prediction_file, header = TRUE, fileEncoding = "UTF-8-BOM")
  
  # Remove spatial coordinate columns
  spatial_cols <- c("i", "j", "Cell_address")
  cent <- subset(cent, select = -which(names(cent) %in% spatial_cols))
  
  # Remove rows with no-data values
  cent <- cent %>% filter_all(all_vars(!grepl(-32767, .)))
  cent <- cent[complete.cases(cent), ]
  
  cat("Prediction data shape:", dim(cent), "\n")
  
  # Convert categorical variables to factors
  categorical_cols <- c("geo", "wl", "sol", "lc", "ia", "fcp")
  cent[categorical_cols] <- lapply(cent[categorical_cols], factor)
  
  # Make predictions with each model
  tryCatch({
    cent$HG <- predict(models$hg, newdata = cent[, config$hg_vars], step = c("pred"))
    cent$HGPT <- predict(models$hgpt, newdata = cent[, config$hgpt_vars], step = c("pred"))
    cent$HG8M <- predict(models$hg8m, newdata = cent[, config$hg8m_vars], step = c("pred"))
    
    # Select relevant columns for output
    pred.result <- select(cent, c('lon_c', 'lat_c', 'HG', 'HGPT', 'HG8M'))
    
    # Save results
    write.csv(pred.result, config$output_file, row.names = FALSE)
    cat("Predictions saved to:", config$output_file, "\n")
    
  }, error = function(e) {
    cat("Error making predictions:", e$message, "\n")
    stop(e)
  })
}

#==============================================================================
# Main Analysis Function
#==============================================================================

#' Run complete analysis for a specific resolution
#' 
#' @param resolution Resolution level (19, 21, or 23)
run_analysis <- function(resolution) {
  cat("=" * 60, "\n")
  cat("Starting analysis for resolution", resolution, "\n")
  cat("=" * 60, "\n")
  
  tryCatch({
    # Load and prepare data
    data_list <- load_and_prepare_data(resolution)
    train_data <- data_list$train
    test_data <- data_list$test
    config <- data_list$config
    
    # Train models
    cat("\nTraining VSURF models...\n")
    
    # Model 1: Hydro-Geomorphological variables
    cat("\n=== Model 1: Hydro-Geomorphological (HG) ===\n")
    hg_model <- train_vsurf_model(
      x = train_data[, config$hg_vars], 
      y = train_data[, 1], 
      model_name = "HG", 
      n_vars = length(config$hg_vars)
    )
    
    # Model 2: HG + Precipitation and Temperature
    cat("\n=== Model 2: HG + Precipitation/Temperature (HGPT) ===\n")
    hgpt_model <- train_vsurf_model(
      x = train_data[, config$hgpt_vars], 
      y = train_data[, 1], 
      model_name = "HGPT", 
      n_vars = length(config$hgpt_vars)
    )
    
    # Model 3: All variables
    cat("\n=== Model 3: All variables (HG8M) ===\n")
    hg8m_model <- train_vsurf_model(
      x = train_data[, config$hg8m_vars], 
      y = train_data[, 1], 
      model_name = "HG8M", 
      n_vars = length(config$hg8m_vars)
    )
    
    # Store models
    models <- list(hg = hg_model, hgpt = hgpt_model, hg8m = hg8m_model)
    
    # Print variable selection results
    cat("\nAnalyzing variable selection results...\n")
    print_variable_selection(hg_model, "HG", config$var_order)
    print_variable_selection(hgpt_model, "HGPT", config$var_order)
    print_variable_selection(hg8m_model, "HG8M", config$var_order)
    
    # Evaluate model performance
    cat("\nEvaluating model performance...\n")
    hg_results <- evaluate_model(
      hg_model, 
      test_data, 
      test_data[, config$hg_vars], 
      "HG"
    )
    
    hgpt_results <- evaluate_model(
      hgpt_model, 
      test_data, 
      test_data[, config$hgpt_vars], 
      "HGPT"
    )
    
    hg8m_results <- evaluate_model(
      hg8m_model, 
      test_data, 
      test_data[, config$hg8m_vars], 
      "HG8M"
    )
    
    # Make predictions on full study area
    make_full_area_predictions(models, resolution, config)
    
    # Save workspace
    save.image(file = paste0("test_nb_vsurf_pred_", resolution, ".RData"))
    
    cat("\nAnalysis completed successfully for resolution", resolution, "\n")
    
  }, error = function(e) {
    cat("Error in analysis for resolution", resolution, ":", e$message, "\n")
    stop(e)
  })
}

#==============================================================================
# Main Execution
#==============================================================================

#' Main function to run analysis for specified resolutions
#' 
#' @param resolutions Vector of resolutions to analyze (default: c(19, 21, 23))
main <- function(resolutions = c(19, 21, 23)) {
  cat("Starting unified flood modeling analysis\n")
  cat("Resolutions to process:", paste(resolutions, collapse = ", "), "\n")
  cat("Configuration:\n")
  cat("- Training fraction:", TRAIN_FRACTION, "\n")
  cat("- Number of trees:", NTREES, "\n")
  cat("- Minimum variables:", NMIN, "\n")
  cat("- Number of cores:", NCORES, "\n")
  
  # Validate resolutions
  valid_resolutions <- names(RESOLUTION_CONFIGS)
  invalid_resolutions <- setdiff(as.character(resolutions), valid_resolutions)
  if (length(invalid_resolutions) > 0) {
    stop(paste("Invalid resolutions:", paste(invalid_resolutions, collapse = ", ")))
  }
  
  # Run analysis for each resolution
  for (resolution in resolutions) {
    tryCatch({
      run_analysis(resolution)
    }, error = function(e) {
      cat("Failed to complete analysis for resolution", resolution, ":", e$message, "\n")
    })
  }
  
  cat("\nAll analyses completed\n")
}

#==============================================================================
# Example Usage
#==============================================================================

# Run analysis for all resolutions
# main()

# Run analysis for specific resolution
# main(c(23))

# Run analysis for multiple resolutions
# main(c(19, 23))

#==============================================================================
# Run the analysis
#==============================================================================

# Uncomment the line below to run the analysis
# main() 