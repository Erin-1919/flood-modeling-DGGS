#==============================================================================
# Flood Modeling using VSURF Random Forest - Resolution 23
#==============================================================================
#
# This script performs flood susceptibility modeling using VSURF (Variable 
# Selection Using Random Forests) with Random Forest classification.
#
# Features:
# - Variable selection using VSURF
# - Multiple model configurations (HG, HGPT, HG8M)
# - Performance evaluation with ROC analysis
# - Prediction on full study area
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
RESOLUTION <- 23
TRAIN_FRACTION <- 0.7
NTREES <- 500
NMIN <- 3
NCORES <- 8

#==============================================================================
# Data Preparation
#==============================================================================

cat("Loading and preparing data...\n")

# Read sample points
data_file <- paste0("../Data/NB_sample_cent_quanti3_", RESOLUTION, ".csv")
if (!file.exists(data_file)) {
  stop(paste("Data file not found:", data_file))
}

spoints.23 <- read.csv(data_file, header = TRUE, fileEncoding = "UTF-8-BOM")

# Remove spatial coordinate columns (not used in modeling)
spatial_cols <- c("i", "j", "Cell_address", "lon_c", "lat_c")
spoints.23 <- subset(spoints.23, select = -which(names(spoints.23) %in% spatial_cols))

# Remove rows with no-data values (-32767)
spoints.23 <- spoints.23 %>% filter_all(all_vars(!grepl(-32767, .)))

# Remove rows with missing values
spoints.23 <- spoints.23[complete.cases(spoints.23), ]

cat("Data shape after cleaning:", dim(spoints.23), "\n")

# Convert categorical variables to factors
categorical_cols <- c("grid_code", "geo", "wl", "sol", "lc", "ia", "fcp")
spoints.23[categorical_cols] <- lapply(spoints.23[categorical_cols], factor)

# Define variable order for consistent modeling
col_order <- c("grid_code", "geo", "wl", "sol", "lc", "ndvi", "msi", "ia", "fcp", 
               "dtm", "slp", "asp", "curv", "rgh", "tri", "tpi", "nhn",
               "precip", "tavg", "r10", "r25", "tm10", "sd50", "ts", "spr")

var_order <- c("geo", "wl", "sol", "lc", "ndvi", "msi", "ia", "fcp", 
               "dtm", "slp", "asp", "curv", "rgh", "tri", "tpi", "nhn",
               "precip", "tavg", "r10", "r25", "tm10", "sd50", "ts", "spr")

# Reorder columns
spoints.23 <- spoints.23[, col_order]

# Split data into training and testing sets
spoints.23.train <- spoints.23 %>% sample_frac(TRAIN_FRACTION)
spoints.23.test <- setdiff(spoints.23, spoints.23.train)

cat("Training set size:", nrow(spoints.23.train), "\n")
cat("Testing set size:", nrow(spoints.23.test), "\n")

#==============================================================================
# Model Training
#==============================================================================

cat("Training VSURF models...\n")

# Function to train VSURF model
train_vsurf_model <- function(x_data, y_data, model_name, n_vars) {
  cat("Training", model_name, "model with", n_vars, "variables...\n")
  
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
}

# Model 1: Hydro-Geomorphological variables (16 variables)
cat("\n=== Model 1: Hydro-Geomorphological (HG) ===\n")
vnb.HG.23 <- train_vsurf_model(
  x = spoints.23.train[, 2:17], 
  y = spoints.23.train[, 1], 
  model_name = "HG", 
  n_vars = 16
)

# Model 2: HG + Precipitation and Temperature (18 variables)
cat("\n=== Model 2: HG + Precipitation/Temperature (HGPT) ===\n")
vnb.HGPT.23 <- train_vsurf_model(
  x = spoints.23.train[, 2:19], 
  y = spoints.23.train[, 1], 
  model_name = "HGPT", 
  n_vars = 18
)

# Model 3: All variables (24 variables)
cat("\n=== Model 3: All variables (HG8M) ===\n")
vnb.HG8M.23 <- train_vsurf_model(
  x = spoints.23.train[, 2:25], 
  y = spoints.23.train[, 1], 
  model_name = "HG8M", 
  n_vars = 24
)

#==============================================================================
# Variable Selection Results
#==============================================================================

cat("Analyzing variable selection results...\n")

# Function to map index to variable name
index_to_var <- function(index_vec) {
  var_name_vec <- c()
  for (i in index_vec) {
    var_name <- var_order[i]
    var_name_vec <- append(var_name_vec, var_name)
  }
  return(var_name_vec)
}

# Function to print variable selection results
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

# Print results for all models
print_variable_selection(vnb.HG.23, "HG")
print_variable_selection(vnb.HGPT.23, "HGPT")
print_variable_selection(vnb.HG8M.23, "HG8M")

#==============================================================================
# Model Performance Evaluation
#==============================================================================

cat("Evaluating model performance...\n")

# Function to evaluate model performance
evaluate_model <- function(model, test_data, test_features, model_name) {
  cat("\n---", model_name, "Performance Evaluation ---\n")
  
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
}

# Evaluate all models
hg_results <- evaluate_model(
  vnb.HG.23, 
  spoints.23.test, 
  spoints.23.test[, 2:17], 
  "HG"
)

hgpt_results <- evaluate_model(
  vnb.HGPT.23, 
  spoints.23.test, 
  spoints.23.test[, 2:19], 
  "HGPT"
)

hg8m_results <- evaluate_model(
  vnb.HG8M.23, 
  spoints.23.test, 
  spoints.23.test[, 2:25], 
  "HG8M"
)

#==============================================================================
# Full Study Area Prediction
#==============================================================================

cat("Making predictions on full study area...\n")

# Read all centroid points for full area prediction
prediction_file <- paste0("../Data/NB_finalPredict_", RESOLUTION, ".csv")
if (!file.exists(prediction_file)) {
  stop(paste("Prediction file not found:", prediction_file))
}

cent.23 <- read.csv(prediction_file, header = TRUE, fileEncoding = "UTF-8-BOM")

# Remove spatial coordinate columns
spatial_cols <- c("i", "j", "Cell_address")
cent.23 <- subset(cent.23, select = -which(names(cent.23) %in% spatial_cols))

# Remove rows with no-data values
cent.23 <- cent.23 %>% filter_all(all_vars(!grepl(-32767, .)))

cat("Prediction data shape:", dim(cent.23), "\n")
cent.23 = cent.23[complete.cases(cent.23), ]

# convert to factor
cols = c("geo", "wl", "sol", "lc", "ia", "fcp")
cent.23[cols] = lapply(cent.23[cols], factor)
cent.23$HG = predict(vnb.HG.23, newdata = cent.23[,3:26], step = c("pred"))
cent.23$HGPT = predict(vnb.HGPT.23, newdata = cent.23[,3:26], step = c("pred"))
cent.23$HG8M = predict(vnb.HG8M.23, newdata = cent.23[,3:26], step = c("pred"))

pred.result.23 = select(cent.23,c('lon_c','lat_c','HG','HGPT','HG8M'))
write.csv(pred.result.23,"../Result/NB_centPredict_23.csv", row.names = FALSE)
save.image(file = "test_nb_vsurf_pred_23.RData")