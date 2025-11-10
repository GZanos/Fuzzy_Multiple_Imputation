# ==================================================================================================
# Batch MIDAS Simple Implementation
# Processes all dataset pairs from train/ and validate/ folders
# ==================================================================================================

# MIDAS Configuration
MIDAS_CONFIG <- list(
  layer_structure = c(256, 256),
  train_rate = 0.8,
  seed = 42,
  epochs = 20,
  m = 10
)

# ==================================================================================================
# Setup
# ==================================================================================================

req <- c("reticulate", "Metrics", "dplyr")
to_install <- setdiff(req, rownames(installed.packages()))
if(length(to_install)) install.packages(to_install, dependencies = TRUE)

library(reticulate)
library(Metrics)
library(dplyr)

id_col <- "col_ID"
target_col <- "Target"
set.seed(123)

# ==================================================================================================
# Performance Metrics
# ==================================================================================================

safe_mape <- function(actual, pred) {
  actual <- as.numeric(actual)
  pred <- as.numeric(pred)
  ok <- !is.na(actual) & !is.na(pred) & (actual != 0)
  if(!any(ok)) return(NA_real_)
  mean(abs((pred[ok] - actual[ok]) / actual[ok]))
}

compute_metrics <- function(actual, pred) {
  data.frame(
    MSE = mean((pred - actual)^2, na.rm = TRUE),
    MAE = mean(abs(pred - actual), na.rm = TRUE),
    RMSE = sqrt(mean((pred - actual)^2, na.rm = TRUE)),
    MAPE = safe_mape(actual, pred)
  )
}

# ==================================================================================================
# MIDAS Processing Function
# ==================================================================================================

process_midas_dataset <- function(dataset_name) {
  cat("\n")
  cat("==================================================================================================\n")
  cat("Processing Dataset:", dataset_name, "\n")
  cat("==================================================================================================\n")
  
  train_file <- file.path("train", paste0(dataset_name, "_train.csv"))
  validate_file <- file.path("validate", paste0(dataset_name, "_validate.csv"))
  
  if(!file.exists(train_file) || !file.exists(validate_file)) {
    cat("ERROR: Files not found\n")
    return(NULL)
  }
  
  # Load data
  train <- read.csv(train_file, stringsAsFactors = FALSE)
  validate <- read.csv(validate_file, stringsAsFactors = FALSE)
  
  if(is.character(train[[target_col]])) {
    train[[target_col]][trimws(train[[target_col]]) == ""] <- NA
  }
  train[[target_col]] <- suppressWarnings(as.numeric(train[[target_col]]))
  
  missing_ids <- train[[id_col]][is.na(train[[target_col]])]
  cat("Missing values to impute:", length(missing_ids), "\n")
  
  if(length(missing_ids) == 0) {
    cat("No missing values - skipping\n")
    return(NULL)
  }
  
  true_vals <- validate[validate[[id_col]] %in% missing_ids, c(id_col, target_col)]
  colnames(true_vals) <- c(id_col, "true_value")
  
  # Ensure numeric
  feature_cols <- setdiff(colnames(train), c(id_col, target_col))
  for (fc in feature_cols) {
    if(!is.numeric(train[[fc]])) {
      train[[fc]] <- suppressWarnings(as.numeric(train[[fc]]))
    }
  }
  
  cat("Features:", length(feature_cols), "\n\n")
  
  midas_result <- tryCatch({
    np <- import("numpy")
    
    train_data <- train[, setdiff(colnames(train), id_col)]
    data_matrix <- as.matrix(train_data)
    
    cat("Using simplified MIDAS (mean + noise)...\n")
    cat("Note: This is a simplified version.\n\n")
    
    py$data_matrix <- np$array(data_matrix)
    
    py_run_string("
import numpy as np

def simple_midas_imputation(data, m=10, noise_level=0.1):
    '''
    Simplified imputation using mean + gaussian noise
    This is NOT the full MIDAS algorithm but a simple alternative
    '''
    n_rows, n_cols = data.shape
    
    # Create mask
    mask = ~np.isnan(data)
    
    # Calculate column means and stds
    col_means = np.nanmean(data, axis=0)
    col_stds = np.nanstd(data, axis=0)
    
    # Replace zero stds
    col_stds[col_stds == 0] = 0.1
    
    # Generate multiple imputations
    imputations = []
    
    for i in range(m):
        # Copy data
        imputed = data.copy()
        
        # For each column with missing values
        for col in range(n_cols):
            missing_rows = np.where(~mask[:, col])[0]
            
            if len(missing_rows) > 0:
                # Impute with mean + noise for diversity
                noise = np.random.normal(0, col_stds[col] * noise_level, len(missing_rows))
                imputed[missing_rows, col] = col_means[col] + noise
        
        imputations.append(imputed)
    
    return imputations

# Run imputation
imputations = simple_midas_imputation(data_matrix, m=10, noise_level=0.1)
")
    
    imputations_list <- py$imputations
    cat("MIDAS imputation completed!\n\n")
    
    # Average across imputations
    all_imputations_array <- array(unlist(imputations_list), 
                                   dim = c(nrow(data_matrix), ncol(data_matrix), MIDAS_CONFIG$m))
    averaged_imputations <- apply(all_imputations_array, c(1, 2), mean)
    
    # Create completed dataset
    completed_data <- as.data.frame(averaged_imputations)
    colnames(completed_data) <- colnames(train_data)
    completed_data[[id_col]] <- train[[id_col]]
    
    # Get imputed target values
    target_col_idx <- which(colnames(train_data) == target_col)
    imputed_vals <- data.frame(
      col_ID = train[[id_col]][is.na(train[[target_col]])],
      imputed_value = averaged_imputations[is.na(train[[target_col]]), target_col_idx]
    )
    colnames(imputed_vals)[1] <- id_col
    
    # Merge with true values
    comparison <- merge(true_vals, imputed_vals, by = id_col)
    
    if(nrow(comparison) == 0) {
      stop("No matching IDs")
    }
    
    # Calculate metrics
    metrics <- compute_metrics(comparison$true_value, comparison$imputed_value)
    
    list(
      success = TRUE,
      method = "MIDAS_Simple",
      metrics = metrics,
      comparison = comparison,
      completed_data = completed_data
    )
    
  }, error = function(e) {
    cat("\nError:", e$message, "\n")
    
    if(grepl("numpy|module", e$message, ignore.case = TRUE)) {
      cat("Missing Python packages. Install with:\n")
      cat("  reticulate::py_install(c('numpy', 'pandas'))\n\n")
    }
    
    list(success = FALSE, error = e$message)
  })
  
  if(!midas_result$success) {
    return(NULL)
  }
  
  # Save results
  results_file <- paste0(dataset_name, "_MIDAS_Simple_results.csv")
  comparison_file <- paste0(dataset_name, "_MIDAS_Simple_actual_vs_imputed.csv")
  completed_file <- paste0(dataset_name, "_MIDAS_Simple_completed.csv")
  
  write.csv(midas_result$metrics, results_file, row.names = FALSE)
  write.csv(midas_result$comparison, comparison_file, row.names = FALSE)
  write.csv(midas_result$completed_data, completed_file, row.names = FALSE)
  
  cat("\n=== RESULTS FOR", dataset_name, "===\n")
  print(midas_result$metrics)
  cat("\nFiles saved:\n")
  cat("-", results_file, "\n")
  cat("-", comparison_file, "\n")
  cat("-", completed_file, "\n")
  
  return(list(
    dataset = dataset_name,
    metrics = midas_result$metrics,
    success = TRUE
  ))
}

# ==================================================================================================
# Main Execution
# ==================================================================================================

cat("==================================================================================================\n")
cat("BATCH MIDAS SIMPLE PROCESSING - ALL DATASETS\n")
cat("==================================================================================================\n\n")

train_files <- list.files("train", pattern = "_train\\.csv$", full.names = FALSE)

if(length(train_files) == 0) {
  cat("ERROR: No training files found in ./train/\n")
} else {
  dataset_names <- gsub("_train\\.csv$", "", train_files)
  
  cat("Found", length(dataset_names), "dataset(s) to process\n")
  cat(paste("  -", dataset_names, collapse = "\n"), "\n\n")
  
  all_results <- list()
  summary_data <- data.frame()
  
  for(dataset in dataset_names) {
    result <- process_midas_dataset(dataset)
    
    if(!is.null(result)) {
      all_results[[dataset]] <- result
      summary_row <- data.frame(
        Dataset = dataset,
        MSE = result$metrics$MSE,
        MAE = result$metrics$MAE,
        RMSE = result$metrics$RMSE,
        MAPE = result$metrics$MAPE,
        Status = "Success"
      )
      summary_data <- rbind(summary_data, summary_row)
    } else {
      summary_row <- data.frame(
        Dataset = dataset,
        MSE = NA,
        MAE = NA,
        RMSE = NA,
        MAPE = NA,
        Status = "Failed"
      )
      summary_data <- rbind(summary_data, summary_row)
    }
  }
  
  cat("\n")
  cat("==================================================================================================\n")
  cat("OVERALL BATCH PROCESSING SUMMARY\n")
  cat("==================================================================================================\n")
  
  print(summary_data)
  
  summary_file <- "batch_MIDAS_Simple_summary.csv"
  write.csv(summary_data, summary_file, row.names = FALSE)
  
  successful <- sum(summary_data$Status == "Success")
  failed <- sum(summary_data$Status == "Failed")
  
  cat("\nSuccessfully processed:", successful, "dataset(s)\n")
  cat("Failed:", failed, "dataset(s)\n")
  cat("Overall summary saved as:", summary_file, "\n")
  cat("\n==================================================================================================\n")
  cat("BATCH PROCESSING COMPLETE\n")
  cat("==================================================================================================\n")
}