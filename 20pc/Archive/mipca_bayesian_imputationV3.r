# ==================================================================================================
# Batch MIPCA Bayesian Imputation Method
# Processes all dataset pairs from train/ and validate/ folders
# ==================================================================================================

# MIPCA Hyperparameters
MIPCA_CONFIG <- list(
  ncp = NULL,              # Number of PCA components (NULL = auto-estimate)
  ncp_max = 5,             # Maximum components to test during auto-estimation
  ncp_min = 2,             # Minimum components (safety fallback)
  scale = TRUE,            # Scale variables to unit variance
  method = "Regularized",  # "Regularized" or "EM"
  threshold = 1e-04,       # Convergence threshold
  method_mi = "Bayes",     # "Bayes" method
  nboot = 100,             # Number of imputed datasets to generate
  Lstart = 1000,           # Burn-in iterations for Bayesian method
  L = 100,                 # Iterations skipped between kept datasets
  verbose = FALSE          # Print iteration progress (set FALSE for batch)
)

# ==================================================================================================
# Setup and Package Loading
# ==================================================================================================

req <- c("missMDA", "Metrics", "dplyr")
to_install <- setdiff(req, rownames(installed.packages()))
if(length(to_install)) install.packages(to_install, dependencies = TRUE)

library(missMDA)
library(Metrics)
library(dplyr)

# Configuration
id_col <- "col_ID"
target_col <- "Target"
set.seed(123)

# ==================================================================================================
# Performance Metrics Functions
# ==================================================================================================

safe_mape <- function(actual, pred) {
  actual <- as.numeric(actual)
  pred <- as.numeric(pred)
  ok <- !is.na(actual) & !is.na(pred) & (actual != 0)
  if(!any(ok)) return(NA_real_)
  mean(abs((pred[ok] - actual[ok]) / actual[ok]))
}

compute_metrics <- function(actual, pred) {
  actual <- as.numeric(actual)
  pred <- as.numeric(pred)
  data.frame(
    MSE = mean((pred - actual)^2, na.rm = TRUE),
    MAE = mean(abs(pred - actual), na.rm = TRUE),
    RMSE = sqrt(mean((pred - actual)^2, na.rm = TRUE)),
    MAPE = safe_mape(actual, pred)
  )
}

# ==================================================================================================
# MIPCA Processing Function
# ==================================================================================================

process_mipca_dataset <- function(dataset_name) {
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
  
  cat("Features:", length(feature_cols), "\n")
  cat("Configuration: nboot =", MIPCA_CONFIG$nboot, ", Lstart =", MIPCA_CONFIG$Lstart, "\n\n")
  
  mipca_result <- tryCatch({
    # Prepare data matrix (exclude ID column)
    X <- train[, setdiff(colnames(train), id_col)]
    
    # Estimate optimal number of components if not specified
    if(is.null(MIPCA_CONFIG$ncp)) {
      cat("Estimating optimal number of PCA components...\n")
      ncp_est <- estim_ncpPCA(X, ncp.max = MIPCA_CONFIG$ncp_max, scale = MIPCA_CONFIG$scale)
      optimal_ncp <- ncp_est$ncp
      cat("  Estimated optimal components:", optimal_ncp, "\n")
      
      # Safeguard: ensure minimum components
      if(optimal_ncp < MIPCA_CONFIG$ncp_min) {
        cat("  WARNING: Using minimum of", MIPCA_CONFIG$ncp_min, "components instead\n")
        optimal_ncp <- MIPCA_CONFIG$ncp_min
      }
    } else {
      optimal_ncp <- MIPCA_CONFIG$ncp
      cat("Using specified number of components:", optimal_ncp, "\n")
    }
    cat("  Final components:", optimal_ncp, "\n\n")
    
    # Run MIPCA with Bayesian method
    cat("Running MIPCA Bayesian imputation...\n")
    cat("(This may take several minutes)\n")
    
    res <- MIPCA(
      X = X,
      ncp = optimal_ncp,
      scale = MIPCA_CONFIG$scale,
      method = MIPCA_CONFIG$method,
      threshold = MIPCA_CONFIG$threshold,
      method.mi = MIPCA_CONFIG$method_mi,
      nboot = MIPCA_CONFIG$nboot,
      Lstart = MIPCA_CONFIG$Lstart,
      L = MIPCA_CONFIG$L,
      verbose = MIPCA_CONFIG$verbose
    )
    
    cat("MIPCA completed successfully!\n\n")
    
    # Extract imputed values from the completed dataset
    imputed_data <- as.data.frame(res$res.imputePCA)
    imputed_data[[id_col]] <- train[[id_col]]
    
    # Get imputed target values for missing rows
    imputed_vals <- imputed_data[imputed_data[[id_col]] %in% missing_ids, c(id_col, target_col)]
    colnames(imputed_vals) <- c(id_col, "imputed_value")
    
    # Merge with true values
    comparison <- merge(true_vals, imputed_vals, by = id_col)
    
    if(nrow(comparison) == 0) {
      stop("No matching IDs between imputed values and validation set")
    }
    
    # Calculate metrics
    metrics <- compute_metrics(comparison$true_value, comparison$imputed_value)
    
    list(
      success = TRUE,
      method = "MIPCA_Bayesian",
      metrics = metrics,
      comparison = comparison,
      imputed_data = imputed_data,
      n_components = optimal_ncp
    )
    
  }, error = function(e) {
    cat("\nError in MIPCA:", e$message, "\n")
    list(success = FALSE, error = e$message)
  })
  
  if(!mipca_result$success) {
    return(NULL)
  }
  
  # Save results
  results_file <- paste0(dataset_name, "_MIPCA_Bayesian_results.csv")
  comparison_file <- paste0(dataset_name, "_MIPCA_Bayesian_actual_vs_imputed.csv")
  completed_file <- paste0(dataset_name, "_MIPCA_Bayesian_completed.csv")
  
  write.csv(mipca_result$metrics, results_file, row.names = FALSE)
  write.csv(mipca_result$comparison, comparison_file, row.names = FALSE)
  write.csv(mipca_result$imputed_data, completed_file, row.names = FALSE)
  
  cat("\n=== RESULTS FOR", dataset_name, "===\n")
  cat("Method: MIPCA Bayesian\n")
  cat("Components used:", mipca_result$n_components, "\n")
  cat("Performance Metrics:\n")
  cat(sprintf("  MSE:  %.6f\n", mipca_result$metrics$MSE))
  cat(sprintf("  MAE:  %.6f\n", mipca_result$metrics$MAE))
  cat(sprintf("  RMSE: %.6f\n", mipca_result$metrics$RMSE))
  cat(sprintf("  MAPE: %.6f\n", mipca_result$metrics$MAPE))
  cat("\nFiles saved:\n")
  cat("-", results_file, "\n")
  cat("-", comparison_file, "\n")
  cat("-", completed_file, "\n")
  
  return(list(
    dataset = dataset_name,
    metrics = mipca_result$metrics,
    n_components = mipca_result$n_components,
    success = TRUE
  ))
}

# ==================================================================================================
# Main Execution
# ==================================================================================================

cat("==================================================================================================\n")
cat("BATCH MIPCA BAYESIAN PROCESSING - ALL DATASETS\n")
cat("==================================================================================================\n\n")

train_files <- list.files("train", pattern = "_train\\.csv$", full.names = FALSE)

if(length(train_files) == 0) {
  cat("ERROR: No training files found in ./train/\n")
} else {
  dataset_names <- gsub("_train\\.csv$", "", train_files)
  
  cat("Found", length(dataset_names), "dataset(s) to process\n")
  cat(paste("  -", dataset_names, collapse = "\n"), "\n\n")
  
  cat("MIPCA Configuration:\n")
  cat("  Method:", MIPCA_CONFIG$method_mi, "\n")
  cat("  Number of imputed datasets (nboot):", MIPCA_CONFIG$nboot, "\n")
  cat("  Burn-in iterations (Lstart):", MIPCA_CONFIG$Lstart, "\n")
  cat("  Iterations between datasets (L):", MIPCA_CONFIG$L, "\n")
  cat("  Auto-estimate components: YES (max =", MIPCA_CONFIG$ncp_max, ")\n")
  cat("\n")
  
  all_results <- list()
  summary_data <- data.frame()
  
  for(dataset in dataset_names) {
    result <- process_mipca_dataset(dataset)
    
    if(!is.null(result)) {
      all_results[[dataset]] <- result
      summary_row <- data.frame(
        Dataset = dataset,
        MSE = result$metrics$MSE,
        MAE = result$metrics$MAE,
        RMSE = result$metrics$RMSE,
        MAPE = result$metrics$MAPE,
        N_Components = result$n_components,
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
        N_Components = NA,
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
  
  summary_file <- "batch_MIPCA_Bayesian_summary.csv"
  write.csv(summary_data, summary_file, row.names = FALSE)
  
  successful <- sum(summary_data$Status == "Success")
  failed <- sum(summary_data$Status == "Failed")
  
  cat("\nSuccessfully processed:", successful, "dataset(s)\n")
  cat("Failed:", failed, "dataset(s)\n")
  cat("Overall summary saved as:", summary_file, "\n")
  
  if(successful > 0) {
    cat("\nAverage MSE across successful datasets:", 
        mean(summary_data$MSE[summary_data$Status == "Success"], na.rm = TRUE), "\n")
    cat("Best performing dataset:", 
        summary_data$Dataset[which.min(summary_data$MSE)], 
        "(MSE =", min(summary_data$MSE, na.rm = TRUE), ")\n")
  }
  
  cat("\n==================================================================================================\n")
  cat("BATCH PROCESSING COMPLETE\n")
  cat("==================================================================================================\n")
}