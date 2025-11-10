# ==================================================================================================
# Batch GAIN (Generative Adversarial Imputation Nets)
# Processes all dataset pairs from train/ and validate/ folders
# ==================================================================================================

# GAIN Hyperparameters
GAIN_CONFIG <- list(
  batch_size = 128,
  hint_rate = 0.9,
  alpha = 100,
  iterations = 5000,
  h_dim = 256,
  learning_rate = 0.001,
  m = 10
)

# Set to TRUE to use simplified version (no TensorFlow needed)
USE_SIMPLIFIED_VERSION <- TRUE

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
# GAIN Processing Function
# ==================================================================================================

process_gain_dataset <- function(dataset_name) {
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
  
  cat("Mode:", ifelse(USE_SIMPLIFIED_VERSION, "SIMPLIFIED", "FULL GAIN"), "\n\n")
  
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
  
  gain_result <- tryCatch({
    np <- import("numpy")
    
    train_data <- train[, setdiff(colnames(train), id_col)]
    data_matrix <- as.matrix(train_data)
    
    cat("Input validation:\n")
    cat("  Dimensions:", nrow(data_matrix), "x", ncol(data_matrix), "\n")
    cat("  Missing values:", sum(is.na(data_matrix)), "\n\n")
    
    if(all(is.na(data_matrix))) {
      stop("All data is missing!")
    }
    
    GAIN_CONFIG$m <- as.integer(GAIN_CONFIG$m)
    GAIN_CONFIG$batch_size <- as.integer(GAIN_CONFIG$batch_size)
    GAIN_CONFIG$iterations <- as.integer(GAIN_CONFIG$iterations)
    GAIN_CONFIG$h_dim <- as.integer(GAIN_CONFIG$h_dim)
    
    if(USE_SIMPLIFIED_VERSION) {
      cat("Using SIMPLIFIED GAIN (iterative mean matching)...\n\n")
      
      py$data_x <- np$array(data_matrix)
      py$gain_params <- GAIN_CONFIG
      
      py_run_string("
import numpy as np

def simplified_gain(data_x, gain_parameters, m=10):
    m = int(m)
    no, dim = data_x.shape
    no = int(no)
    dim = int(dim)
    
    mask = ~np.isnan(data_x)
    norm_data = np.copy(data_x)
    min_val = np.nanmin(norm_data, axis=0)
    max_val = np.nanmax(norm_data, axis=0)
    range_val = max_val - min_val
    range_val[range_val == 0] = 1.0
    norm_data = (norm_data - min_val) / range_val
    norm_data = np.nan_to_num(norm_data, 0)
    
    imputations = []
    
    for imp_iter in range(m):
        imputed = norm_data.copy()
        
        for col in range(dim):
            missing_mask = ~mask[:, col]
            if np.any(missing_mask):
                observed_values = norm_data[mask[:, col], col]
                if len(observed_values) > 0:
                    col_mean = np.mean(observed_values)
                    col_std = np.std(observed_values)
                    if col_std == 0:
                        col_std = 0.1
                    noise = np.random.normal(0, col_std * 0.1, np.sum(missing_mask))
                    imputed[missing_mask, col] = col_mean + noise
        
        for iteration in range(10):
            for col in range(dim):
                missing_mask = ~mask[:, col]
                if np.any(missing_mask) and np.sum(mask[:, col]) > 0:
                    other_cols = [c for c in range(dim) if c != col and np.sum(mask[:, c]) > 3]
                    if len(other_cols) > 0:
                        complete_mask = mask[:, col]
                        for oc in other_cols:
                            complete_mask = complete_mask & mask[:, oc]
                        if np.sum(complete_mask) > 3:
                            X_train = imputed[complete_mask][:, other_cols]
                            y_train = norm_data[complete_mask, col]
                            X_pred = imputed[missing_mask][:, other_cols]
                            for i in range(len(X_pred)):
                                distances = np.sum((X_train - X_pred[i])**2, axis=1)
                                k = min(3, len(X_train))
                                nearest = np.argsort(distances)[:k]
                                imputed[missing_mask, col][i] = np.mean(y_train[nearest])
        
        imputed_denorm = imputed * range_val + min_val
        final_imputed = np.where(mask, data_x, imputed_denorm)
        imputations.append(final_imputed)
    
    return imputations

imputations_list = simplified_gain(data_x, gain_params, m=gain_params['m'])
")
      
    } else {
      if(!py_module_available("tensorflow")) {
        stop("TensorFlow not available. Set USE_SIMPLIFIED_VERSION <- TRUE")
      }
      
      cat("Using FULL GAIN with TensorFlow...\n\n")
      
      py$data_x <- np$array(data_matrix)
      py$gain_params <- GAIN_CONFIG
      
      py_run_string("
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def gain_imputation(data_x, gain_parameters):
    batch_size = int(gain_parameters['batch_size'])
    hint_rate = float(gain_parameters['hint_rate'])
    alpha = float(gain_parameters['alpha'])
    iterations = int(gain_parameters['iterations'])
    h_dim = int(gain_parameters['h_dim'])
    lr = float(gain_parameters['learning_rate'])
    
    no, dim = data_x.shape
    no = int(no)
    dim = int(dim)
    
    norm_data = np.copy(data_x)
    min_val = np.nanmin(norm_data, axis=0)
    max_val = np.nanmax(norm_data, axis=0)
    norm_data = (norm_data - min_val) / (max_val - min_val + 1e-6)
    
    mask = 1 - np.isnan(norm_data)
    norm_data = np.nan_to_num(norm_data, 0)
    
    def generator(x, m):
        inputs = tf.concat([x, m], axis=1)
        G_h1 = layers.Dense(h_dim, activation='relu')(inputs)
        G_h2 = layers.Dense(h_dim, activation='relu')(G_h1)
        G_out = layers.Dense(dim, activation='sigmoid')(G_h2)
        return G_out
    
    def discriminator(x, h):
        inputs = tf.concat([x, h], axis=1)
        D_h1 = layers.Dense(h_dim, activation='relu')(inputs)
        D_h2 = layers.Dense(h_dim, activation='relu')(D_h1)
        D_out = layers.Dense(dim, activation='sigmoid')(D_h2)
        return D_out
    
    X_input = keras.Input(shape=(dim,))
    M_input = keras.Input(shape=(dim,))
    H_input = keras.Input(shape=(dim,))
    
    G_sample = generator(X_input, M_input)
    Hat_X = M_input * X_input + (1 - M_input) * G_sample
    D_prob = discriminator(Hat_X, H_input)
    
    generator_model = keras.Model(inputs=[X_input, M_input], outputs=G_sample)
    discriminator_model = keras.Model(inputs=[X_input, M_input, H_input], outputs=D_prob)
    
    G_optimizer = keras.optimizers.Adam(learning_rate=lr)
    D_optimizer = keras.optimizers.Adam(learning_rate=lr)
    
    for it in range(iterations):
        batch_idx = np.random.choice(no, min(int(batch_size), int(no)), replace=False)
        X_mb = norm_data[batch_idx, :]
        M_mb = mask[batch_idx, :]
        H_mb = np.random.binomial(1, hint_rate, [len(batch_idx), int(dim)])
        H_mb = M_mb * H_mb
        
        X_mb_t = tf.constant(X_mb, dtype=tf.float32)
        M_mb_t = tf.constant(M_mb, dtype=tf.float32)
        H_mb_t = tf.constant(H_mb, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            G_sample = generator_model([X_mb_t, M_mb_t])
            Hat_X = M_mb_t * X_mb_t + (1 - M_mb_t) * G_sample
            D_prob = discriminator_model([Hat_X, M_mb_t, H_mb_t])
            D_loss = -tf.reduce_mean(M_mb_t * tf.math.log(D_prob + 1e-8) + 
                                    (1 - M_mb_t) * tf.math.log(1 - D_prob + 1e-8))
        
        D_grads = tape.gradient(D_loss, discriminator_model.trainable_variables)
        D_optimizer.apply_gradients(zip(D_grads, discriminator_model.trainable_variables))
        
        with tf.GradientTape() as tape:
            G_sample = generator_model([X_mb_t, M_mb_t])
            Hat_X = M_mb_t * X_mb_t + (1 - M_mb_t) * G_sample
            D_prob = discriminator_model([Hat_X, M_mb_t, H_mb_t])
            G_loss1 = -tf.reduce_mean((1 - M_mb_t) * tf.math.log(D_prob + 1e-8))
            G_loss2 = tf.reduce_mean((M_mb_t * X_mb_t - M_mb_t * G_sample) ** 2) / tf.reduce_mean(M_mb_t)
            G_loss = G_loss1 + alpha * G_loss2
        
        G_grads = tape.gradient(G_loss, generator_model.trainable_variables)
        G_optimizer.apply_gradients(zip(G_grads, generator_model.trainable_variables))
        
        if it % 500 == 0:
            print(f'Iteration {it}/{iterations}')
    
    M_full = tf.constant(mask, dtype=tf.float32)
    X_full = tf.constant(norm_data, dtype=tf.float32)
    imputed_data = generator_model([X_full, M_full]).numpy()
    imputed_data = mask * norm_data + (1 - mask) * imputed_data
    imputed_data = imputed_data * (max_val - min_val + 1e-6) + min_val
    
    return imputed_data

def gain_multiple_imputation(data_x, gain_parameters, m=10):
    m = int(m)
    imputations = []
    for i in range(m):
        imputed = gain_imputation(data_x, gain_parameters)
        imputations.append(imputed)
    return imputations

imputations_list = gain_multiple_imputation(data_x, gain_params, m=gain_params['m'])
")
    }
    
    imputations_list <- py$imputations_list
    cat("GAIN imputation completed!\n\n")
    
    if(length(imputations_list) == 0 || all(is.na(imputations_list[[1]]))) {
      stop("Imputation returned no valid data")
    }
    
    m_value <- as.integer(GAIN_CONFIG$m)
    all_imputations_array <- array(
      unlist(imputations_list), 
      dim = c(nrow(data_matrix), ncol(data_matrix), m_value)
    )
    averaged_imputations <- apply(all_imputations_array, c(1, 2), mean, na.rm = TRUE)
    
    completed_data <- as.data.frame(averaged_imputations)
    colnames(completed_data) <- colnames(train_data)
    completed_data[[id_col]] <- train[[id_col]]
    
    target_col_idx <- which(colnames(train_data) == target_col)
    imputed_vals <- data.frame(
      col_ID = train[[id_col]][is.na(train[[target_col]])],
      imputed_value = averaged_imputations[is.na(train[[target_col]]), target_col_idx]
    )
    colnames(imputed_vals)[1] <- id_col
    
    comparison <- merge(true_vals, imputed_vals, by = id_col)
    
    if(nrow(comparison) == 0) {
      stop("No matching IDs")
    }
    
    metrics <- compute_metrics(comparison$true_value, comparison$imputed_value)
    
    list(
      success = TRUE,
      method = ifelse(USE_SIMPLIFIED_VERSION, "GAIN_Simplified", "GAIN_Full"),
      metrics = metrics,
      comparison = comparison,
      completed_data = completed_data
    )
    
  }, error = function(e) {
    cat("\nError:", e$message, "\n")
    list(success = FALSE, error = e$message)
  })
  
  if(!gain_result$success) {
    return(NULL)
  }
  
  # Save results
  method_name <- ifelse(USE_SIMPLIFIED_VERSION, "GAIN_Simplified", "GAIN_Full")
  results_file <- paste0(dataset_name, "_", method_name, "_results.csv")
  comparison_file <- paste0(dataset_name, "_", method_name, "_actual_vs_imputed.csv")
  completed_file <- paste0(dataset_name, "_", method_name, "_completed.csv")
  
  write.csv(gain_result$metrics, results_file, row.names = FALSE)
  write.csv(gain_result$comparison, comparison_file, row.names = FALSE)
  write.csv(gain_result$completed_data, completed_file, row.names = FALSE)
  
  cat("\n=== RESULTS FOR", dataset_name, "===\n")
  print(gain_result$metrics)
  cat("\nFiles saved:\n")
  cat("-", results_file, "\n")
  cat("-", comparison_file, "\n")
  cat("-", completed_file, "\n")
  
  return(list(
    dataset = dataset_name,
    metrics = gain_result$metrics,
    success = TRUE
  ))
}

# ==================================================================================================
# Main Execution
# ==================================================================================================

cat("==================================================================================================\n")
cat("BATCH GAIN PROCESSING - ALL DATASETS\n")
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
    result <- process_gain_dataset(dataset)
    
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
  
  method_name <- ifelse(USE_SIMPLIFIED_VERSION, "GAIN_Simplified", "GAIN_Full")
  summary_file <- paste0("batch_", method_name, "_summary.csv")
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