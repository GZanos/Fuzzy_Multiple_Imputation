# ==================================================================================================
# Batch Generalized FBLiR (Fuzzy Bayesian Linear Regression) Imputation Method
# Processes all dataset pairs from train/ and validate/ folders
# ==================================================================================================

# FBLIR Hyperparameter Configuration
# ===================================
# JAGS MCMC Settings
JAGS_N_CHAINS <- 4      # Number of MCMC chains
JAGS_ADAPT <- 500       # Adaptation steps
JAGS_BURNIN <- 500      # Burn-in steps
JAGS_THIN <- 7          # Thinning interval
JAGS_SAMPLE <- 8000     # Sample size per chain

# Hyperparameter Grid Search Settings
M_VALUES <- seq(-1, 1, by = 0.1)
SYMMETRY_THRESHOLDS <- c(0.1, 0.5, 1)
K_VALUES <- seq(-1, 1, by = 0.1)
UNCERTAINTY_WEIGHTS <- c(0, 0.25, 0.75, 1)
FUZZIFY_SCALES <- c(0.01, 0.05, 0.1)  # Variance values for fuzzification

# ==================================================================================================
# Setup and Package Loading
# ==================================================================================================

req <- c("Metrics","dplyr", "rjags", "runjags", "coda")
to_install <- setdiff(req, rownames(installed.packages()))
if(length(to_install)) install.packages(to_install, dependencies = TRUE)

library(Metrics)
library(dplyr)
library(rjags)
library(runjags)
library(coda)

# --- Configuration ---
id_col <- "col_ID"
target_col <- "Target"
set.seed(123)

# ==================================================================================================
# Required GFN Functions
# ==================================================================================================

fuzzify_feature <- function(feature, variance) {
  feature <- as.numeric(feature)
  n <- length(feature)
  var_vec <- rep(variance, n)
  cbind(Mean = feature, Variance = var_vec)
}

GFN.add <- function(A, B) {
  A <- as.numeric(A); B <- as.numeric(B)
  mean <- A[1] + B[1]
  variance <- A[2] + B[2]
  setNames(c(mean, variance), c("Mean", "Variance"))
}

GFN.multi <- function(A, B, symmetry.threshold = 4) {
  A <- as.numeric(A); B <- as.numeric(B)
  mean <- A[1] * B[1]
  variance <- (B[2] * A[1]^2) + (A[2] * B[1]^2) + (B[2] * A[2])
  
  if ((mean != 0) & (abs(mean/sqrt(variance)) < symmetry.threshold )) {
    while (abs(mean/sqrt(variance)) < symmetry.threshold ){
      variance <- variance*0.1
    }
  }
  setNames(c(mean, variance), c("Mean", "Variance"))
}

defuzzify <- function(gfn, k, m, symmetry.threshold) {
  mean_val <- gfn[1]
  var_val <- gfn[2]
  
  if (is.na(mean_val) || is.na(var_val) || var_val <= 0) {
    return(mean_val)
  }
  
  delta_val <- abs(mean_val / sqrt(var_val))
  if (delta_val > symmetry.threshold) {
    return(mean_val)
  } else {
    adj_factor <- m / (1 + exp(-k * (delta_val - symmetry.threshold)))
    return(mean_val + adj_factor * var_val)
  }
}

# ==================================================================================================
# Performance Metrics Functions
# ==================================================================================================

safe_mape <- function(actual, pred) {
  actual <- as.numeric(actual); pred <- as.numeric(pred)
  ok <- !is.na(actual) & !is.na(pred) & (actual != 0)
  if(!any(ok)) return(NA_real_)
  mean(abs((pred[ok] - actual[ok]) / actual[ok]))
}

compute_metrics <- function(actual, pred) {
  actual <- as.numeric(actual); pred <- as.numeric(pred)
  data.frame(
    MSE = mean((pred - actual)^2, na.rm = TRUE),
    MAE = mean(abs(pred - actual), na.rm = TRUE),
    RMSE = sqrt(mean((pred - actual)^2, na.rm = TRUE)),
    MAPE = safe_mape(actual, pred),
    check.names = FALSE
  )
}

score_model <- function(model_name, imputed_vector, train_data, missing_ids, true_vals) {
  pred_df <- data.frame(id = train_data[[id_col]],
                        imputed = as.numeric(imputed_vector))
  colnames(pred_df)[1] <- id_col
  
  eval_df <- pred_df %>%
    dplyr::filter(!!sym(id_col) %in% missing_ids) %>%
    inner_join(true_vals, by = id_col)
  
  if(nrow(eval_df) == 0) {
    return(data.frame(Model = model_name, MSE = NA, MAE = NA, RMSE = NA, MAPE = NA))
  }
  
  mets <- compute_metrics(eval_df$true_target, eval_df$imputed)
  cbind(data.frame(Model = model_name), mets)
}

# ==================================================================================================
# Main FBLiR Processing Function
# ==================================================================================================

process_fblir_dataset <- function(dataset_name) {
  cat("\n")
  cat("==================================================================================================\n")
  cat("Processing Dataset:", dataset_name, "\n")
  cat("==================================================================================================\n")
  
  train_file <- file.path("train", paste0(dataset_name, "_train.csv"))
  validate_file <- file.path("validate", paste0(dataset_name, "_validate.csv"))
  
  cat("Train file:", train_file, "\n")
  cat("Validate file:", validate_file, "\n")
  
  # Check if files exist
  if(!file.exists(train_file)) {
    cat("ERROR: Training file not found:", train_file, "\n")
    return(NULL)
  }
  if(!file.exists(validate_file)) {
    cat("ERROR: Validation file not found:", validate_file, "\n")
    return(NULL)
  }
  
  # Load data
  train <- read.csv(train_file, stringsAsFactors = FALSE)
  validate <- read.csv(validate_file, stringsAsFactors = FALSE)
  
  # Convert target to numeric if it's character
  if(is.character(train[[target_col]])) {
    train[[target_col]][trimws(train[[target_col]]) == ""] <- NA
  }
  train[[target_col]] <- suppressWarnings(as.numeric(train[[target_col]]))
  
  # Identify missing values
  missing_ids <- train[[id_col]][is.na(train[[target_col]])]
  cat("Found", length(missing_ids), "missing values to impute\n")
  
  if(length(missing_ids) == 0) {
    cat("No missing values found - skipping dataset\n")
    return(NULL)
  }
  
  # Get true values
  true_vals <- validate[validate[[id_col]] %in% missing_ids, c(id_col, target_col)]
  colnames(true_vals) <- c(id_col, "true_target")
  cat("Retrieved", nrow(true_vals), "true values for validation\n")
  
  # Prepare features
  feature_cols <- setdiff(colnames(train), c(id_col, target_col))
  cat("Feature columns:", length(feature_cols), "\n")
  
  for (fc in feature_cols) {
    if(!is.numeric(train[[fc]])) {
      train[[fc]] <- suppressWarnings(as.numeric(train[[fc]]))
    }
  }
  
  # Initialize results
  all_results <- list()
  actual_vs_imputed <- data.frame(
    id = missing_ids,
    actual = true_vals$true_target[match(missing_ids, true_vals[[id_col]])]
  )
  colnames(actual_vs_imputed)[1] <- id_col
  
  # Prepare data for JAGS
  X_full_df <- train[, feature_cols, drop = FALSE]
  Y_full <- train[, target_col]
  
  obs_idx <- which(!is.na(Y_full))
  miss_idx <- which(is.na(Y_full))
  
  if(length(obs_idx) <= 10) {
    cat("Insufficient observations (", length(obs_idx), ") - need > 10 complete cases\n")
    return(NULL)
  }
  
  cat("Sufficient observations found (", length(obs_idx), "), proceeding with FBLiR...\n")
  
  # Standardize data
  X_obs_matrix <- as.matrix(X_full_df[obs_idx, ])
  x_means <- colMeans(X_obs_matrix, na.rm = TRUE)
  x_sds <- apply(X_obs_matrix, 2, sd, na.rm = TRUE)
  x_sds[x_sds == 0] <- 1e-6
  X_obs_z <- scale(X_obs_matrix, center = x_means, scale = x_sds)
  
  Y_obs_raw <- Y_full[obs_idx]
  y_mean_train <- mean(Y_obs_raw, na.rm = TRUE)
  y_sd_train <- sd(Y_obs_raw, na.rm = TRUE)
  if (is.na(y_sd_train) || y_sd_train == 0) y_sd_train <- 1e-6
  Y_obs_z <- (Y_obs_raw - y_mean_train) / y_sd_train
  
  # JAGS model
  modelString <- "
  model {
    for (i in 1:N_obs) {
      y[i] ~ dnorm(mu[i], tau)
      mu[i] <- beta0 + inprod(beta[], X[i,])
    }
    
    beta0 ~ dnorm(0, tau_beta0)
    tau_beta0 ~ dgamma(1, 1)
    
    for (j in 1:P) {
      beta[j] ~ dnorm(0, tau_beta[j])
      tau_beta[j] ~ dgamma(alpha_beta, beta_beta)
    }
    
    alpha_beta ~ dgamma(1, 1)
    beta_beta ~ dgamma(1, 1)
    
    tau ~ dgamma(1, 1)
    sigma <- 1 / sqrt(tau)
    
    sigma2_resid <- 1/tau
  }
  "
  
  model_file <- paste0(dataset_name, "_FBLR_model.txt")
  writeLines(modelString, con = model_file)
  
  jags_data <- list(
    X = as.matrix(X_obs_z),
    y = as.numeric(Y_obs_z),
    N_obs = nrow(X_obs_z),
    P = ncol(X_obs_z)
  )
  
  params_to_monitor <- c("beta0", "beta", "sigma", "tau_beta0", "tau_beta", "sigma2_resid")
  
  fblir_success <- FALSE
  try({
    cat("Running JAGS model...\n")
    runJagsOut <- run.jags(
      model = model_file,
      data = jags_data,
      n.chains = JAGS_N_CHAINS,
      adapt = JAGS_ADAPT,
      burnin = JAGS_BURNIN,
      thin = JAGS_THIN,
      sample = JAGS_SAMPLE,
      monitor = params_to_monitor
    )
    
    mcmc_mat <- as.matrix(as.mcmc.list(runJagsOut))
    cat("JAGS model completed successfully\n")
    
    # Extract GFN coefficients
    beta0_samples <- mcmc_mat[, "beta0"]
    beta0_gfn <- c(mean(beta0_samples), var(beta0_samples))
    
    beta_gfn_matrix <- matrix(NA, nrow = ncol(X_obs_z), ncol = 2,
                              dimnames = list(colnames(X_obs_z), c("Mean", "Variance")))
    
    for (j in 1:ncol(X_obs_z)) {
      beta_col_name <- paste0("beta[", j, "]")
      tau_beta_col_name <- paste0("tau_beta[", j, "]")
      
      beta_samples <- mcmc_mat[, beta_col_name]
      tau_beta_samples <- mcmc_mat[, tau_beta_col_name]
      
      beta_mean <- mean(beta_samples)
      beta_var <- var(beta_samples)
      beta_uncertainty <- mean(1/tau_beta_samples)
      combined_var <- beta_var + 0.5 * beta_uncertainty
      
      beta_gfn_matrix[j, ] <- c(beta_mean, combined_var)
    }
    
    sigma2_resid_samples <- mcmc_mat[, "sigma2_resid"]
    sigma_gfn <- c(0, mean(sigma2_resid_samples))
    
    # Prepare standardized feature matrix
    X_full_z <- sweep(as.matrix(X_full_df), 2, x_means, FUN = "-")
    X_full_z <- sweep(X_full_z, 2, x_sds, FUN = "/")
    X_full_z[is.na(X_full_z)] <- 0
    
    n_all <- nrow(X_full_z)
    
    # Hyperparameter grid search
    fblr_grid <- expand.grid(
      m = M_VALUES,
      symmetry.threshold = SYMMETRY_THRESHOLDS,
      k = K_VALUES,
      uncertainty_weight = UNCERTAINTY_WEIGHTS,
      fuzzify_variance = FUZZIFY_SCALES
    )
    
    best_fblr_mse <- Inf
    best_fblr_preds <- NULL
    best_fblr_params <- NULL
    
    cat("Testing", nrow(fblr_grid), "hyperparameter combinations...\n")
    
    for (i in 1:nrow(fblr_grid)) {
      params <- fblr_grid[i, ]
      try({
        beta_gfn_weighted <- beta_gfn_matrix
        beta_gfn_weighted[, 2] <- beta_gfn_matrix[, 2] * params$uncertainty_weight + 
          (1 - params$uncertainty_weight) * mean(beta_gfn_matrix[, 2])
        
        # Fuzzify features
        fuzzified_features <- list()
        for (col_j in 1:ncol(X_full_z)) {
          fuzzified_features[[col_j]] <- fuzzify_feature(X_full_z[, col_j], params$fuzzify_variance)
        }
        
        # GFN Arithmetic
        Y_estimated_weighted <- matrix(NA, nrow = n_all, ncol = 2,
                                       dimnames = list(NULL, c("Mean", "Variance")))
        
        for (row_i in 1:n_all) {
          Y_gfn_temp <- beta0_gfn
          
          for (col_j in 1:ncol(X_full_z)) {
            X_gfn_temp <- fuzzified_features[[col_j]][row_i, ]
            product <- GFN.multi(beta_gfn_weighted[col_j, ], X_gfn_temp)
            Y_gfn_temp <- GFN.add(Y_gfn_temp, product)
          }
          Y_gfn_temp <- GFN.add(Y_gfn_temp, sigma_gfn)
          Y_estimated_weighted[row_i, ] <- Y_gfn_temp
        }
        
        # Defuzzify
        pred_defuzzified_z <- apply(Y_estimated_weighted, 1, defuzzify,
                                    k = params$k,
                                    m = params$m,
                                    symmetry.threshold = params$symmetry.threshold)
        
        pred_defuzzified_orig <- pred_defuzzified_z * y_sd_train + y_mean_train
        
        imputed_train <- train
        imputed_train[[target_col]][miss_idx] <- pred_defuzzified_orig[miss_idx]
        
        score <- score_model("FBLR_Bayesian_Tune", imputed_train[[target_col]], 
                             train, missing_ids, true_vals)
        
        if(!is.na(score$MSE) && score$MSE < best_fblr_mse) {
          best_fblr_mse <- score$MSE
          best_fblr_preds <- imputed_train[[target_col]]
          best_fblr_params <- params
        }
      }, silent = TRUE)
      
      if(i %% 100 == 0) {
        cat("  Completed", i, "/", nrow(fblr_grid), "combinations\n")
      }
    }
    
    if(!is.null(best_fblr_params)) {
      cat("\nBest hyperparameters:\n")
      cat("  - m:", best_fblr_params$m, "\n")
      cat("  - symmetry.threshold:", best_fblr_params$symmetry.threshold, "\n")
      cat("  - k:", best_fblr_params$k, "\n")
      cat("  - uncertainty_weight:", best_fblr_params$uncertainty_weight, "\n")
      cat("  - fuzzify_variance:", best_fblr_params$fuzzify_variance, "\n")
      cat("  - Best MSE:", best_fblr_mse, "\n")
    }
    
    if(!is.null(best_fblr_preds)) {
      fblr_score <- score_model("FBLR_Bayesian_Best", best_fblr_preds,
                                train, missing_ids, true_vals)
      all_results[["FBLR_Bayesian_Best"]] <- fblr_score
      actual_vs_imputed$FBLR_Bayesian_Best <- best_fblr_preds[match(missing_ids, train[[id_col]])]
      fblir_success <- TRUE
      cat("FBLiR method completed successfully\n")
    }
  }, silent = FALSE)
  
  if(!fblir_success) {
    cat("FBLiR method failed during execution\n")
    return(NULL)
  }
  
  # Compile results
  final_results <- dplyr::bind_rows(all_results) %>%
    mutate(across(c(MSE, MAE, RMSE, MAPE), as.numeric)) %>%
    select(Model, MSE, MAE, RMSE, MAPE) %>%
    arrange(MSE)
  
  # Save results in main directory
  results_file <- paste0(dataset_name, "_fblir_imputation_results.csv")
  values_file <- paste0(dataset_name, "_fblir_actual_vs_imputed_values.csv")
  
  write.csv(final_results, results_file, row.names = FALSE)
  write.csv(actual_vs_imputed, values_file, row.names = FALSE)
  
  cat("\n=== RESULTS FOR", dataset_name, "===\n")
  print(final_results)
  cat("\nFiles saved:\n")
  cat("-", results_file, "\n")
  cat("-", values_file, "\n")
  cat("-", model_file, "\n")
  
  return(list(
    dataset = dataset_name,
    results = final_results,
    best_mse = best_fblr_mse,
    success = TRUE
  ))
}

# ==================================================================================================
# Main Execution: Process All Datasets
# ==================================================================================================

cat("==================================================================================================\n")
cat("BATCH FBLIR PROCESSING - ALL DATASETS\n")
cat("==================================================================================================\n\n")

# Find all training files
train_files <- list.files("train", pattern = "_train\\.csv$", full.names = FALSE)

if(length(train_files) == 0) {
  cat("ERROR: No training files found in ./train/ folder\n")
} else {
  # Extract dataset names (remove _train.csv suffix)
  dataset_names <- gsub("_train\\.csv$", "", train_files)
  
  cat("Found", length(dataset_names), "dataset(s) to process:\n")
  cat(paste("  -", dataset_names, collapse = "\n"), "\n\n")
  
  # Process each dataset
  all_dataset_results <- list()
  summary_data <- data.frame()
  
  for(dataset in dataset_names) {
    result <- process_fblir_dataset(dataset)
    
    if(!is.null(result)) {
      all_dataset_results[[dataset]] <- result
      
      # Add to summary
      summary_row <- data.frame(
        Dataset = dataset,
        MSE = result$best_mse,
        Status = "Success"
      )
      summary_data <- rbind(summary_data, summary_row)
    } else {
      summary_row <- data.frame(
        Dataset = dataset,
        MSE = NA,
        Status = "Failed"
      )
      summary_data <- rbind(summary_data, summary_row)
    }
  }
  
  # Save overall summary
  cat("\n")
  cat("==================================================================================================\n")
  cat("OVERALL BATCH PROCESSING SUMMARY\n")
  cat("==================================================================================================\n")
  
  print(summary_data)
  
  summary_file <- "batch_fblir_summary.csv"
  write.csv(summary_data, summary_file, row.names = FALSE)
  
  successful <- sum(summary_data$Status == "Success")
  failed <- sum(summary_data$Status == "Failed")
  
  cat("\n")
  cat("Successfully processed:", successful, "dataset(s)\n")
  cat("Failed:", failed, "dataset(s)\n")
  cat("\nOverall summary saved as:", summary_file, "\n")
  cat("\n==================================================================================================\n")
  cat("BATCH PROCESSING COMPLETE\n")
  cat("==================================================================================================\n")
}