# ==================================================================================================
# Batch Generalized Imputation Methods
# Processes all dataset pairs from train/ and validate/ folders
# ==================================================================================================

# ==================================================================================================
# Setup and Package Loading
# ==================================================================================================

req <- c("Amelia","missForest","VIM","softImpute","xgboost","Metrics","dplyr","reticulate","mice")
to_install <- setdiff(req, rownames(installed.packages()))
if(length(to_install)) install.packages(to_install, dependencies = TRUE)

library(Amelia)
library(missForest)
library(VIM)
library(softImpute)
library(xgboost)
library(Metrics)
library(dplyr)
library(reticulate)
library(mice)

# Configuration
id_col <- "col_ID"
target_col <- "Target"
set.seed(123)

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
# Helper function to save model results
# ==================================================================================================

save_model_results <- function(dataset_name, model_name, metrics, imputed_values, 
                               completed_data, missing_ids, true_vals, train_data) {
  # Save results metrics
  results_file <- paste0(dataset_name, "_", model_name, "_results.csv")
  write.csv(metrics, results_file, row.names = FALSE)
  
  # Save actual vs imputed for missing values only
  actual_vs_imputed <- data.frame(
    id = missing_ids,
    actual = true_vals$true_target[match(missing_ids, true_vals[[id_col]])],
    imputed = imputed_values[match(missing_ids, train_data[[id_col]])]
  )
  colnames(actual_vs_imputed)[1] <- id_col
  values_file <- paste0(dataset_name, "_", model_name, "_actual_vs_imputed.csv")
  write.csv(actual_vs_imputed, values_file, row.names = FALSE)
  
  # Save completed dataset
  completed_file <- paste0(dataset_name, "_", model_name, "_completed.csv")
  write.csv(completed_data, completed_file, row.names = FALSE)
  
  return(list(
    results_file = results_file,
    values_file = values_file,
    completed_file = completed_file
  ))
}

# ==================================================================================================
# Main Processing Function
# ==================================================================================================

process_generalized_dataset <- function(dataset_name) {
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
  colnames(true_vals) <- c(id_col, "true_target")
  
  feature_cols <- setdiff(colnames(train), c(id_col, target_col))
  for (fc in feature_cols) {
    if(!is.numeric(train[[fc]])) {
      train[[fc]] <- suppressWarnings(as.numeric(train[[fc]]))
    }
  }
  
  cat("Features:", length(feature_cols), "\n\n")
  
  # Storage for summary
  model_summary <- data.frame()
  
  # ==================================================================================================
  # 1) MVN (Amelia)
  # ==================================================================================================
  cat("Running MVN (Amelia)...\n")
  try({
    complete_rows <- complete.cases(train[, c(target_col, feature_cols)])
    if(sum(complete_rows) > 20) {
      cors <- sapply(feature_cols, function(x) {
        abs(cor(train[complete_rows, target_col], train[complete_rows, x], use = "complete.obs"))
      })
      cors[is.na(cors)] <- 0
      
      top_predictors <- names(sort(cors, decreasing = TRUE)[1:min(3, length(feature_cols))])
      am_data <- train[, c(target_col, top_predictors)]
      
      a.out <- amelia(am_data, m = 1, ridge = 0.05, p2s = 0)
      
      if(!is.null(a.out$imputations) && length(a.out$imputations) > 0) {
        mvn_imp <- a.out$imputations[[1]][[target_col]]
        
        completed_data <- train
        completed_data[[target_col]] <- mvn_imp
        
        metrics <- score_model("MVN_Amelia", mvn_imp, train, missing_ids, true_vals)
        files <- save_model_results(dataset_name, "MVN_Amelia", metrics, mvn_imp, 
                                    completed_data, missing_ids, true_vals, train)
        
        model_summary <- rbind(model_summary, data.frame(
          Model = "MVN_Amelia",
          MSE = metrics$MSE,
          Status = "Success"
        ))
        cat("MVN completed\n")
      }
    }
  }, silent = TRUE)
  
  # ==================================================================================================
  # 2) missForest
  # ==================================================================================================
  cat("Running missForest...\n")
  for (ntree in c(5, 10)) {
    for (maxiter in c(1, 2)) {
      model_name <- paste0("missForest_ntree", ntree, "_iter", maxiter)
      try({
        mf <- missForest(train[, c(target_col, feature_cols)],
                         ntree = ntree, maxiter = maxiter, verbose = FALSE)
        mf_imp <- mf$ximp[[target_col]]
        
        completed_data <- train
        completed_data[[target_col]] <- mf_imp
        
        metrics <- score_model(model_name, mf_imp, train, missing_ids, true_vals)
        files <- save_model_results(dataset_name, model_name, metrics, mf_imp,
                                    completed_data, missing_ids, true_vals, train)
        
        model_summary <- rbind(model_summary, data.frame(
          Model = model_name,
          MSE = metrics$MSE,
          Status = "Success"
        ))
      }, silent = TRUE)
    }
  }
  cat("missForest completed\n")
  
  # ==================================================================================================
  # 3) KNN
  # ==================================================================================================
  cat("Running KNN...\n")
  for (k in c(3, 5, 7, 9)) {
    model_name <- paste0("KNN_k", k)
    try({
      knn_df <- VIM::kNN(train[, c(target_col, feature_cols)], k = k, imp_var = FALSE)
      knn_imp <- knn_df[[target_col]]
      
      completed_data <- train
      completed_data[[target_col]] <- knn_imp
      
      metrics <- score_model(model_name, knn_imp, train, missing_ids, true_vals)
      files <- save_model_results(dataset_name, model_name, metrics, knn_imp,
                                  completed_data, missing_ids, true_vals, train)
      
      model_summary <- rbind(model_summary, data.frame(
        Model = model_name,
        MSE = metrics$MSE,
        Status = "Success"
      ))
    }, silent = TRUE)
  }
  cat("KNN completed\n")
  
  # ==================================================================================================
  # 4) XGBoost
  # ==================================================================================================
  cat("Running XGBoost...\n")
  xgb_grid <- expand.grid(max_depth = c(4, 6),
                          nrounds = c(200, 500),
                          eta = c(0.05, 0.1))
  
  nonmiss <- which(!is.na(train[[target_col]]))
  miss <- which(is.na(train[[target_col]]))
  
  for (i in seq_len(nrow(xgb_grid))) {
    params <- xgb_grid[i, ]
    model_name <- paste0("XGB_d", params$max_depth,
                         "_n", params$nrounds,
                         "_eta", gsub("\\.", "", as.character(params$eta)))
    try({
      dtrain <- xgb.DMatrix(
        data = as.matrix(train[nonmiss, feature_cols]),
        label = train[nonmiss, target_col],
        missing = NA
      )
      
      xgb_mod <- xgboost(
        data = dtrain,
        objective = "reg:squarederror",
        max_depth = params$max_depth,
        nrounds = params$nrounds,
        eta = params$eta,
        subsample = 1.0,
        colsample_bytree = 1.0,
        verbose = 0
      )
      
      preds <- rep(NA_real_, nrow(train))
      if(length(miss)) {
        dtest <- xgb.DMatrix(data = as.matrix(train[miss, feature_cols]), missing = NA)
        preds[miss] <- predict(xgb_mod, dtest)
        preds[nonmiss] <- train[nonmiss, target_col]
      } else {
        preds <- train[[target_col]]
      }
      
      completed_data <- train
      completed_data[[target_col]] <- preds
      
      metrics <- score_model(model_name, preds, train, missing_ids, true_vals)
      files <- save_model_results(dataset_name, model_name, metrics, preds,
                                  completed_data, missing_ids, true_vals, train)
      
      model_summary <- rbind(model_summary, data.frame(
        Model = model_name,
        MSE = metrics$MSE,
        Status = "Success"
      ))
    }, silent = TRUE)
  }
  cat("XGBoost completed\n")
  
  # ==================================================================================================
  # 5) softImpute
  # ==================================================================================================
  cat("Running softImpute...\n")
  si_grid <- expand.grid(lambda = c(0, 0.1, 1),
                         rank.max = c(2, 5, 8))
  
  for (i in seq_len(nrow(si_grid))) {
    params <- si_grid[i, ]
    model_name <- paste0("softImpute_lambda", gsub("\\.", "", as.character(params$lambda)),
                         "_rank", params$rank.max)
    
    try({
      M_data <- train[, c(target_col, feature_cols)]
      M <- as.matrix(M_data)
      mode(M) <- "numeric"
      
      non_missing_count <- sum(!is.na(M))
      total_elements <- nrow(M) * ncol(M)
      
      if(non_missing_count > total_elements * 0.1) {
        col_means <- colMeans(M, na.rm = TRUE)
        M_centered <- sweep(M, 2, col_means, FUN = "-")
        
        fit <- softImpute(M_centered, lambda = params$lambda, 
                          rank.max = params$rank.max, type = "svd", trace.it = FALSE)
        
        Mhat_centered <- complete(M_centered, fit)
        Mhat <- sweep(Mhat_centered, 2, col_means, FUN = "+")
        
        si_imp <- Mhat[, 1]
        
        completed_data <- train
        completed_data[[target_col]] <- si_imp
        
        metrics <- score_model(model_name, si_imp, train, missing_ids, true_vals)
        files <- save_model_results(dataset_name, model_name, metrics, si_imp,
                                    completed_data, missing_ids, true_vals, train)
        
        model_summary <- rbind(model_summary, data.frame(
          Model = model_name,
          MSE = metrics$MSE,
          Status = "Success"
        ))
      }
    }, silent = TRUE)
  }
  cat("softImpute completed\n")
  
  # ==================================================================================================
  # 6) GAIN (sklearn IterativeImputer)
  # ==================================================================================================
  cat("Running GAIN (IterativeImputer)...\n")
  model_name <- "GAIN_default"
  try({
    sklearn <- import("sklearn.experimental")
    sklearn$enable_iterative_imputer()
    impute <- import("sklearn.impute")
    
    complete_rows <- complete.cases(train[, c(target_col, feature_cols)])
    if(sum(complete_rows) > 20) {
      cors <- sapply(feature_cols, function(x) {
        abs(cor(train[complete_rows, target_col], train[complete_rows, x], use = "complete.obs"))
      })
      cors[is.na(cors)] <- 0
      top_features <- names(sort(cors, decreasing = TRUE)[1:min(5, length(feature_cols))])
      
      X_gain <- as.matrix(train[, c(target_col, top_features)])
      X_np <- r_to_py(X_gain)
      
      imputer <- impute$IterativeImputer(max_iter = 50L, random_state = 123L)
      X_imputed <- imputer$fit_transform(X_np)
      Xhat_r <- py_to_r(X_imputed)
      gain_imp <- Xhat_r[, 1]
      
      completed_data <- train
      completed_data[[target_col]] <- gain_imp
      
      metrics <- score_model(model_name, gain_imp, train, missing_ids, true_vals)
      files <- save_model_results(dataset_name, model_name, metrics, gain_imp,
                                  completed_data, missing_ids, true_vals, train)
      
      model_summary <- rbind(model_summary, data.frame(
        Model = model_name,
        MSE = metrics$MSE,
        Status = "Success"
      ))
      cat("GAIN completed\n")
    }
  }, silent = TRUE)
  
  # ==================================================================================================
  # 7) MICE
  # ==================================================================================================
  cat("Running MICE...\n")
  mice_configs <- list(
    list(method = "pmm", m = 5, maxit = 5, tag = "MICE_pmm_m5"),
    list(method = "pmm", m = 10, maxit = 10, tag = "MICE_pmm_m10"),
    list(method = "norm", m = 5, maxit = 5, tag = "MICE_norm_m5")
  )
  
  for(config in mice_configs) {
    model_name <- config$tag
    try({
      complete_rows <- complete.cases(train[, c(target_col, feature_cols)])
      if(sum(complete_rows) > 20) {
        cors <- sapply(feature_cols, function(x) {
          abs(cor(train[complete_rows, target_col], train[complete_rows, x], use = "complete.obs"))
        })
        cors[is.na(cors)] <- 0
        top_predictors <- names(sort(cors, decreasing = TRUE)[1:min(5, length(feature_cols))])
        
        mice_data <- train[, c(target_col, top_predictors)]
        
        if(config$method == "pmm") {
          method_vec <- rep("pmm", ncol(mice_data))
        } else {
          method_vec <- rep("norm", ncol(mice_data))
        }
        names(method_vec) <- colnames(mice_data)
        
        mice_result <- mice(mice_data, m = config$m, method = method_vec,
                            maxit = config$maxit, printFlag = FALSE, seed = 123)
        
        completed_list <- complete(mice_result, action = "all")
        
        mice_imp <- train[[target_col]]
        missing_idx <- which(is.na(train[[target_col]]))
        
        if(length(missing_idx) > 0) {
          avg_imputations <- sapply(missing_idx, function(idx) {
            values <- sapply(completed_list, function(df) df[idx, target_col])
            mean(values, na.rm = TRUE)
          })
          mice_imp[missing_idx] <- avg_imputations
        }
        
        completed_data <- train
        completed_data[[target_col]] <- mice_imp
        
        metrics <- score_model(model_name, mice_imp, train, missing_ids, true_vals)
        files <- save_model_results(dataset_name, model_name, metrics, mice_imp,
                                    completed_data, missing_ids, true_vals, train)
        
        model_summary <- rbind(model_summary, data.frame(
          Model = model_name,
          MSE = metrics$MSE,
          Status = "Success"
        ))
      }
    }, silent = TRUE)
  }
  cat("MICE completed\n")
  
  # ==================================================================================================
  # 8) Baseline methods
  # ==================================================================================================
  cat("Running baseline methods...\n")
  
  # Mean imputation
  mean_imp <- train[[target_col]]
  mean_imp[is.na(mean_imp)] <- mean(train[[target_col]], na.rm = TRUE)
  completed_data <- train
  completed_data[[target_col]] <- mean_imp
  metrics <- score_model("Mean_imputation", mean_imp, train, missing_ids, true_vals)
  files <- save_model_results(dataset_name, "Mean_imputation", metrics, mean_imp,
                              completed_data, missing_ids, true_vals, train)
  model_summary <- rbind(model_summary, data.frame(
    Model = "Mean_imputation",
    MSE = metrics$MSE,
    Status = "Success"
  ))
  
  # Median imputation
  median_imp <- train[[target_col]]
  median_imp[is.na(median_imp)] <- median(train[[target_col]], na.rm = TRUE)
  completed_data <- train
  completed_data[[target_col]] <- median_imp
  metrics <- score_model("Median_imputation", median_imp, train, missing_ids, true_vals)
  files <- save_model_results(dataset_name, "Median_imputation", metrics, median_imp,
                              completed_data, missing_ids, true_vals, train)
  model_summary <- rbind(model_summary, data.frame(
    Model = "Median_imputation",
    MSE = metrics$MSE,
    Status = "Success"
  ))
  
  cat("Baseline methods completed\n")
  
  # ==================================================================================================
  # Summary
  # ==================================================================================================
  
  cat("\n=== RESULTS FOR", dataset_name, "===\n")
  model_summary <- model_summary %>% arrange(MSE)
  print(model_summary)
  
  return(list(
    dataset = dataset_name,
    summary = model_summary,
    success = TRUE
  ))
}

# ==================================================================================================
# Main Execution
# ==================================================================================================

cat("==================================================================================================\n")
cat("BATCH GENERALIZED IMPUTATION METHODS - ALL DATASETS\n")
cat("==================================================================================================\n\n")

train_files <- list.files("train", pattern = "_train\\.csv$", full.names = FALSE)

if(length(train_files) == 0) {
  cat("ERROR: No training files found in ./train/\n")
} else {
  dataset_names <- gsub("_train\\.csv$", "", train_files)
  
  cat("Found", length(dataset_names), "dataset(s) to process\n")
  cat(paste("  -", dataset_names, collapse = "\n"), "\n\n")
  
  all_results <- list()
  overall_summary <- data.frame()
  
  for(dataset in dataset_names) {
    result <- process_generalized_dataset(dataset)
    
    if(!is.null(result)) {
      all_results[[dataset]] <- result
      
      # Add dataset name to summary
      dataset_summary <- result$summary
      dataset_summary$Dataset <- dataset
      overall_summary <- rbind(overall_summary, dataset_summary)
    }
  }
  
  cat("\n")
  cat("==================================================================================================\n")
  cat("OVERALL BATCH PROCESSING SUMMARY\n")
  cat("==================================================================================================\n")
  
  # Reshape summary for better view
  summary_wide <- overall_summary %>%
    select(Dataset, Model, MSE) %>%
    group_by(Dataset) %>%
    summarise(
      Best_Model = Model[which.min(MSE)],
      Best_MSE = min(MSE, na.rm = TRUE),
      Models_Tested = n(),
      .groups = "drop"
    )
  
  print(summary_wide)
  
  summary_file <- "batch_generalized_methods_summary.csv"
  write.csv(overall_summary, summary_file, row.names = FALSE)
  
  summary_wide_file <- "batch_generalized_methods_best_summary.csv"
  write.csv(summary_wide, summary_wide_file, row.names = FALSE)
  
  cat("\nFiles saved:\n")
  cat("- Individual model results for each dataset in main directory\n")
  cat("-", summary_file, "(detailed summary of all models)\n")
  cat("-", summary_wide_file, "(best model per dataset)\n")
  
  cat("\n==================================================================================================\n")
  cat("BATCH PROCESSING COMPLETE\n")
  cat("==================================================================================================\n")
  cat("\nTotal datasets processed:", length(all_results), "\n")
  cat("Total model runs:", nrow(overall_summary), "\n")
}