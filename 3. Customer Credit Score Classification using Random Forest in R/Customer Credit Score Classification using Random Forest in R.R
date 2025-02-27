################################################################################
# Cleaning the environment
rm(list=ls())

# Loading data
file_path = "C:/Users/wojci/OneDrive - SGH/School_/SGH/IRD/preprocessed_V2.2_oversampling.csv"
dane <- read.csv(file_path)

# Converting columns to factor type
dane$Credit_Score <- as.factor(dane$Credit_Score)
dane$Credit_Mix <- as.factor(dane$Credit_Mix)
dane$Occupation <- as.factor(dane$Occupation)
dane$Payment_of_Min_Amount <- as.factor(dane$Payment_of_Min_Amount)

# Splitting data into training and test sets
set.seed(40)
test_prop <- 0.30
test.index <- runif(nrow(dane)) < test_prop
test <- dane[test.index, ]
train <- dane[!test.index, ]

# Loading necessary libraries
library(rpart)
library(rpart.plot)
library(randomForest)
library(ROCR)
library(MASS)

# Training decision tree
tree <- rpart(Credit_Score ~ ., 
              data = train, 
              method = "class")
rpart.plot(tree, under = FALSE, fallen.leaves = TRUE, tweak = 0.8)

# Training deeper decision tree
tree.deeper <- rpart(Credit_Score ~ ., 
                     data = train, 
                     method = "class", 
                     control = list(cp = 0.005))
rpart.plot(tree.deeper, under = FALSE, fallen.leaves = TRUE, tweak = 0.8)

### Hyperparameters from tuning were substituted here, before acc=88.7%, after acc=88.9%
forest <- randomForest(Credit_Score ~ ., 
                       data = train, 
                       ntree = 150, 
                       nodesize = 1, 
                       mtry = 8)
varImpPlot(forest)
print(forest$importance)

# Creating confusion matrices
CM <- list()
CM[["tree"]] <- table(predict(tree, newdata = test, type = "class"), test$Credit_Score)
CM[["tree.deeper"]] <- table(predict(tree.deeper, newdata = test, type = "class"), test$Credit_Score)
CM[["forest"]] <- table(predict(forest, newdata = test, type = "class"), test$Credit_Score)

# Defining model evaluation function
EvaluateModel <- function(classif_mx) {
  true_positive <- diag(classif_mx)
  true_negative <- sum(classif_mx) - sum(rowSums(classif_mx)) + true_positive
  condition_positive <- rowSums(classif_mx)
  condition_negative <- colSums(classif_mx)
  predicted_positive <- colSums(classif_mx)
  predicted_negative <- rowSums(classif_mx)
  
  accuracy <- sum(true_positive) / sum(classif_mx)
  precision <- true_positive / predicted_positive
  sensitivity <- true_positive / condition_positive 
  specificity <- true_negative / condition_negative
  
  return(list(
    accuracy = accuracy,
    precision = precision, 
    sensitivity = sensitivity, 
    specificity = specificity
  ))
}

# Evaluating results for each confusion matrix
for (i in seq_along(CM)) {
  cat("Model:", names(CM)[i], "\n")
  evaluation_results <- EvaluateModel(CM[[i]])
  print(evaluation_results)
}

########################################################################################################
# Installing and loading libraries for tuning using parallel processing
########################################################################################################

library(caret)
library(doParallel)

# Setting up parallel processing
num_cores <- detectCores() - 1  # Using all cores except one
cl <- makeCluster(num_cores)
registerDoParallel(cl)

# Defining control for cross-validation
train_control <- trainControl(method = "cv", number = 5, allowParallel = TRUE)

# Defining hyperparameter search grid for random forest
tune_grid_rf <- expand.grid(mtry = c(4)) # c(4,8)

# Search loop for ntree and nodesize
best_model <- NULL
best_accuracy <- 0
best_params <- list()

for (ntree in c(150)) { # c(50, 100, 150)
  for (nodesize in c(1)) { # c(1, 5, 10)
    set.seed(123)
    model <- train(
      Credit_Score ~ ., 
      data = train, 
      method = "rf", 
      trControl = train_control, 
      tuneGrid = tune_grid_rf,
      ntree = ntree,
      nodesize = nodesize
    )
    
    # Getting the best accuracy from model results
    model_accuracy <- model$results$Accuracy[model$results$mtry == model$bestTune$mtry]
    
    # Checking if accuracy is not NA and is greater than the current best accuracy
    if (!is.na(model_accuracy) && model_accuracy > best_accuracy) {
      best_accuracy <- model_accuracy
      best_model <- model
      best_params <- list(ntree = ntree, nodesize = nodesize, mtry = model$bestTune$mtry)
    }
  }
}

# Stopping parallel processing
stopCluster(cl)

# Printing best parameters
print(best_params)

# Variable importance plot for the best random forest model
varImpPlot(best_model$finalModel)
print(best_model$finalModel$importance)

# Predictions on the test set
predicted_classes <- predict(best_model, newdata = test)

# Confusion matrix
conf_matrix <- table(predicted_classes, test$Credit_Score)

# Printing confusion matrix
print(conf_matrix)

# Model evaluation using confusion matrix
evaluation_results <- EvaluateModel(conf_matrix)
print(evaluation_results)

########################################################################################################
# Multiclass ROC curve and LIFT curves
########################################################################################################
dev.off()
#install.packages("pROC")
library(pROC)

# Predicting probabilities and classes on the test set
prob_pred <- predict(forest, newdata = test, type = "prob")
predicted_classes <- predict(forest, newdata = test, type = "class")

# Function to plot ROC curves for multiclass problem
plot_multiclass_roc <- function(test_labels, prob_pred) {
  multiclass_roc <- multiclass.roc(test_labels, prob_pred)
  roc_list <- multiclass_roc$rocs
  colors <- c("green3", "red", "blue")
  
  # Ensuring X-axis is from 0 to 1 and Y-axis from 0 to 1
  plot(1 - roc_list[[1]][[1]]$specificities, roc_list[[1]][[1]]$sensitivities, 
       col = colors[1], main = "Multiclass ROC Curve", 
       xlab = "1 - Specificity", ylab = "Sensitivity", 
       xlim = c(0, 1), ylim = c(0, 1), type = "l", lwd = 2)
  
  for (i in 1:length(roc_list)) {
    roc_curve <- roc_list[[i]]
    for (j in seq(1, length(roc_curve), by = 2)) {
      lines(1 - roc_curve[[j]]$specificities, roc_curve[[j]]$sensitivities, col = colors[i], lwd = 2)
    }
  }
  
  legend("bottomright", legend = levels(test_labels), col = colors[1:length(roc_list)], lwd = 2)
  print(multiclass_roc$auc)
}

# Calling the function for test data
plot_multiclass_roc(test$Credit_Score, prob_pred)

# Function to plot LIFT curves
plot_lift_curve <- function(test_labels, prob_pred) {
  # Creating an empty plot
  plot(1, type = "n", xlab = "Sample Percent", ylab = "Lift", 
       xlim = c(1, nrow(prob_pred)), ylim = c(0, 3), 
       main = "Multiclass Lift Curve")
  
  # List of colors for plots
  colors <- c("green3", "red", "blue")
  
  # Iterating through all columns (classes)
  for (i in 1:ncol(prob_pred)) {
    class <- colnames(prob_pred)[i]
    data <- data.frame(obs = (test_labels == class), prob = prob_pred[, i])
    data <- data[order(-data$prob), ]
    data$cum_gains <- cumsum(data$obs) / sum(data$obs)
    data$lift <- data$cum_gains / (1:nrow(data) / nrow(data))
    
    # Adding lift curve to the plot
    lines(data$lift, col = colors[i], lwd = 2)
  }
  
  # Adding legend
  legend("topright", legend = colnames(prob_pred), col = colors[1:ncol(prob_pred)], lwd = 2)
}

# Function to evaluate and plot model metrics
evaluate_and_plot_model <- function(forest, test_data) {
  prob_pred <- predict(forest, newdata = test_data, type = "prob")
  predicted_classes <- predict(forest, newdata = test_data, type = "class")
  
  # Plotting ROC curves
  plot_multiclass_roc(test_data$Credit_Score, prob_pred)
  
  # Plotting Lift curves
  plot_lift_curve(test_data$Credit_Score, prob_pred)
  
  # Confusion matrix
  conf_matrix <- table(predicted_classes, test_data$Credit_Score)
  print(conf_matrix)
  
  # Defining evaluation function
  EvaluateModel <- function(classif_mx) {
    true_positive <- diag(classif_mx)
    true_negative <- sum(classif_mx) - sum(rowSums(classif_mx)) + true_positive
    condition_positive <- rowSums(classif_mx)
    condition_negative <- colSums(classif_mx)
    predicted_positive <- colSums(classif_mx)
    predicted_negative <- rowSums(classif_mx)
    
    accuracy <- sum(true_positive) / sum(classif_mx)
    precision <- true_positive / predicted_positive
    sensitivity <- true_positive / condition_positive 
    specificity <- true_negative / condition_negative
    return(list(accuracy = accuracy,
                precision = precision, 
                sensitivity = sensitivity, 
                specificity = specificity))
  }
  
  # Evaluating model
  evaluation_results <- EvaluateModel(conf_matrix)
  print(evaluation_results)
}

# Evaluating and plotting model metrics
evaluate_and_plot_model(forest, test)

################################################################################
# ROC for different models
################################################################################

# Predict probabilities for each model
prob_pred_tree <- predict(tree, newdata = test, type = "prob")
prob_pred_tree_deeper <- predict(tree.deeper, newdata = test, type = "prob")
prob_pred_forest <- predict(forest, newdata = test, type = "prob")

# Function to plot multiclass ROC curve for a given class
plot_model_roc_for_class <- function(test_labels, prob_pred_tree, prob_pred_tree_deeper, prob_pred_forest, class_name, colors) {
  # Preparing data for the given class
  class_labels <- ifelse(test_labels == class_name, 1, 0)
  
  # ROC curves
  roc_tree <- roc(class_labels, prob_pred_tree[, class_name])
  roc_tree_deeper <- roc(class_labels, prob_pred_tree_deeper[, class_name])
  roc_forest <- roc(class_labels, prob_pred_forest[, class_name])
  
  # Plot
  plot(1 - roc_tree$specificities, roc_tree$sensitivities, col = colors[1], main = paste("ROC Curve for class:", class_name), xlab = "1 - Specificity", ylab = "Sensitivity", type = "l", lwd = 2)
  lines(1 - roc_tree_deeper$specificities, roc_tree_deeper$sensitivities, col = colors[2], lwd = 2)
  lines(1 - roc_forest$specificities, roc_forest$sensitivities, col = colors[3], lwd = 2)
  
  # AUC values
  auc_tree <- auc(roc_tree)
  auc_tree_deeper <- auc(roc_tree_deeper)
  auc_forest <- auc(roc_forest)
  
  # Add legend
  legend("bottomright", legend = c(paste("Tree  AUC =", round(auc_tree, 4)),
                                   paste("Tree Deeper AUC =", round(auc_tree_deeper, 4)),
                                   paste("Random Forest AUC =", round(auc_forest, 4))),
         col = colors, lwd = 2)
}

# Colors for models
colors <- c("red", "green", "blue")

# Plotting ROC curves for each class on separate plots
# Plot for class "Good"
plot_model_roc_for_class(test$Credit_Score, prob_pred_tree, prob_pred_tree_deeper, prob_pred_forest, "Good", colors)
plot_model_roc_for_class(test$Credit_Score, prob_pred_tree, prob_pred_tree_deeper, prob_pred_forest, "Standard", colors)
plot_model_roc_for_class(test$Credit_Score, prob_pred_tree, prob_pred_tree_deeper, prob_pred_forest, "Poor", colors)