# Clear environment
rm(list=ls())

# Load dataset
dane <- read.csv("C:/Users/wojci/OneDrive - SGH/School_/SGH/IRD/preprocessed_V2.2_oversampling.csv")

# Convert columns to factors
dane$Credit_Score = as.factor(dane$Credit_Score)
dane$Credit_Mix = as.factor(dane$Credit_Mix)
dane$Occupation = as.factor(dane$Occupation)
dane$Payment_of_Min_Amount = as.factor(dane$Payment_of_Min_Amount)

# Split dataset into train and test sets, 30 - 70 split
set.seed(40)
test_prop <- 0.30
test.index <- (runif(nrow(dane)) < test_prop)
test <- dane[test.index, ]
train <- dane[!test.index, ]

# Load necessary libraries
library(randomForest)
library(pROC)
library(caret)

# Train the Random Forest model
forest <- randomForest(Credit_Score ~ ., 
                       data = train, 
                       ntree = 150, 
                       nodesize = 1, 
                       mtry = 8)

varImpPlot(forest)
print(forest$importance)

# Predict probabilities and classes on the test set
prob_pred <- predict(forest, newdata = test, type = "prob")
predicted_classes <- predict(forest, newdata = test, type = "class")

# Function to plot Multiclass ROC Curve
plot_multiclass_roc <- function(test_labels, prob_pred) {
  multiclass_roc <- multiclass.roc(test_labels, prob_pred)
  roc_list <- multiclass_roc$rocs
  plot(roc_list[[1]][[1]], col = "red", main = "Multiclass ROC Curve")
  for (i in 1:length(roc_list)) {
    roc_curve <- roc_list[[i]]
    for (j in seq(1, length(roc_curve), by = 2)) {
      plot(roc_curve[[j]], add = TRUE, col = i)
    }
  }
  legend("bottomright", legend = levels(test_labels), col = 1:length(roc_list), lwd = 2)
  print(multiclass_roc$auc)
}

# Function to compute and plot Lift Curve for each class
plot_lift_curve <- function(test_labels, prob_pred) {
  par(mfrow = c(2, 2))  # Adjust layout for multiple plots
  for (i in 1:ncol(prob_pred)) {
    class <- colnames(prob_pred)[i]
    data <- data.frame(obs = (test_labels == class), prob = prob_pred[, i])
    data <- data[order(-data$prob), ]
    data$cum_gains <- cumsum(data$obs) / sum(data$obs)
    data$lift <- data$cum_gains / (1:nrow(data) / nrow(data))
    plot(data$lift, type = "l", col = i, lwd = 2, main = paste("Lift Curve -", class), 
         xlab = "Percentage of Sample", ylab = "Lift", ylim = c(0, max(data$lift)))
    abline(h = 1, col = "gray", lty = 2)
  }
  par(mfrow = c(1, 1))  # Reset layout
}

# Function to evaluate and plot model metrics
evaluate_and_plot_model <- function(forest, test_data) {
  prob_pred <- predict(forest, newdata = test_data, type = "prob")
  predicted_classes <- predict(forest, newdata = test_data, type = "class")
  
  # Plot ROC Curves
  plot_multiclass_roc(test_data$Credit_Score, prob_pred)
  
  # Plot Lift Curve
  plot_lift_curve(test_data$Credit_Score, prob_pred)
  
  # Confusion Matrix
  conf_matrix <- table(predicted_classes, test_data$Credit_Score)
  print(conf_matrix)
  
  # Define Evaluation Function
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
  
  # Evaluate the model
  evaluation_results <- EvaluateModel(conf_matrix)
  print(evaluation_results)
}

# Evaluate and plot model metrics
evaluate_and_plot_model(forest, test)

