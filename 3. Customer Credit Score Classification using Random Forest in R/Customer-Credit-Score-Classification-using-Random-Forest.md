Customer Credit Score Classification using Random Forest in R
================
2024-13-06

``` r
# Cleaning the environment
rm(list=ls())

# Loading necessary libraries

library(rpart)
library(rpart.plot)
```

    ## Warning: package 'rpart.plot' was built under R version 4.3.3

``` r
library(randomForest)
```

    ## Warning: package 'randomForest' was built under R version 4.3.3

    ## randomForest 4.7-1.1

    ## Type rfNews() to see new features/changes/bug fixes.

``` r
library(ROCR)
```

    ## Warning: package 'ROCR' was built under R version 4.3.3

``` r
library(MASS)
library(caret)
```

    ## Warning: package 'caret' was built under R version 4.3.3

    ## Loading required package: ggplot2

    ## Warning: package 'ggplot2' was built under R version 4.3.3

    ## 
    ## Attaching package: 'ggplot2'

    ## The following object is masked from 'package:randomForest':
    ## 
    ##     margin

    ## Loading required package: lattice

``` r
library(doParallel)
```

    ## Warning: package 'doParallel' was built under R version 4.3.3

    ## Loading required package: foreach

    ## Warning: package 'foreach' was built under R version 4.3.3

    ## Loading required package: iterators

    ## Warning: package 'iterators' was built under R version 4.3.3

    ## Loading required package: parallel

``` r
# Loading data
file_path = "C:/Users/wojci/OneDrive - SGH/School_/SGH/IRD/preprocessed_V2.2_oversampling.csv"
dane <- read.csv(file_path)

# Converting columns to factor type
dane$Credit_Score <- as.factor(dane$Credit_Score)
dane$Credit_Mix <- as.factor(dane$Credit_Mix)
dane$Occupation <- as.factor(dane$Occupation)
dane$Payment_of_Min_Amount <- as.factor(dane$Payment_of_Min_Amount)
summary(dane)
```

    ##      Month            Age               Occupation    Annual_Income     
    ##  Min.   :1.000   Min.   :14.00   Lawyer      : 4095   Min.   :    7006  
    ##  1st Qu.:3.000   1st Qu.:24.00   Architect   : 4032   1st Qu.:   19380  
    ##  Median :5.000   Median :33.00   Teacher     : 3999   Median :   37486  
    ##  Mean   :4.537   Mean   :33.41   Scientist   : 3865   Mean   :  172144  
    ##  3rd Qu.:7.000   3rd Qu.:42.00   Entrepreneur: 3861   3rd Qu.:   73119  
    ##  Max.   :8.000   Max.   :95.00   Accountant  : 3843   Max.   :24198062  
    ##                                  (Other)     :33173                     
    ##  Monthly_Inhand_Salary Num_Bank_Accounts Num_Credit_Card   Interest_Rate    
    ##  Min.   :  303.6       Min.   :  -1.00   Min.   :   0.00   Min.   :   1.00  
    ##  1st Qu.: 1628.8       1st Qu.:   3.00   1st Qu.:   4.00   1st Qu.:   7.00  
    ##  Median : 3084.1       Median :   5.00   Median :   5.00   Median :  12.00  
    ##  Mean   : 4204.7       Mean   :  16.86   Mean   :  22.45   Mean   :  78.12  
    ##  3rd Qu.: 5971.8       3rd Qu.:   7.00   3rd Qu.:   7.00   3rd Qu.:  21.00  
    ##  Max.   :15204.6       Max.   :1798.00   Max.   :1499.00   Max.   :5797.00  
    ##                                                                             
    ##  Delay_from_due_date Num_of_Delayed_Payment Changed_Credit_Limit
    ##  Min.   :-5.00       Min.   :   0.00        Min.   :-6.44       
    ##  1st Qu.: 9.00       1st Qu.:   8.00        1st Qu.: 4.99       
    ##  Median :17.00       Median :  13.00        Median : 9.04       
    ##  Mean   :20.65       Mean   :  30.75        Mean   :10.04       
    ##  3rd Qu.:28.00       3rd Qu.:  18.00        3rd Qu.:14.11       
    ##  Max.   :67.00       Max.   :4397.00        Max.   :36.29       
    ##                                                                 
    ##  Num_Credit_Inquiries    Credit_Mix    Outstanding_Debt 
    ##  Min.   :   0.00      Bad     :13894   Min.   :   0.23  
    ##  1st Qu.:   3.00      Good    :21577   1st Qu.: 562.21  
    ##  Median :   6.00      Standard:21397   Median :1179.97  
    ##  Mean   :  28.32                       Mean   :1428.14  
    ##  3rd Qu.:   9.00                       3rd Qu.:1982.38  
    ##  Max.   :2597.00                       Max.   :4998.07  
    ##                                                         
    ##  Credit_Utilization_Ratio Credit_History_Age Payment_of_Min_Amount
    ##  Min.   :20.88            Min.   :  0.0      No :25521            
    ##  1st Qu.:28.11            1st Qu.:117.0      Yes:31347            
    ##  Median :32.30            Median :210.0                           
    ##  Mean   :32.27            Mean   :202.9                           
    ##  3rd Qu.:36.50            3rd Qu.:294.0                           
    ##  Max.   :49.56            Max.   :404.0                           
    ##                                                                   
    ##  Total_EMI_per_month Amount_invested_monthly Monthly_Balance    
    ##  Min.   :    4.46    Min.   :    0.00        Min.   :   0.1311  
    ##  1st Qu.:   41.35    1st Qu.:   73.45        1st Qu.: 266.9764  
    ##  Median :   79.01    Median :  135.24        Median : 331.0458  
    ##  Mean   : 1388.45    Mean   :  618.70        Mean   : 391.2514  
    ##  3rd Qu.:  171.58    3rd Qu.:  264.15        3rd Qu.: 456.8032  
    ##  Max.   :82204.00    Max.   :10000.00        Max.   :1552.9461  
    ##                                                                 
    ##  Count_Auto.Loan  Count_Credit.Builder.Loan Count_Personal.Loan
    ##  Min.   :0.0000   Min.   :0.000             Min.   :0.000      
    ##  1st Qu.:0.0000   1st Qu.:0.000             1st Qu.:0.000      
    ##  Median :0.0000   Median :0.000             Median :0.000      
    ##  Mean   :0.4072   Mean   :0.436             Mean   :0.416      
    ##  3rd Qu.:1.0000   3rd Qu.:1.000             3rd Qu.:1.000      
    ##  Max.   :4.0000   Max.   :4.000             Max.   :4.000      
    ##                                                                
    ##  Count_Home.Equity.Loan Count_Not.Specified Count_Mortgage.Loan
    ##  Min.   :0.0000         Min.   :0.0000      Min.   :0.0000     
    ##  1st Qu.:0.0000         1st Qu.:0.0000      1st Qu.:0.0000     
    ##  Median :0.0000         Median :0.0000      Median :0.0000     
    ##  Mean   :0.4252         Mean   :0.4239      Mean   :0.4253     
    ##  3rd Qu.:1.0000         3rd Qu.:1.0000      3rd Qu.:1.0000     
    ##  Max.   :5.0000         Max.   :4.0000      Max.   :5.0000     
    ##                                                                
    ##  Count_Student.Loan Count_Debt.Consolidation.Loan Count_Payday.Loan
    ##  Min.   :0.0000     Min.   :0.0000                Min.   :0.0000   
    ##  1st Qu.:0.0000     1st Qu.:0.0000                1st Qu.:0.0000   
    ##  Median :0.0000     Median :0.0000                Median :0.0000   
    ##  Mean   :0.4212     Mean   :0.4183                Mean   :0.4395   
    ##  3rd Qu.:1.0000     3rd Qu.:1.0000                3rd Qu.:1.0000   
    ##  Max.   :5.0000     Max.   :5.0000                Max.   :5.0000   
    ##                                                                    
    ##  Spent.Amount.Payment_Behaviour Value.Amount.Payment_Behaviour   Credit_Score  
    ##  Min.   :0.0000                 Min.   :0.0000                 Good    :18956  
    ##  1st Qu.:0.0000                 1st Qu.:0.0000                 Poor    :18956  
    ##  Median :0.0000                 Median :1.0000                 Standard:18956  
    ##  Mean   :0.4696                 Mean   :0.8723                                 
    ##  3rd Qu.:1.0000                 3rd Qu.:2.0000                                 
    ##  Max.   :1.0000                 Max.   :2.0000                                 
    ## 

``` r
# Splitting data into training and test sets
set.seed(40)
test_prop <- 0.30
test.index <- runif(nrow(dane)) < test_prop
test <- dane[test.index, ]
train <- dane[!test.index, ]

# Training decision tree
tree <- rpart(Credit_Score ~ ., 
              data = train, 
              method = "class")
rpart.plot(tree, under = FALSE, fallen.leaves = TRUE, tweak = 0.8)
```

![](Customer-Credit-Score-Classification-using-Random-Forest_files/figure-gfm/unnamed-chunk-2-1.png)<!-- -->

``` r
# Training deeper decision tree
tree.deeper <- rpart(Credit_Score ~ ., 
                     data = train, 
                     method = "class", 
                     control = list(cp = 0.005))
rpart.plot(tree.deeper, under = FALSE, fallen.leaves = TRUE, tweak = 0.8)
```

![](Customer-Credit-Score-Classification-using-Random-Forest_files/figure-gfm/unnamed-chunk-2-2.png)<!-- -->

``` r
### Hyperparameters from tuning were substituted here, before acc=88.7%, after acc=88.9%
forest <- randomForest(Credit_Score ~ ., 
                       data = train, 
                       ntree = 150, 
                       nodesize = 1, 
                       mtry = 8)


# Left image is a Simple Tree
# Right image is a Pruned Tree
```

``` r
# Extract variable importance
var_imp <- importance(forest)

# Create a data frame for plotting
df_var_imp <- data.frame(Variable = rownames(var_imp), Importance = var_imp[, 1])

# Sort by importance
df_var_imp <- df_var_imp[order(df_var_imp$Importance, decreasing = TRUE), ]

# Create the plot
ggplot(df_var_imp, aes(x = reorder(Variable, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Variable Importance", 
 x = "Variable", y = "Importance") +
  theme_minimal()
```

![](Customer-Credit-Score-Classification-using-Random-Forest_files/figure-gfm/unnamed-chunk-3-1.png)<!-- -->

``` r
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
```

    ## Model: tree 
    ## $accuracy
    ## [1] 0.7201735
    ## 
    ## $precision
    ##      Good      Poor  Standard 
    ## 0.8284253 0.7899757 0.5393238 
    ## 
    ## $sensitivity
    ##      Good      Poor  Standard 
    ## 0.7336040 0.7142857 0.7088400 
    ## 
    ## $specificity
    ##      Good      Poor  Standard 
    ## 0.8284253 0.7899757 0.5393238 
    ## 
    ## Model: tree.deeper 
    ## $accuracy
    ## [1] 0.7265639
    ## 
    ## $precision
    ##      Good      Poor  Standard 
    ## 0.8211956 0.8159903 0.5393238 
    ## 
    ## $sensitivity
    ##      Good      Poor  Standard 
    ## 0.7536818 0.7126628 0.7088400 
    ## 
    ## $specificity
    ##      Good      Poor  Standard 
    ## 0.8211956 0.8159903 0.5393238 
    ## 
    ## Model: forest 
    ## $accuracy
    ## [1] 0.8897227
    ## 
    ## $precision
    ##      Good      Poor  Standard 
    ## 0.9836008 0.9287201 0.7549822 
    ## 
    ## $sensitivity
    ##      Good      Poor  Standard 
    ## 0.8985180 0.8649653 0.9109060 
    ## 
    ## $specificity
    ##      Good      Poor  Standard 
    ## 0.9836008 0.9287201 0.7549822

``` r
# Creating confusion matrices
confusion_matrices <- list(
  tree = table(predict(tree, newdata = test, type = "class"), test$Credit_Score),
  tree_deeper = table(predict(tree.deeper, newdata = test, type = "class"), test$Credit_Score),
  forest = table(predict(forest, newdata = test, type = "class"), test$Credit_Score)
)

# Displaying confusion matrices with better formatting
for (model_name in names(confusion_matrices)) {
  cat("Model:", model_name, "\n")
  print(confusionMatrix(confusion_matrices[[model_name]]))
  cat("\n")
}
```

    ## Model: tree 
    ## Confusion Matrix and Statistics
    ## 
    ##           
    ##            Good Poor Standard
    ##   Good     4698  745      961
    ##   Poor      194 4555     1628
    ##   Standard  779  466     3031
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.7202          
    ##                  95% CI : (0.7134, 0.7269)
    ##     No Information Rate : 0.338           
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.58            
    ##                                           
    ##  Mcnemar's Test P-Value : < 2.2e-16       
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: Good Class: Poor Class: Standard
    ## Sensitivity               0.8284      0.7900          0.5393
    ## Specificity               0.8502      0.8386          0.8911
    ## Pos Pred Value            0.7336      0.7143          0.7088
    ## Neg Pred Value            0.9087      0.8866          0.7974
    ## Prevalence                0.3325      0.3380          0.3295
    ## Detection Rate            0.2754      0.2670          0.1777
    ## Detection Prevalence      0.3754      0.3739          0.2507
    ## Balanced Accuracy         0.8393      0.8143          0.7152
    ## 
    ## Model: tree_deeper 
    ## Confusion Matrix and Statistics
    ## 
    ##           
    ##            Good Poor Standard
    ##   Good     4657  595      927
    ##   Poor      235 4705     1662
    ##   Standard  779  466     3031
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.7266          
    ##                  95% CI : (0.7198, 0.7332)
    ##     No Information Rate : 0.338           
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.5895          
    ##                                           
    ##  Mcnemar's Test P-Value : < 2.2e-16       
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: Good Class: Poor Class: Standard
    ## Sensitivity               0.8212      0.8160          0.5393
    ## Specificity               0.8663      0.8320          0.8911
    ## Pos Pred Value            0.7537      0.7127          0.7088
    ## Neg Pred Value            0.9068      0.8985          0.7974
    ## Prevalence                0.3325      0.3380          0.3295
    ## Detection Rate            0.2730      0.2758          0.1777
    ## Detection Prevalence      0.3623      0.3871          0.2507
    ## Balanced Accuracy         0.8438      0.8240          0.7152
    ## 
    ## Model: forest 
    ## Confusion Matrix and Statistics
    ## 
    ##           
    ##            Good Poor Standard
    ##   Good     5578   86      547
    ##   Poor        5 5354      833
    ##   Standard   88  326     4240
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.8895          
    ##                  95% CI : (0.8847, 0.8942)
    ##     No Information Rate : 0.338           
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.8341          
    ##                                           
    ##  Mcnemar's Test P-Value : < 2.2e-16       
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: Good Class: Poor Class: Standard
    ## Sensitivity               0.9836      0.9285          0.7544
    ## Specificity               0.9444      0.9258          0.9638
    ## Pos Pred Value            0.8981      0.8647          0.9110
    ## Neg Pred Value            0.9914      0.9621          0.8887
    ## Prevalence                0.3325      0.3380          0.3295
    ## Detection Rate            0.3270      0.3139          0.2486
    ## Detection Prevalence      0.3641      0.3630          0.2728
    ## Balanced Accuracy         0.9640      0.9272          0.8591

``` r
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
```

    ## $ntree
    ## [1] 150
    ## 
    ## $nodesize
    ## [1] 1
    ## 
    ## $mtry
    ## [1] 4

``` r
# Extract variable importance
var_imp <- importance(best_model$finalModel)

# Create a data frame for plotting
df_var_imp <- data.frame(Variable = rownames(var_imp), Importance = var_imp[, 1])

# Sort by importance
df_var_imp <- df_var_imp[order(df_var_imp$Importance, decreasing = TRUE), ]

# Plot
ggplot(df_var_imp, aes(x = reorder(Variable, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Variable Importance", 
 x = "Variable", y = "Importance") +
  theme_minimal() +
  theme(axis.text.y = element_text(size = 6))
```

![](Customer-Credit-Score-Classification-using-Random-Forest_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->

``` r
# Predictions on the test set
predicted_classes <- predict(best_model, newdata = test)

# Confusion matrix
conf_matrix <- table(predicted_classes, test$Credit_Score)

# Printing confusion matrix
print(conf_matrix)
```

    ##                  
    ## predicted_classes Good Poor Standard
    ##          Good     5589  132      640
    ##          Poor        4 5328      875
    ##          Standard   78  306     4105

``` r
# Model evaluation using confusion matrix
evaluation_results <- EvaluateModel(conf_matrix)
print(evaluation_results)
```

    ## $accuracy
    ## [1] 0.8806941
    ## 
    ## $precision
    ##      Good      Poor  Standard 
    ## 0.9855405 0.9240375 0.7304270 
    ## 
    ## $sensitivity
    ##      Good      Poor  Standard 
    ## 0.8786354 0.8583857 0.9144576 
    ## 
    ## $specificity
    ##      Good      Poor  Standard 
    ## 0.9855405 0.9240375 0.7304270

``` r
# Multiclass ROC curve and LIFT curves

#install.packages("pROC")
library(pROC)
```

    ## Warning: package 'pROC' was built under R version 4.3.3

    ## Type 'citation("pROC")' for a citation.

    ## 
    ## Attaching package: 'pROC'

    ## The following objects are masked from 'package:stats':
    ## 
    ##     cov, smooth, var

``` r
# Predicting probabilities and classes on the test set
prob_pred <- predict(forest, newdata = test, type = "prob")
predicted_classes <- predict(forest, newdata = test, type = "class")

# Function to plot ROC curves for multiclass problem
plot_multiclass_roc <- function(test_labels, prob_pred) {
  multiclass_roc <- multiclass.roc(test_labels, prob_pred)
  roc_list <- multiclass_roc$rocs

  # Create a data frame to store ROC data
  roc_data <- data.frame()
  for (i in 1:length(roc_list)) {
    roc_curve <- roc_list[[i]]
    roc_data <- rbind(roc_data, data.frame(
      Specificity = 1 - roc_curve[[1]]$specificities,
      Sensitivity = roc_curve[[1]]$sensitivities,
      Class = levels(test_labels)[i]
    ))
  }

  # Create the plot
  ggplot(roc_data, aes(x = Specificity, y = Sensitivity, color = Class)) +
    geom_line(linewidth = 1) +
    geom_abline(intercept = 1, slope = -1, linetype = "dashed") +
    labs(title = paste("Multiclass ROC Curve (AUC:", round(multiclass_roc$auc, 3), ")"),
         x = "1 - Specificity", y = "Sensitivity") +
    theme_minimal() +
    theme(legend.position = "bottom")
}

# Calling the function for test data
plot_multiclass_roc(test$Credit_Score, prob_pred)
```

![](Customer-Credit-Score-Classification-using-Random-Forest_files/figure-gfm/unnamed-chunk-9-1.png)<!-- -->

``` r
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
```

![](Customer-Credit-Score-Classification-using-Random-Forest_files/figure-gfm/unnamed-chunk-10-1.png)<!-- -->

    ##                  
    ## predicted_classes Good Poor Standard
    ##          Good     5578   85      546
    ##          Poor        5 5354      828
    ##          Standard   88  327     4246
    ## $accuracy
    ## [1] 0.8898399
    ## 
    ## $precision
    ##      Good      Poor  Standard 
    ## 0.9836008 0.9285467 0.7555160 
    ## 
    ## $sensitivity
    ##      Good      Poor  Standard 
    ## 0.8983733 0.8653629 0.9109633 
    ## 
    ## $specificity
    ##      Good      Poor  Standard 
    ## 0.9836008 0.9285467 0.7555160

``` r
# ROC for different models

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
```

    ## Setting levels: control = 0, case = 1

    ## Setting direction: controls < cases

    ## Setting levels: control = 0, case = 1

    ## Setting direction: controls < cases

    ## Setting levels: control = 0, case = 1

    ## Setting direction: controls < cases

![](Customer-Credit-Score-Classification-using-Random-Forest_files/figure-gfm/unnamed-chunk-11-1.png)<!-- -->

``` r
plot_model_roc_for_class(test$Credit_Score, prob_pred_tree, prob_pred_tree_deeper, prob_pred_forest, "Standard", colors)
```

    ## Setting levels: control = 0, case = 1

    ## Setting direction: controls < cases

    ## Setting levels: control = 0, case = 1

    ## Setting direction: controls < cases

    ## Setting levels: control = 0, case = 1

    ## Setting direction: controls < cases

![](Customer-Credit-Score-Classification-using-Random-Forest_files/figure-gfm/unnamed-chunk-12-1.png)<!-- -->

``` r
plot_model_roc_for_class(test$Credit_Score, prob_pred_tree, prob_pred_tree_deeper, prob_pred_forest, "Poor", colors)
```

    ## Setting levels: control = 0, case = 1

    ## Setting direction: controls < cases

    ## Setting levels: control = 0, case = 1

    ## Setting direction: controls < cases

    ## Setting levels: control = 0, case = 1

    ## Setting direction: controls < cases

![](Customer-Credit-Score-Classification-using-Random-Forest_files/figure-gfm/unnamed-chunk-13-1.png)<!-- -->
