# My-Main-Projects
This repository contains a selection of my best and most advanced ML and Deep Learning projects

---

| **Project** | **Scores** |
|---|---|
| 1. Breast Cancer Classification using Computer Vision in Keras | **99% F1_score, 94% F1_score** |
| 2. Real-Time Car Price Recommendation Engine built on Spark Streaming in Databricks | **2350$ Mean Average Error** |
| 3. Customer Credit Score Classification using Random Forest in R | **98% Multivariate Area Under Curve** |
| 4. Text Sentiment Analysis and Topic Classification of Restaurant Reviews in Python | **85% Correlation, 61% F1_score** |
| 5. Human Obesity Classification using Boosting Classifier in Python | **95% F1_score** |

---

### **1. Breast Cancer Classification using Computer Vision in Keras** 
[dataset #1](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis) \
 \
We developed a deep learning model to assist doctors in classifying different types of breast cancer using microscopic images. By utilizing the DenseNet121 neural network and fine-tuning it with our specific dataset, we enhanced the model's accuracy in identifying cancerous cells. To address limited data and computational resources, we designed a custom real-time dataflow connected to Google Drive that automatically augments images based on predefined cancer classes, effectively rebalancing the dataset and optimizing the training process to run efficiently on Google Colab. Our model achieved a 99% F1-score for distinguishing between benign and malignant cancers and a 94% F1-score for more detailed classifications. This tool can support medical professionals in making more accurate and timely diagnoses, potentially improving patient care.

### **2. Real-Time Car Price Recommendation Engine built on Spark Streaming in Databricks** 
[dataset #2](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data) \
 \
Created a real-time, scalable system that can instantly determine the value of a used car by analyzing large amounts of market data from the past. This tool examines factors like the car’s model, age, mileage, condition, and region that the sales took place to provide accurate price estimates in real time. The model was trained on 180k of car listings, ensuring that the valuations are both timely and reliable. The system is fully designed on Spark so it handles and adapts to growing data, making it scalable for future needs. Model has been developped using Spark's Pipeline API, meaning after some small tweaks it can be easily saved and deployed on a processing cluster. It offers valuable insights for buyers and sellers, making it an excellent asset for companies in the automotive market. \
\
Full "Project Summary" file is available in a designated folder 


### **3. Customer Credit Score Classification using Random Forest in R** 
[dataset #3](https://www.kaggle.com/datasets/parisrohan/credit-score-classification) \
 \
This project aimed to develop a classification model to predict the credit score of clients applying for loans at a bank based on a set of provided features. To address class imbalance, SMOTE oversampling was employed to augment the minority class. The model was optimized using a grid search algorithm and its performance was assessed with the ROC (Receiver Operating Characteristic) Area Under the Curve (AUC) metric.

### **4. Text Sentiment Analysis and Topic Classification of Restaurant Reviews in Python** 
dataset #4 - data available inside the folder \
 \
\
\
\

### **5. Human Obesity Classification using Boosting Classifier in Python** 
dataset #5 - data available inside the folder \
\
This project focused on classifying the stages of human obesity based on a variety of features, including weight, height, age, gender, and lifestyle-related factors. To handle missing data, K-Nearest Neighbors (KNN) was used to impute numerical features, while the Iterative Imputer was employed to fill in missing categorical values. Among the models tested, Gradient Boosting emerged as the most accurate, achieving high F1 scores, with performance evaluated using Stratified K-Fold Cross-Validation.
