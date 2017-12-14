# Use-of-advanced-regression-techniques-for-home-value-prediction

This is regarding a kaggle competition which goes under the name Zillow Prize: Zillow’s Home Value Prediction (Zestimate).I took part in this competition and submitted several solutions using several techniques and implementations.I will give a brief description of the implementation.

The problem was to create a machine learning model using regression techniques to determine the error of properties in future periods in time.To be specific we had to predict Oct 16,Nov 16,Dec 16,Oct 17,Nov 17,Dec 17.

The implementation was done in two methods.As a time series forecasting and a constant trend based method.All the implementations can be found above.

The project was executed along the following phases:

* Data analysis and visualization
* Data pre-processing 
* Feature engineering
* Model building and parameter tuning

## Data analysis and visualization

Under this phase we analyzed the datatypes of the dataset,missing values in the dataset,categorical values in the dataset,statistical parameters relating to the dataset(mean ,median,standard deviation,1st quartile,3rd quartile).further we had to identify which variables can be taken as features and what should be the target variable.

Next in the visualization part we looked at the distribution of each variable in the dataset.Both the features and targets separately.we also analysed how the target variable changed with each feature.Performing clustering was also a done to gather insights from data.Finally the use of boxplots to identify outliers were plotted.

## Data pre-processing

In this phase we performed various preprocessing.Such as data standardization(scaling data to be part of a gaussian distribution with a mean of zero and a standard deviation of one),Imputing missing values(replacing missing values in each column by the mean,meadian and standard deviation of values in that column) and handling categorical values(replacing each category by the mean of targets values for each category,Label encoding,frequency based class assigning,clustering based label assigning and performing one-hot encoding on those labels).

## Feature Engineering

Here we mainly create new features with the use of existing features to uncover hidden predictive inferences in data.This makes the model more powerful.Features were engineered by multiplying ,dividing ,adding and subtracting selected existing features.such engineered features are as follows.
 
The complete list of engineered features can be found above under the following file name "FeaturesEngineered.txt".

## Model Building and parameter tuning

We used few techniques to deal with this problem.Those were Neural networks and ensembling techniques(bagging,boosting and stacking).For bagging we used random forest algorithm,for boosting we used xgboost and catboost.for stacking we used the two boosting implemantations as base learners and a decision tree regressor as the meta learner.The neural network was a multi-layer perceptron.
Stacking performed very well while results from catboost were also promising.The other implementations gave acceptable results.but not thesignificant improvements to the evaluation metric.

In terms of parameter tuning neural network hyper parameters as well as tree parameters for other implementations were configured to give decent results. 

Technology Used

*Keras-Implementing neural network
*CatBoost-Implementing boosting trees
*XGBoost-Implementing boosting trees
*Pandas-Data manipulation
*Numpy-data handling
*Sklearn-Building machine learning models
*matplotlib-Plotting results
*joblib-Saving models


_File instruction_:

EngineeredFeatures.txt-This file holds the engineered features we used.
NeuralNetImplementation.py-This script has the neural network implementation constructed using Keras.

RandomForestImplementation.py-This script holds the implementation of the bagging technique.

ReadMe.md-contains description of the project execution and file contents.

Stacking.py-Stacking technique implementation

Time-SeriesForcatingImplementation.py-This is the implementation done considering the the problem to be a time series modelling.

TrendBasedImplementation.py-This is where the problem was considered to be a a repetition of previous months values.





### References:

https://www.kaggle.com/c/zillow-prize-1

https://docs.google.com/a/wso2.com/document/d/1aV6lEkSBcXAW4G1Xyj5lcNbQBqBGzHae04ev4IVE75U/edit?usp=sharing

https://www.kaggle.com/tharindraparanagama/eda-py
