# Use-of-advanced-regression-techniques-for-home-value-prediction

This is regarding a kaggle competition which goes under the name Zillow Prize: Zillow’s Home Value Prediction (Zestimate).I took part in this competition and submitted several solutions using several techniques and implementations.I will give a brief description of the implementation.

The problem was to create a machine learning model using regression techniques to determine the error of properties in future periods in time.To be specific we had to predict Oct 16,Nov 16,Dec 16,Oct 17,Nov 17,Dec 17.

The project was executed along the following phases:

* Data analysis and visualization
* Data pre-processing 
* Feature engineering
* Model building and parameter tuning

Data analysis and visualization

Under this phase we analyzed the datatypes of the dataset,missing values in the dataset,categorical values in the dataset,statistical parameters relating to the dataset(mean ,median,standard deviation,1st quartile,3rd quartile).further we had to identify which variables can be taken as features and what should be the target variable.

Next in the visualization part we looked at the distribution of each variable in the dataset.Both the features and targets separately.we also analysed how the target variable changed with each feature.Performing clustering was also a done to gather insights from data.Finally the use of boxplots to identify outliers were plotted.

Data pre-processing

In this phase we performed various preprocessing.Such as data standardization(scaling data to be part of a gaussian distribution with a mean of zero and a standard deviation of one),Imputing missing values(replacing missing values in each column by the mean,meadian and standard deviation of values in that column) and handling categorical values(replacing each category by the mean of targets values for each category,Label encoding,frequency based class assigning,clustering based label assigning and performing one-hot encoding on those labels).

Feature Engineering

Here we mainly create new features with the use of existing features to uncover hidden predictive inferences in data.This makes the model more powerful.Features were engineered by multiplying ,dividing ,adding and subtracting selected existing features.such engineered features are as follows.
 
Engineered Features:

1.Day of transaction

2.Month of transaction

3.Month to day ratio

4.Missing value count in each row

5.How old the property was(in number of years)

6.Calculated(estimated) vs actual finish living area ratio

7.Proportion of living area of living area against total property area(based on estimated completion)

8.Proportion of living area of living area against total property area(based on actual completion)

9.Extra space(when removed space allocated to house from total space.this is based on estimated values)

10.Extra space (based on actual values)

11.Extra number of rooms(derived from deducting bathrooms and bedrooms from total number of rooms)

12.Value of house over value of land ratio

13.Whether a property at least has an A/C,garage & pool/hub.

14.Addition of longitude and latitude

15.Multiplication of longitude and latitude(location)

16.Rounded values of location

17.Rounded values for longitude and latitude

18.Ratio for property tax over land tax

19.Multiplication of property tax into land tax

20.Polynomials of the year for which unpaid taxes were due

21.For how long tax was due

22.Number of properties in each zip

23.Number of properties in each county

24.Number of properties in each city

25.Polynomials of property value

26.Mean value of properties grouped by city

27.Land value deviation from average value

28.Actual living area proportion from total land area

29.Proportion of land allocated for garages out of total land

30.Proportion of land allocated for pools out of total land

31.Ratio between value of land and total area.

32.Tax paid per square feet of land.

Model Building and parameter tuning

We used few techniques to deal with this problem.Those were Neural networks and ensembling techniques(bagging,boosting and stacking).For bagging we used random forest algorithm,for boosting we used xgboost and catboost.for stacking we used the two boosting implemantations as base learners and a decision tree regressor as the meta learner.The neural network was a multi-layer perceptron.
Stacking performed very well while results from catboost were also promising.The other implementations gave acceptable results.but not thesignificant improvements to the evaluation metric.

In terms of parameter tuning neural network hyper parameters as well as tree parameters for other implementations were configured to give decent results. 



