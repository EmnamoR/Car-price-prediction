# Car-price-prediction
This projects shows how to apply linear regression to make used car prices prediction. 
First we went threw and EDA(Exploratory Data Analysis), then feature selection then modeling and finally evaluating our model using difirent metrics and diffirent techniques.
## Part 1 – Data Cleaning and EDA
### Data Cleaning
We need to preprocess the data to transform it to a useful one and also to reduce its size, so that it
becomes easier to analyze.
In our case we need to:
• built_year: Remove unwanted string from built year _year and convert it to int type.
• first_registration_date: extract month and year for analysis
• delete “??” from those columns
manufacturer 3
main_type 3
built_year 3
first_registration_date 2
evaluation_datetime 3
fuel 2
gear 1
ac 4
gps 2
color 1
• There is 4 rows with car_height equal to 0, we can delete them or use the average ( in our
case we deleted them)
#### Unwanted columns
• has_all_wheel_drive: drop this column because only 314 cells are not empty from a total
of 2121 cell.
• gps: contains 883 no empty entries from a total of 2121. we kept it for later (EDA) to see if
it has an effect on the pricing. (but we will deleted when building the model)
### EDA
After the data preprocessing step, we should visualize the data to better understand the business,
how the data is distributed and have more insights about what is happening under the hoods. (please
find all analysis in the notebook)
## Part 2 - Feature selection

In the analysis made in the notebook we can see that the “length” , “built_year “,
”first_registration_date”, “horsepower” , “kw” and “width” are strongly correlated with the price. So
“Mileage” and “KW” are not the only relevant features to work with.
We can see as well that after training the model with only 2 features, we have had a very bad accuracy
equal to 0.463 what is already predicted by the cross validation score
scores = cross_val_score(RF, features, label, cv=5)
The Cross validation score estimates the accuracy of the Random forest classifier on the used cars dataset
by splitting the data, fitting a model and computing the score 5 consecutive times (with different splits
each time [4 for training and 1 for testing]). In our case, it was obvious that the chosen classifier want be
accurate on our dataset (cross valid score)
The diagonal cells show the number of correct classifications by the random forest classifier, while the off
diagonal cells represent the misclassified predictions. In the first class, we have 214 points . Out of
them the random forest classifier was able to get 132 elements correctly(TP). That's the recall.
132/214 = 0.61. Let’s look at the 2nd cell, there are elements scattered in all the five rows. Only 82
points from 183 points are correctly classified. All rest are incorrect. So that reduces the precision.
## Modeling
Normally for model selection we should use cross validation to test the data on multiple models,
for example :
classifiers = [
 svm.SVR(),
 linear_model.SGDRegressor(),
 linear_model.BayesianRidge(),
 linear_model.LassoLars(),
 linear_model.ARDRegression(),
 linear_model.PassiveAggressiveRegressor(),
 linear_model.TheilSenRegressor(),
 linear_model.LinearRegression()]
And then calculate the cross validation score for each model ( use shuffle for automatic split)
In our example, i arbitrary tried with the most 2 famous regression models: Linear regression and
gradient boosting regressor, calculated the cross validation score, the root mean square error , the
mean score and the variance score for both model (Please see banchmarking table in the notebook)
After selecting the best model, which was gradient boosting regressor, I used gridsearchcv for the
parameter fine tuning, with param grids:
{'n_estimators':[100],
 'learning_rate': [0.1],# 0.05, 0.02, 0.01],
 'max_depth':[6],#4,6],
 'min_samples_leaf':[3],#,5,9,17],
 'max_features':[1.0],#,0.3]#,0.1] }
and then we will have the best estimator that was found by gridsearchCV.
Question 4: Comparing models
Question 4a:
Models evaluations:
There is a lot of metrics to evaluate the models but we can also see the plot where we show the real against the predicted prices. We can easily see that both curves share the same pace. 
