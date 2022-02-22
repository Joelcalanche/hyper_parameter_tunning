"""
Parameters Overview

What is a parameter?

* Components of the model learned during the modeling process

* You do not set these manually(you can't in fact !)

* The algorithm will discover these for you

A simple logistic regression model:

log_reg_clf = LogisticRegression()

log_reg_clf.fit(X_train, y_train)

print(log_reg_clf.coef_)
Tidy up the coefficients:
# Get the original variable names

original_variables = list (X_train.columns)

# Zip together the names and coefficients

zipped_together = list(zip(original_variables, log_reg_clf.coef_[0]))

coefs = [list(x) for in zipped_together]

# Put into a DataFrame with column labels

coefs = pd.DataFrame(coefs, columns=["Variable", "Coefficient"])


Now sort and print the top three coefficients

coefs.sort_values(by=["Coeffcient"], axis=0, inplace=True, ascending=False)
print(coefs.head(3))

To find parameters we need:

1. To know a bit about the algorithm
2. Consult the Scikit Learn documentation
Parameters will be found under the 'Attributes'section, not the 'parameter's section

Parameters in Random Forest

What about tree based algorithms?

Random forest has no coefficients, but node decisions(what feature and what value to split on)

# A simple random forest estimator

rf_clf = RandomForestClassifier(max_depth=2)

rf_clf.fit(X_train, y_train)

# Pull out one tree from the forest
chosen_tree = rf_clf.estimators_[7]

for simplicity we will show the final product(an image) of the decision tree. Feel free to explore the package used for this (graphviz & pydotplus) yourself

Extracting Node Decisions

We can pull out details of the left, second-from-top node:

# Get the column it split on

split_column = chosen_tree.tree_.feature[1]
split_column_name = X_train.columns[split_column]

# Get the level it split on

split_value = chosen_tree.tree_.threshold[1]

print("This node split on feature{}, ar a value of {}".format(split_column_name, split_value))
"""
"""
Extracting a Logistic Regression parameter
You are now going to practice extracting an important parameter of the logistic regression model. The logistic regression has a few other parameters you will not explore here but you can review them in the scikit-learn.org documentation for the LogisticRegression() module under 'Attributes'.

This parameter is important for understanding the direction and magnitude of the effect the variables have on the target.

In this exercise we will extract the coefficient parameter (found in the coef_ attribute), zip it up with the original column names, and see which variables had the largest positive effect on the target variable.

You will have available:

A logistic regression model object named log_reg_clf
The X_train DataFrame
sklearn and pandas have been imported for you.
"""

# Create a list of original variable names from the training DataFrame
original_variables = list (X_train.columns)

# Extract the coefficients of the logistic regression estimator
print(log_reg_clf.coef_)
model_coefficients = log_reg_clf.coef_[0]
print(model_coefficients)

# Create a dataframe of the variables and coefficients & print it out
coefficient_df = pd.DataFrame({"Variable" : original_variables , "Coefficient": model_coefficients})
print(coefficient_df)

# Print out the top 3 positive variables
top_three_df = coefficient_df.sort_values(by=["Coefficient"], axis=0, ascending=False)[0:3]
print(top_three_df)


"""
Extracting a Random Forest parameter
You will now translate the work previously undertaken on the logistic regression model to a random forest model. A parameter of this model is, for a given tree, how it decided to split at each level.

This analysis is not as useful as the coefficients of logistic regression as you will be unlikely to ever explore every split and every tree in a random forest model. However, it is a very useful exercise to peak under the hood at what the model is doing.

In this exercise we will extract a single tree from our random forest model, visualize it and programmatically extract one of the splits.

You have available:

A random forest model object, rf_clf
An image of the top of the chosen decision tree, tree_viz_image
The X_train DataFrame & the original_variables list

"""

# Extract the 7th (index 6) tree from the random forest
chosen_tree = rf_clf.estimators_[7]

# Visualize the graph using the provided image
imgplot = plt.imshow(tree_viz_image)
plt.show()

# Extract the parameters and level of the top (index 0) node
split_column = chosen_tree.tree_.feature[0]
split_column_name = X_train.columns[split_column]
split_value = chosen_tree.tree_.threshold[0]

# Print out the feature and level
print("This node split on feature {}, at a value of {}".format(split_column_name, split_value))












"""

Hyper parameter tunning


* Something you set before the modeling process(like knobs on an old radio)
	* You also 'tune' your hyperparameters!

	* The algorithm does not learn these

Create a simple random forest estimator and print it out


rf_clf = RandomForestClassifier()

print(rf_clf)

RandomForestClassifier(n_estimatos='warn', criterion='gini',
max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=2,
min_weight_fraction_leaf = 0.0, n_jobs=None,
oob_score=False, random_state=None, verbose=0, bootstrap=True,
class_weight=None, warm_start=False)


Hyperparameters in Logistic Regression

log_reg_clf = LogisticRegression()

print(log_reg_clf)

LogisticRegression(C=1.0, class_weight= None, dual=False, fit_intercept=True, n_jobs=None, penalty='l2', random_state=None, solver='warn', tol=0.0001, verbose=0. war_start=False)

Hyperparameter Importance

Some hyperparameters are more important than others.

Some will not help model performance:

for the random forest classfier:
	* n_jobs
	*random_state
	* verborse

	Not all hyperparameters make sense to 'train'

Some important hyperparameters:

* n_estimators (high value)
* max_features (try different values)
* max_depth & min_sample_leaf(importante for overfitting)

* (maybe) criterion
How to find hyperparameters that matter?

Some resources for learning this:

* Academic papers

* Blogs and tutorials from trusted sources(Like dataCamp)

* The Scikit learn module documentation

* Experience
"""


"""
Hyperparameters in Random Forests
As you saw, there are many different hyperparameters available in a Random Forest model using Scikit Learn. Here you can remind yourself how to differentiate between a hyperparameter and a parameter, and easily check whether something is a hyperparameter.

You can create a random forest estimator yourself from the imported Scikit Learn package. Then print this estimator out to see the hyperparameters and their values.

Which of the following is a hyperparameter for the Scikit Learn random forest model?
"""


"""
Exploring Random Forest Hyperparameters
Understanding what hyperparameters are available and the impact of different hyperparameters is a core skill for any data scientist. As models become more complex, there are many different settings you can set, but only some will have a large impact on your model.

You will now assess an existing random forest model (it has some bad choices for hyperparameters!) and then make better choices for a new random forest model and assess its performance.

You will have available:

X_train, X_test, y_train, y_test DataFrames
An existing pre-trained random forest estimator, rf_clf_old
The predictions of the existing random forest estimator on the test set, rf_old_predictions

"""

# Print out the old estimator, notice which hyperparameter is badly set
print(rf_clf_old)

# Get confusion matrix & accuracy for the old rf_model
print("Confusion Matrix: \n\n {} \n Accuracy Score: \n\n {}".format(
  	confusion_matrix(y_test, rf_old_predictions),
  	accuracy_score(y_test, rf_old_predictions))) 




# Print out the old estimator, notice which hyperparameter is badly set
print(rf_clf_old)

# Get confusion matrix & accuracy for the old rf_model
print("Confusion Matrix: \n\n {} \n Accuracy Score: \n\n {}".format(
  confusion_matrix(y_test, rf_old_predictions),
  accuracy_score(y_test, rf_old_predictions))) 

# Create a new random forest classifier with better hyperparamaters
rf_clf_new = RandomForestClassifier(n_estimators=500)

# Fit this to the data and obtain predictions
rf_new_predictions = rf_clf_new.fit(X_train, y_train).predict(X_test)
print("Confusion Matrix: \n\n", confusion_matrix(y_test, rf_new_predictions))
print("Accuracy Score: \n\n", accuracy_score(y_test, rf_new_predictions))

"""
Hyperparameters of KNN
To apply the concepts learned in the prior exercise, it is good practice to try out learnings on a new algorithm. The k-nearest-neighbors algorithm is not as popular as it used to be but can still be an excellent choice for data that has groups of data that behave similarly. Could this be the case for our credit card users?

In this case you will try out several different values for one of the core hyperparameters for the knn algorithm and compare performance.

You will have available:

X_train, X_test, y_train, y_test DataFrames

"""

"""
knn_20 = KNeighborsClassifier(n_neighbors=20)

# Fit each to the training data & produce predictions
knn_5_predictions = knn_5.fit(X_train, y_train).predict(X_test)
knn_10_predictions = knn_10.fit(X_train, y_train).predict(X_test)
knn_20_predictions = knn_20.fit(X_train, y_train).predict(X_test)

# Get an accuracy score for each of the models
knn_5_accuracy = accuracy_score(y_test, knn_5_predictions  )
knn_10_accuracy = accuracy_score(y_test,knn_10_predictions)
knn_20_accuracy = accuracy_score(y_test, knn_20_predictions)
print("The accuracy of 5, 10, 20 neighbours was {}, {}, {}".format(knn_5_accuracy, knn_10_accuracy, knn_20_accuracy))

"""




"""
Hyperparameter Values

Some hyperparameters are more important than others to begin tuning.

But which values to try for hyperparameters?

* Specific to each algorithm & hyperparameter

* Some best practice guidelines & tips do exist

Let's look at some top tips!

Conflicting hyperparameter Choices

Be aware of conflicting hyperparameter choices

* LogisticRegression() conflicting parameter options of solver & penalty that conflict.

* Some are not explicit but will just 'ignore'(from ElasticNet with the normalize hyperparameter):

Make sure to consult the Scikit Learn documentation


Silly Hyperparameter Values

Be aware of setting 'silly' values for different algorithmns:

* Random forest with low number of trees

	* Would you consider it a' forest' with only 2 trees?
* 1 Neighbor in KNN algorithm
	* Averaging the 'votes' of one person doesn't sound very robust!

* Increasing a hyperparameter by a very small amount

Spending time documenting sensible values for hyperparameters is as valueble activity.

Automating Hyperparameter Choice

In the previos exercise, we built models as:

knn_5 = KNeighborsClassifier(n_neighbors=5)

knn_10 = KNeighbosClassifier(n_neighbors=10)

knn_20 = KNeighborsClassifier(n_neighbors=20)

This is quite inefficient. Can we do better?

Automating Hyperparameter Tuning
try a for loop to iterate through options:

	for test_number in neighbors_list:
		model = KNeighborsClassifier(n_neighbors=test_number)
		predictions = model.fit(X_train, y_train).predict(X_test)

		accuracy = accuracy_score(y_test, predictions)

		accuracy_list.append(accuracy)

	We can store the results in a DataFrame to view:

	results_df = pd.DataFrame({'neighbors': neighbors_list, 'accuracy':accuracy_list})

	print(results_df)

Let's create a learning curve graph

We'll test many more values this time

neighbors_list = list(range(5, 500, 5))

accuracy_list = []

for test_number in neighbors_list:
	model = KNeighborsClassifier(n_neighbors=test_number)
	predictions = model.fit(X_train, y_train).predict(X_test)

	accuracy = accuracy_score(y_test, predictions)

	accuracy_list.append(accuracy)

results_df = pd.DataFrame({'neighbors':neighbors_list, 'accuracy': accuracy_list})

Learning Curves
We can plot the larger DataFrame:

plt.plot(results_df['neighbors'],
	results_df['accuracy']
)

# Add the labels and title

plt.gca().set(xlabel='n_neighbors', ylabel='Accuracy', title='Accuracy for different n_neighbors')
plt.show()

A handy trick for generating values
Python's range function does not work for decimal steps.

* Create a numbe of values(num) evenly spread within an interval(start, end) that you specify.
print(np.linspace(1,2,5))
"""


"""
Automating Hyperparameter Choice
Finding the best hyperparameter of interest without writing hundreds of lines of code for hundreds of models is an important efficiency gain that will greatly assist your future machine learning model building.

An important hyperparameter for the GBM algorithm is the learning rate. But which learning rate is best for this problem? By writing a loop to search through a number of possibilities, collating these and viewing them you can find the best one.

Possible learning rates to try include 0.001, 0.01, 0.05, 0.1, 0.2 and 0.5

You will have available X_train, X_test, y_train & y_test datasets, and GradientBoostingClassifier has been imported for you.
"""


learning_rates = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]
results_list = []

# Create the for loop to evaluate model predictions for each learning rate
for learning_rate in learning_rates:
    model = GradientBoostingClassifier(learning_rate=learning_rate)
    predictions = model.fit(X_train, y_train).predict(X_test)
    # Save the learning rate and accuracy score
    results_list.append([learning_rate, accuracy_score(y_test, predictions)])

# Gather everything into a DataFrame
results_df = pd.DataFrame(results_list, columns=['learning_rate', 'accuracy'])
print(results_df)



"""
Building Learning Curves
If we want to test many different values for a single hyperparameter it can be difficult to easily view that in the form of a DataFrame. Previously you learned about a nice trick to analyze this. A graph called a 'learning curve' can nicely demonstrate the effect of increasing or decreasing a particular hyperparameter on the final result.

Instead of testing only a few values for the learning rate, you will test many to easily see the effect of this hyperparameter across a large range of values. A useful function from NumPy is np.linspace(start, end, num) which allows you to create a number of values (num) evenly spread within an interval (start, end) that you specify.

You will have available X_train, X_test, y_train & y_test datasets.

"""
# Set the learning rates & accuracies list
learn_rates = np.linspace(0.01, 2, num=30)
accuracies = []

# Create the for loop
for learn_rate in learn_rates:
  	# Create the model, predictions & save the accuracies as before
    model = GradientBoostingClassifier(learning_rate=learn_rate)
    predictions = model.fit(X_train, y_train).predict(X_test)
    accuracies.append(accuracy_score(y_test, predictions))

# Plot results    
plt.plot(learn_rates, accuracies)
plt.gca().set(xlabel='learning_rate', ylabel='Accuracy', title='Accuracy for different learning_rates')
plt.show()





#-------------------------------------------------------------------------------


"""

Introducing Grid Search
Hyperparameter tunning in python

Automating 2 Hyperparameters

What about testing values of 2 hyperparameters?

Using a GBM algorithm:
* learn_rate [0.001, 0.01, 0.05]

* max_depth [4,6,8,10]

We could use a (nested) for loop!

Automating 2 hyperparameters

firtly a model creation function:

def gbm_grid_search(learn_rate, max_depth):
	
	model = GradientBoostingClassifier(
		learning_rate = learn_rate,
		max_depth=max_depth)
	predictions = model.fit(X_train, y_train).predict(X_test)
	return([learn_rate, max_depth, accuracy_score(y_test, predictions)])

results_list = []

for learn_rate in learn_rate_list:
	for max_depth in max_depth_list:
		results_list.append(gbm_grid_search(learn_rate, max_depth))

	We can put these results into a DataFrame as well and print out:

	results_df = pd.DataFrame(results_list, columns=['learning_rate', 'max_depth', 'accuracy'])
	print(results_df)

How many models?

There were many more models built by adding more hyperparameters and values.

* The relationship is not linear, it is exponential
* One more value of hyperparameter is not just one model

* 5 for Hyperparameter 1 and 10 for hyperparameters 2 is 50 models

What about cross-validation?

* 10-fold cross-validation would make 50x10 =  500 models!

From 2 to N hyperparameters

What about adding more hypeparameters?

We could nest our loop!

# Adjust the list of values to test

learn_rate_list = [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5]

max_depth_list = [4, 6, 8, 10, 12, 15, 20, 25, 30]

subsample_list = [0.4, 0.6, 0.7, 0.8, 0.9]

max_features_list =['auto', 'sqrt']

Adjust or function:


def gbm_grid_search(learn_rate, max_depth, subsample, max_features):
	model = GradientBoostingClassifier(
		learning_rate = learn_rate,
		max_depth=max_depth,
		subsample=subsample,
		max_features=max_features
	)
	predictions = model.fit(X_train, y_train).predict(X_test)
	return([learn_rate, max_depth, accuracy_score(y_test, predictions)])



for learn_rate in learn_rate_list:
	for max_depth in max_depth_list:
		for subsample in subsample_list:
			for max_features in max_features_list:
				result_list.append(gbm_grid_search(learn_rate, max_depth, subsample, max_features))

results_df = pd.DataFrame(results_list, columns=['learning_rate', 'max_depth', 'subsample', 'max_features', 'accuracy'])

print(results_df)


From 2 to N hyperparameters

How many models now?

* 7*9*5*2 = 630(6300 if cross-validated!)
We can't keep nesting forever!

Plus, what if we wanted:

* Details on training times & scores
* Details on cross-validation scores

Let's create a grid:

* Down the left all values of max_depth

* Across the top all values of learning_rate
Working through each cell on the grid:

Some advantages of this approach:

Advantages:

* You don't have to write thousands of lines of code

* Finds the best model within the grid(*special note here!)

* Easy to explain

Some disadvantages of this approach:

* Computationally expensive! Remember how quickly we made 6000 + models?
* It is 'uninformed'. Results of one model don't help creating the next model.

We will cover 'informed' methods later!
"""




"""
Exercise
Build Grid Search functions
In data science it is a great idea to try building algorithms, models and processes 'from scratch' so you can really understand what is happening at a deeper level. Of course there are great packages and libraries for this work (and we will get to that very soon!) but building from scratch will give you a great edge in your data science work.

In this exercise, you will create a function to take in 2 hyperparameters, build models and return results. You will use this function in a future exercise.

You will have available the X_train, X_test, y_train and y_test datasets available.

"""
# Create the function
def gbm_grid_search(learning_rate, max_depth):

	# Create the model
    model = GradientBoostingClassifier(learning_rate=learning_rate, max_depth=max_depth)
    
    # Use the model to make predictions
    predictions = model.fit(X_train,y_train).predict(X_test)
    
    # Return the hyperparameters and score
    return([learning_rate, max_depth, accuracy_score(y_test, predictions)])



"""
Iteratively tune multiple hyperparameters
In this exercise, you will build on the function you previously created to take in 2 hyperparameters, build a model and return the results. You will now use that to loop through some values and then extend this function and loop with another hyperparameter.

The function gbm_grid_search(learn_rate, max_depth) is available in this exercise.

If you need to remind yourself of the function you can run the function print_func() that has been created for you
"""

# Create the relevant lists
results_list = []
learn_rate_list = [0.01, 0.1, 0.5]
max_depth_list = [2,4,6]

# Create the for loop
for learn_rate in learn_rate_list:
    for max_depth in max_depth_list:
        results_list.append(gbm_grid_search(learn_rate,max_depth))

# Print the results
print(results_list)   

"""
Extend the gbm_grid_search function to include the hyperparameter subsample. Name this new function gbm_grid_search_extended
"""
results_list = []
learn_rate_list = [0.01, 0.1, 0.5]
max_depth_list = [2,4,6]

# Extend the function input
def gbm_grid_search_extended(learn_rate, max_depth, subsample):

	# Extend the model creation section
    model = GradientBoostingClassifier(learning_rate=learn_rate, max_depth=max_depth, subsample=subsample)
    
    predictions = model.fit(X_train, y_train).predict(X_test)
    
    # Extend the return part
    return([learn_rate, max_depth, subsample, accuracy_score(y_test, predictions)])



 """
Grid Search with Scikit Learn

GridSearchCV Object

Introducing a GridSearchCV object

Introducing a GridSeatchCV object:

sklearn.model_selection.GridSearchCV(
	estimator,
	param_grid, scoring=None, fit_params=None,
	n_jobs=None, iid='warn', refit=True, cv='warn',
	verbose=0, pre_dispatch='2*n_jobs',
	error_score='raise-deprecating',
	return_train_score='warn'
)
 Steps in a Grid Search

 Steps in a Grid Search:

 1. An algorithm to tune the hyperparameters.(Sometimes called an 'estimator')

 2. Defining which hyperparameters we will tune

 3. Defining a range of values for each hyperparameter

 4. Setting a cross-validation scheme; and 

 5. Define a score function so we can decide which square on our grid was 'the best'.

 6. Include extra usefull information or functions
 
GridSearchCV Object Inputs

The important imputs are:
* estimator
* param_grid
* cv
* scoring
* refit
* n_jobs
* return_train_score

The estimator input:
* Essentially our algorithm
* You have already worked with KNN, Random Forest, GBM, Logistic Regression

Remember:

* Only one estimator per GridSearchCV object
 
The param_grid input:

* Setting which hyperparameters and values to test

Rather than a list:

max_depth_list = [2, 4, 6, 8]

min_samples_leaf_list = [1, 2, 4, 6]


This would be:

param_grid = {'max_depth': [2, 4, 6, 8],
			  'min_samples_leaf': [1, 2, 3, 4, 6]}

The param_grid input:

Remember: The keys in your param_grid dictionary must be valid hyperparameters
The cv input:

* choice of how to undertake cross-validation

* Using an integer undertakes k-fold cross validation where 5 or 10 is usually standard
 The scoring input:

 * Which score to use to choose the best grid square(model)

 * Use your own or Scikit Learn's metrics module

 You can check all the built in scoring functions this way:

 from sklearn import metrics

 sorted(metrics.SCORERS.keys())

 The refit input:
 * Fits the best hyperparameters to the training data
 * Allows the GridSearchCV object to be used as an estimator(for prediction)

 * A very handy option!

GridSearchCV 'n_jobs'

The n_jobs input:
* Assist with parallel execution

* Allows multiple models to be created at the same time, rather than one after ther other

Some handy code:

import os
print(os.cpu_count())
 
Careful using all your cores for modelling if you want to do other work
 
The return_train_score input:

* Logs statistics about the training runs that were undertaken

* Useful for analyzing bias-variance trade-off but adds computational expense

* Does not assit in picking the best model, only for analysis purposes
 
# Create a grid

param_grid = {'max_depth': [2, 4, 6, 8], 'min_samples_leaf': [1, 2, 4, 6] }

#Get a base classifier with some set parameters.

rf_class = RandomForestClassifier(criterion='entropy', max_features='auto')

Putting the pieces together:

grid_rf_class = GridSearchCV(
	estimator = rf_class,
	param_grid = parameter_grid,
	scoring = 'accuracy',
	n_jobs=4,
	cv = 10,
	refit = True,
	return_train_score=True
)


# Fit the object to our data

grid_rf_class.fit(X_train, y_train)

# Make predictions

grid_rf_class.predict(X_test)
 """
"""
GridSearchCV inputs
Let's test your knowledge of GridSeachCV inputs by answering the question below.

Three GridSearchCV objects are available in the console, named model_1, model_2, model_3. Note that there is no data available to fit these models. Instead, you must answer by looking at their construct.

Which of these GridSearchCV objects would not work when we try to fit it?
"""
"""
Course Outline
Daily XP
616




Exercise
Exercise
GridSearchCV with Scikit Learn
The GridSearchCV module from Scikit Learn provides many useful features to assist with efficiently undertaking a grid search. You will now put your learning into practice by creating a GridSearchCV object with certain parameters.

The desired options are:

A Random Forest Estimator, with the split criterion as 'entropy'
5-fold cross validation
The hyperparameters max_depth (2, 4, 8, 15) and max_features ('auto' vs 'sqrt')
Use roc_auc to score the models
Use 4 cores for processing in parallel
Ensure you refit the best model and return training scores
You will have available X_train, X_test, y_train & y_test datasets.



"""


# Create a Random Forest Classifier with specified criterion
rf_class = RandomForestClassifier(criterion='entropy')

# Create the parameter grid
param_grid = {'max_depth': [2, 4, 8, 15], 'max_features': ['auto', 'sqrt']} 

# Create a GridSearchCV object
grid_rf_class = GridSearchCV(
    estimator=rf_class,
    param_grid=param_grid,
    scoring='roc_auc',
    n_jobs=4,
    cv=5,
    refit=True, return_train_score=True)
print(grid_rf_class)


"""
Understanding a  grid search output

Anlyzing the output

Three different groups for the GridSearchCV properties:

* A resuls log
	cv_results_
* The best results_
	best_index_, best_params_ & best_score_

* 'Extra information'
	scorer_, n_splits_ & refit_time_

Properties are accessed using the dot notation.

For example:

grid_search_object.property

Where property is the actual property you want to retrieve

The cv_results_ property:

Read this into a DataFrame to print and analyze:

cv_results_df = pd.DataFrame(grid_rf_class.cv_results_)

print(cv_results_df.shape)

* The 12 rows for the 12 squares in our grid or 12 models we ran

The time columns refer to the time it took to fit(and score) the model.

Remember how we did a 5-fold cross-validation? This ran 5 times and stored the average and  standard deviation of the times it took in seconds


The .cv_results_ 'param_' columns

The param_ columns store the parameters it tested on that row, one column per parameter


the params column contains dictionary of all the parameters:

pd.set_option("display.max_colwidth", -1)
print(cv_results_df.loc[:, "params"])

The test_score columns contain the scoes on our test set for each of our cross-folds as well as some summary statistics:


The rank column, ordering the mean_test_score from best to worst

We can select the best grid square easily from cv_results_ using the rank_test_score column

best_row = cv_results_df[cv_results_df["rank_test_score"] == 1]

print(best_row)

The test_score columns are then repeated for the training_scores.

Some important notes to keep in mind:

* return_train_score must be True to include training scores columns.

* There is no ranking column for the training scores, as we only care about test set performance

The best grid square

Information on the best grid square is neatly summarized in the following three properties:

* best_params_, the dictionary of parameters that gave the best score.

* best_score_, the actual best score.

* best_index_, the row in our cv_results_.rank_test_score that was the best.

The 'best_estimator_' property

The best_estimator_ property is an estimator build using the best parameters from the grid search.

For us this is as Random Forest estimator:

type(grid_rf_class.best_estimator_)

sklearn.ensemble.forest.RandomForestClassifier

We could also directly use this object as an estimator if we want

print(grid_rf_class.best_estimator_)


Extra information

Some extra information is available in the following properties:

* scorer_

What scorer function was used on the held out data. (we set it to AUC)

* n_splits_

How many cross-validation splits. (We set to 5)

The number of seconds used for refitting the best model on the whole dataset
"""
"""
Exploring the grid search results
You will now explore the cv_results_ property of the GridSearchCV object defined in the video. This is a dictionary that we can read into a pandas DataFrame and contains a lot of useful information about the grid search we just undertook.

A reminder of the different column types in this property:

time_ columns
param_ columns (one for each hyperparameter) and the singular params column (with all hyperparameter settings)
a train_score column for each cv fold including the mean_train_score and std_train_score columns
a test_score column for each cv fold including the mean_test_score and std_test_score columns
a rank_test_score column with a number from 1 to n (number of iterations) ranking the rows based on their mean_test_score
"""

# Read the cv_results property into a dataframe & print it out
cv_results_df = pd.DataFrame(grid_rf_class.cv_results_)
print(cv_results_df)

# Extract and print the column with a dictionary of hyperparameters used
column = cv_results_df.loc[:, ["params"]]
print(column)

# Extract and print the row that had the best mean test score
best_row = cv_results_df[cv_results_df["rank_test_score"] == 1]
print(best_row)




"""
Analyzing the best results
At the end of the day, we primarily care about the best performing 'square' in a grid search. Luckily Scikit Learn's gridSearchCv objects have a number of parameters that provide key information on just the best square (or row in cv_results_).

Three properties you will explore are:

best_score_ – The score (here ROC_AUC) from the best-performing square.
best_index_ – The index of the row in cv_results_ containing information on the best-performing square.
best_params_ – A dictionary of the parameters that gave the best score, for example 'max_depth': 10
The grid search object grid_rf_class is available.

A dataframe (cv_results_df) has been created from the cv_results_ for you on line 6. This will help you index into the results.


"""

# Print out the ROC_AUC score from the best-performing square
best_score = grid_rf_class.best_score_
print(best_score)

# Create a variable from the row related to the best-performing square
cv_results_df = pd.DataFrame(grid_rf_class.cv_results_)
best_row = cv_results_df.loc[[grid_rf_class.best_index_]]
print(best_row.head())

# Get the n_estimators parameter from the best-performing square and print
best_n_estimators = grid_rf_class.best_params_["n_estimators"]
print(best_n_estimators)



"""
Using the best results
While it is interesting to analyze the results of our grid search, our final goal is practical in nature; we want to make predictions on our test set using our estimator object.

We can access this object through the best_estimator_ property of our grid search object.

Let's take a look inside the best_estimator_ property, make predictions, and generate evaluation scores. We will firstly use the default predict (giving class predictions), but then we will need to use predict_proba rather than predict to generate the roc-auc score as roc-auc needs probability scores for its calculation. We use a slice [:,1] to get probabilities of the positive class.

You have available the X_test and y_test datasets to use and the grid_rf_class object from previous exercises.

"""

# See what type of object the best_estimator_ property is
print(type(grid_rf_class.best_estimator_))

# Create an array of predictions directly using the best_estimator_ property
predictions = grid_rf_class.best_estimator_.predict(X_test)

# Take a look to confirm it worked, this should be an array of 1's and 0's
print(predictions[0:5])

# Now create a confusion matrix 
print("Confusion Matrix \n", confusion_matrix(y_test, predictions))

# Get the ROC-AUC score
# predictions_proba_2 = grid_rf_class.best_estimator_.predict_proba(X_test)
# print(predictions_proba_2)

predictions_proba = grid_rf_class.best_estimator_.predict_proba(X_test)[:,1]
print("ROC-AUC Score \n", roc_auc_score(y_test,predictions_proba))





# Random search

"""

What you already know

Very similar to grid search:

* Define an estimator, which hyperparameters to tune and the range of values for each
hyperparameter


*  We still set a cross-validation scheme and scoring function

BUT we instead randomly select grid squares


Why does this work?

Bengio & Bergstra(2012):

This paper shows empirically and theoretically that randomly chosen trials are more
efficient for hyper-parameter optimization than trial on a grid.

Two main reasons:

1. Not every hyperparameter is as important

2. A little trick of probability

if we randomly select hyperparameter combinations uniformly, let's consider the chance of MISSING every single trial, to show how unlikely that is


Trial 1 = 0.05 chance of success and (1- 0.05) of missing

Trial 2 = (1-0.05)*(1-0.05) of missing the range

trial 3 = (1-0.05)*(1-0.05)(1-0.05) of missing again


* In fact, with n trials we have (1-0.05)^n chance that every single trial misses that desired spot


So how many trial to have a high (95%) chance of getting in that region?

* We have (1-0.05)^n chance to miss everythin.

* So we must have(1-miss everything) chance to get in there or (1-(1-0.05)^n)

* Solving 1-(1-0.005)^n >= 95
n > = 59

What does that all mean?

* You are unlikely to keep completely missing the 'good area' for a long time when randomly picking new spots


* A grid search may spend lots of time in the 'bad area' as it covers exhaustively


Some important notes

Remember:

1. The maximum is still only as good as the grid you set!

2. Remember to fairly compare this to grid search, you need to have the same modeling


# Set some hyperparameter list

learn_rate_list = np.linspace(0.001, 2, 150)

min_samples_leaf_list = list(range(1, 51))


# Create list of combinations

from itertools import product

combination_list = [list(x) for x in product(learn_rate_list, min_samples_leaf_list)]


# Select 100 models from our larger set

random_combinations_index = np.random.choice(range(0, len(combinations_list)), 100,
replace=False)



combinations_random_chosen = [combinations_list[x] for in random_combinations_index]


Visualizing a Random Search

We can also visualize the random search coverage by plotting the hyperparameter choices on an X and Y axis
"""
"""
Randomly Sample Hyperparameters
To undertake a random search, we firstly need to undertake a random sampling of our hyperparameter space.

In this exercise, you will firstly create some lists of hyperparameters that can be zipped up to a list of lists. Then you will randomly sample hyperparameter combinations preparation for running a random search.

You will use just the hyperparameters learning_rate and min_samples_leaf of the GBM algorithm to keep the example illustrative and not overly complicated.
"""


# Create a list of values for the learning_rate hyperparameter
learn_rate_list = list(np.linspace(0.01,1.5,200))

# Create a list of values for the min_samples_leaf hyperparameter
min_samples_list = list(range(10,41))

# Combination list
combinations_list = [list(x) for x in product(learn_rate_list, min_samples_list)]

# Sample hyperparameter combinations for a random search.
random_combinations_index = np.random.choice(range(0, len(combinations_list)), 250, replace=False)
combinations_random_chosen = [combinations_list[x] for x in random_combinations_index]

# Print the result
print(combinations_random_chosen)


"""
Randomly Search with Random Forest
To solidify your knowledge of random sampling, let's try a similar exercise but using different hyperparameters and a different algorithm.

As before, create some lists of hyperparameters that can be zipped up to a list of lists. You will use the hyperparameters criterion, max_depth and max_features of the random forest algorithm. Then you will randomly sample hyperparameter combinations in preparation for running a random search.

You will use a slightly different package for sampling in this task, random.sample().

"""
# Create lists for criterion and max_features
criterion_list = ['gini', 'entropy']
max_feature_list = ["auto", "sqrt", "log2", None]

# Create a list of values for the max_depth hyperparameter
max_depth_list = list(range(3, 56))

# Combination list
combinations_list = [list(x) for x in product(criterion_list, max_feature_list, max_depth_list)]

# Sample hyperparameter combinations for a random search
combinations_random_chosen = random.sample(combinations_list, 150)

# Print the result
print(combinations_random_chosen)
"""
Visualizing a Random Search
Visualizing the search space of random search allows you to easily see the coverage of this technique and therefore allows you to see the effect of your sampling on the search space.

In this exercise you will use several different samples of hyperparameter combinations and produce visualizations of the search space.

The function sample_and_visualize_hyperparameters() takes a single argument (number of combinations to sample) and then randomly samples hyperparameter combinations, just like you did in the last exercise! The function will then visualize the combinations.

If you want to see the function definition, you can use Python's handy inspect library, like so:

print(inspect.getsource(sample_and_visualize_hyperparameters))

# Plot 
  plt.clf() 
  plt.scatter(rand_y, rand_x, c=['blue']*len(combinations_random_chosen))
  plt.gca().set(xlabel='learn_rate', ylabel='min_samples_leaf', title='Random Search Hyperparameters')
  plt.gca().set_xlim(x_lims)
  plt.gca().set_ylim(y_lims)
  plt.show()

"""


"""
Random search in Scikit learn

1. Decide an algorthm/estimator

2. Defining which hyperparameters we will tune

3. Defining a range of values for each hyperparameter

4. Setting a cross-validation scheme; and

5. Define a score function

6. Include extra useful information or functions

7. Step 7 = Decide how many samples to take (then sample)


Two key differences:

* n_iter which is the number of samples for the random search to take from your grid. In the previous example you did 300
* param_distributions is slightly different from param_grid, allowing optional ability to set a distribution for sampling
* The default is all combinations have equal chance to be chosen

Now we can build a random search object just like the grid search, but with our small change:

# Set up the sample space

learn_rate_list = np.linspace(0.001, 2, 150)

min_samples_leaf_list = list(range(1, 51))


# Create the grid

parameter_grid = {
	
	'learning_rate': learn_rate_list,
	'min_samples_leaf': min_samples_leaf_list
}

# Define how many samples

number_models =10

Now we can build the object

# Create a random search object

random_GBM_class = RandomizedSearchCV(
		estimator = GradientBoostingClassifier();
		param_distributions = parameter_grid,
		n_iter = number_models,
		scoring = 'accuracy',
		n_jobs=4,
		cv =10,
		refit=True,
		return_train_score= True)

		# Fit the object to our data

		random_GBM_class.fit(X_train, y_train)

Analyze the output

The output is exactly the same!

How do we see what hyperparameter values were chosen?

The cv_results_ dictionary (in the relevant param_ columns)!


Extract the lists:

rand_x = list(random_GBM_class.cv_results_['param_learning_rate'])

rand_y = list (random_GBM_class.cv_results_['param_min_samples_leaf'])

# Make sure we set the limiits of Y and X appriately

x_lims = [np.min(learn_rate_list), np.max(learn_rate_list)]

y_lims = [np.min(min_samples_leaf_list), np.max(min_samples_leaf_list)]

# PLot grid results


plt.scatter(rand_y, rand_x, c['blue']*10)

plt.gca().set(xlabel='learn_rate', ylabel='min_samples_leaf',
		title = 'Random Search Hyperparameters'
)

plt.show()

"""


"""
The RandomizedSearchCV Object
Just like the GridSearchCV library from Scikit Learn, RandomizedSearchCV provides many useful features to assist with efficiently undertaking a random search. You're going to create a RandomizedSearchCV object, making the small adjustment needed from the GridSearchCV object.

The desired options are:

A default Gradient Boosting Classifier Estimator
5-fold cross validation
Use accuracy to score the models
Use 4 cores for processing in parallel
Ensure you refit the best model and return training scores
Randomly sample 10 models
The hyperparameter grid should be for learning_rate (150 values between 0.1 and 2) and min_samples_leaf (all values between and including 20 and 64).

You will have available X_train & y_train datasets.
"""


# Create the parameter grid
param_grid = {'learning_rate': np.linspace(0.1,2,150), 'min_samples_leaf': list(range(20,65))} 

# Create a random search object
random_GBM_class = RandomizedSearchCV(
    estimator = GradientBoostingClassifier(),
    param_distributions = param_grid,
    n_iter = 10,
    scoring='accuracy', n_jobs= 4, cv = 5, refit=True, return_train_score = True)

# Fit to the training data
random_GBM_class.fit(X_train, y_train)

# Print the values used for both hyperparameters
print(random_GBM_class.cv_results_['param_learning_rate'])
print(random_GBM_class.cv_results_['param_min_samples_leaf'])




"""
RandomSearchCV in Scikit Learn
Let's practice building a RandomizedSearchCV object using Scikit Learn.

The hyperparameter grid should be for max_depth (all values between and including 5 and 25) and max_features ('auto' and 'sqrt').

The desired options for the RandomizedSearchCV object are:

A RandomForestClassifier Estimator with n_estimators of 80.
3-fold cross validation (cv)
Use roc_auc to score the models
Use 4 cores for processing in parallel (n_jobs)
Ensure you refit the best model and return training scores
Only sample 5 models for efficiency (n_iter)
X_train & y_train datasets are loaded for you.

Remember, to extract the chosen hyperparameters these are found in cv_results_ with a column per hyperparameter. For example, the column for the hyperparameter criterion would be param_criterion.

"""


# Create the parameter grid
param_grid = {'max_depth': list(range(5 ,26)), 'max_features': ['auto' , 'sqrt']} 

# Create a random search object
random_rf_class = RandomizedSearchCV(
    estimator = RandomForestClassifier(n_estimators=80),
    param_distributions = param_grid, n_iter = 5,
    scoring="roc_auc", n_jobs=4, cv = 3, refit=True, return_train_score = True )

# Fit to the training data
random_rf_class.fit(X_train, y_train)

# print(random_rf_class.cv_results_)

# Print the values used for both hyperparameters
print(random_rf_class.cv_results_['param_max_depth'])
print(random_rf_class.cv_results_['param_max_features'])



"""

Comparing Grid and Random Search

What's the same?

Similarities between Random and Grid Search?

* Both are automated ways to tunning different hyperparameters
* For both you set the grid to sample from(which hyperparameters and values for each)

Remember to think carafully about your grid!

* For both you set a cross-validation scheme and scoring function

Wha's different?

Grid Search:

* Exhaustively tries all combination within

the sample space

* No Sampling Methodology

* More computationally expensive

* Guaranteed to find the best score in the sample space






Random Search
* Randomly selects a subset of combinations within the sample space(that you must specify)

* Can select a samplin methodology(other than unifomr which i default)

* Less computationally expensive

* Not a guaranteed to find the best score in the sample space(but likely to find a good one faster)


Which should i use?

So which one should i use? what are my considetations

* How much data do you have?

* How many hyperparameters and values do you want to tune?

* How much resources do you have?(Time,computing powers)


* More data means random search may be better option

* More of theses means randomo search may be a better option

* Les resources means random search may be a better option

"""

"""
Grid and Random Search Side by Side
Visualizing the search space of random and grid search together allows you to easily see the coverage that each technique has and therefore brings to life their specific advantages and disadvantages.

In this exercise, you will sample hyperparameter combinations in a grid search way as well as a random search way, then plot these to see the difference.

You will have available:

combinations_list which is a list of combinations of learn_rate and min_samples_leaf for this algorithm
The function visualize_search() which will make your hyperparameter combinations into X and Y coordinates and plot both grid and random search combinations on the same graph. It takes as input two lists of hyperparameter combinations.
If you wish to view the visualize_search() function definition, you can run this code:

"""

 # Plot all together
  plt.scatter(grid_y + rand_y, grid_x + rand_x, c=['red']*300 + ['blue']*300)
  plt.gca().set(xlabel='learn_rate', ylabel='min_samples_leaf', title='Grid and Random Search Hyperparameters')
  plt.gca().set_xlim(x_lims)
  plt.gca().set_ylim(y_lims)
  plt.show()

  # Sample grid coordinates
grid_combinations_chosen = combinations_list[0:300]

# Print result
print(grid_combinations_chosen)

"""
Let's randomly sample too. Create a list of every index in combinations_list to sample from using range()
Use np.random.choice() to sample 300 combinations. The first two arguments are a list to sample from and the number of samples.
"""

# Sample grid coordinates
grid_combinations_chosen = combinations_list[0:300]

# Create a list of sample indexes
sample_indexes = list(range(0,len(combinations_list)))

# Randomly sample 300 indexes
random_indexes = np.random.choice(sample_indexes, 300, replace=False)


# Sample grid coordinates
grid_combinations_chosen = combinations_list[0:300]

# Create a list of sample indexes
sample_indexes = list(range(0,len(combinations_list)))

# Randomly sample 300 indexes
random_indexes = np.random.choice(sample_indexes, 300, replace=False)

# Use indexes to create random sample
random_combinations_chosen = [combinations_list[index] for index in random_indexes]


# Sample grid coordinates
grid_combinations_chosen = combinations_list[0:300]

# Create a list of sample indexes
sample_indexes = list(range(0,len(combinations_list)))

# Randomly sample 300 indexes
random_indexes = np.random.choice(sample_indexes, 300, replace=False)

# Use indexes to create random sample
random_combinations_chosen = [combinations_list[index] for index in random_indexes]

# Call the function to produce the visualization
visualize_search(grid_combinations_chosen, random_combinations_chosen)





"""

Informed Search: Coarse to Fine

So far everything we have done has been uninformed search:

Uninformed search: Where each iteration of hyperparameter tunning does not learn from the previous iterations



This is what allows us to parallelize our work. Though this doesn't sound very efficient?


A basic informed search methodology:

Start out with a rough, random approach and iteratively refine your search.

The process is:

1. Random search

2. Find promising areas

3. Grid search in the smaller area

4. Continue until optimal score obtained

You could substitute(3) with further random searches before the grid search


Why Coarse to Fine?

Coarse to fine tuning has some advantages:


* Utilize the advantages of grid and random search.

		* Wide search to begin with

		 Deeper search once you know where a good spot is likely to be

* Better spending of time and computational efforts mean you can iterate quicker 

No need to waste time on search spaces that are not giving good results!

Note: This isn't informed on one model but batches

Let's take an example with the following hyperparameter ranges:

* max_depth_list between 1 and 65

* min_sample_list between 3 and 17

* learn_rate_list 150 values between 0.01 and 150

How many possible models do we have?


combinations_list = [list(x) for x in product(max_depth_list, min_sample_list, learn_rate_list)]

print(len(combinations_list))



Let's do a random search on just 500 combinations.

Here we plot our accuracy scores:

Which models were the good ones?

Let's visualize the max_depth values vs accuracy score:

hacemos scatter plots para ver el impacto de cada hyperparameter

What we know from iteration one:
* max_depth between 8 and 30
* learn_rate less than 1.3
* min_samples_leaf perhaps less than 8

Where to next? Another random or grid search with wath we know!

Note: This was only bivariate analysis. You can explore looking at multiple hyperparameter(3, 4 or more !) on a single graph, but that's beyond the scope of this course
"""

"""
Visualizing Coarse to Fine
You're going to undertake the first part of a Coarse to Fine search. This involves analyzing the results of an initial random search that took place over a large search space, then deciding what would be the next logical step to make your hyperparameter search finer.

You have available:

combinations_list - a list of the possible hyperparameter combinations the random search was undertaken on.
results_df - a DataFrame that has each hyperparameter combination and the resulting accuracy of all 500 trials. Each hyperparameter is a column, with the header the hyperparameter name.
visualize_hyperparameter() - a function that takes in a column of the DataFrame (as a string) and produces a scatter plot of this column's values compared to the accuracy scores. An example call of the function would be visualize_hyperparameter('accuracy')
If you wish to view the visualize_hyperparameter() function definition, you can run this code:

"""
def visualize_hyperparameter(name):
  plt.clf()
  plt.scatter(results_df[name],results_df['accuracy'], c=['blue']*500)
  plt.gca().set(xlabel='{}'.format(name), ylabel='accuracy', title='Accuracy for different {}s'.format(name))
  plt.gca().set_ylim([0,100])
  plt.show()

# Confirm the size of the combinations_list
print(len(combinations_list))
# print(results_df.head())

# Sort the results_df by accuracy and print the top 10 rows
print(results_df.sort_values(by="accuracy", ascending=False).head(10))

# Confirm which hyperparameters were used in this search
print(results_df.columns)

# Call visualize_hyperparameter() with each hyperparameter in turn
visualize_hyperparameter('max_depth')
visualize_hyperparameter('min_samples_leaf')
visualize_hyperparameter('learn_rate')


"""
Coarse to Fine Iterations
You will now visualize the first random search undertaken, construct a tighter grid and check the results. You will have available:

results_df - a DataFrame that has the hyperparameter combination and the resulting accuracy of all 500 trials. Only the hyperparameters that had the strongest visualizations from the previous exercise are included (max_depth and learn_rate)
visualize_first() - This function takes no arguments but will visualize each of your hyperparameters against accuracy for your first random search.
If you wish to view the visualize_first() (or the visualize_second()) function definition, you can run this code:
"""

def visualize_first():
  for name in results_df.columns[0:2]:
    plt.clf()
    plt.scatter(results_df[name],results_df['accuracy'], c=['blue']*500)
    plt.gca().set(xlabel='{}'.format(name), ylabel='accuracy', title='Accuracy for different {}s'.format(name))
    plt.gca().set_ylim([0,100])
    x_line = 20
    if name == "learn_rate":
      	x_line = 1
    plt.axvline(x=x_line, color="red", linewidth=4)
    plt.show() 


# Use the provided function to visualize the first results
# visualize_first()

# Create some combinations lists & combine
max_depth_list = list(range(1, 21))
learn_rate_list = np.linspace(0.001, 1, 50)


"""
We ran the 1,000 model grid search in the background based on those new combinations. Now use the visualize_second() function to visualize the second iteration (grid search) and see if there is any improved results. This function takes no arguments, just run it in-place to generate the plots!
"""

def visualize_second():
  for name in results_df2.columns[0:2]:
    plt.clf()
    plt.scatter(results_df2[name],results_df2['accuracy'], c=['blue']*1000)
    plt.gca().set(xlabel='{}'.format(name), ylabel='accuracy', title='Accuracy for different {}s'.format(name))
    plt.gca().set_ylim([0,100])
    plt.show()


# Use the provided function to visualize the first results
# visualize_first()

# Create some combinations lists & combine:
max_depth_list = list(range(1,21))
learn_rate_list = np.linspace(0.001,1,50)

# Call the function to visualize the second results
visualize_second()



"""

Informed Methods:
Bayesian Statistics

Bayes Rule:

A statistical method of using new evidence to iteratively update our beliefs about some outcome


* Intuitively fits with the idea of informed search. Getting better as we get more evidence


Bayes Rule has the form:

P(A|b) = (p(b|A)*P(A))/ P(B)

* LHS(left hand side) = The probability of A, given B has ocurred. B is some new evidence.
* This is known as the 'posterior'

* RHS is how we calculate this.

* P(A)  is the 'prior'. The initial hypotesis about the event. It is different to P(A|B), the P(A|B) is the probability given new evidence

* P(B) is the 'marginal likelihood' and it is the probability of observing this new evidence

* P(B|A) is the 'likelihood' which is the probability of oberserving the evidence, given the event we care about

This all may be quite confusing, but let's use a commom example of a medical diagnosis to demonstrate


A medical example:

* 5% of peopel in the general population have a certain disease
	* P(D)

* 10 % of people are predisposed

	* P(Pre)

* 20% of people with disease are predisposed

		* P(Pre|D)


What is the probability that  any person has the disease?
(sin informacion)
P(D) = 0.05

This is simply our prior as we have no evidence

What is the probability that a predisposed person has the disease?

P(D|Pre) = (P(Pre|D)*P(D))/P(pre)


Bayes in Hyperparameter Tuning

* Pick a hyperparameter  combination

* Build a model

* Get new evidence(the score of the model)

* Update our beliefs and chose better hyperparameter next round

Bayesian hyperparameter tunning is very new but quite popular for larger and more complex hyperparameter tuning task as they work well to find optimal hyperparameter combinations in these situations



Bayesian Hyperparameter Tunning with Hyperopt

Introducing the Hyperopt package


To undertake bayesian hyperparameter tunning we need to:

1. Set the Domain: Our Grid(with a bit of a twist)

2. Set the Optimization algorithm(use default TPE)

3. Objective function to minimize: we will use 1-Accuracy

Many options to set the grid:


* Simple numbers

* Choose from a list

* Distribution of values

Hyperopt does not use point values on the grid but instead each point represents probabilities

for each hyperparameter value.

We will do a simple uniform distribution but there are many more if you check the documentation.


The Domain

Set up the grid:


space = {
	
	'max_depth': hp.quniform('max_depth', 2, 10, 2),
	'min_samples_leaf': hp.quniform('min_samples_leaf', 2, 8, 2),
	'learning_rate': hp.uniform('learning_rate', 0.01, 1, 55),
}

The objective function

The objective function runs the algorithm:

"""


def objective(params):
	params = {'max_depth': int(params['max_depth'],
		'min_samples_leaf': int(params['min_samples_leaf']),
		'learning_rate': params['learn_rate'])}

	gbm_clf = GradientBoostingClassifier(n_estimators=500, **params)

	best_score = cross_val_score(gbm_clf, X_train, y_train,
		scoring='accuracy', cv=10, n_jobs=4).mean()

	loss = 1 - best_score

	write_results(best_score, params, iteration)

	return loss

"""
best_result = fmin(
		fn=objective,
		space=space,
		max_evals=500,
		rstate=np.random.RandomState(42),
		algo=tpe.suggest

)
"""

"""
Bayes Rule in Python
In this exercise you will undertake a practical example of setting up Bayes formula, obtaining new evidence and updating your 'beliefs' in order to get a more accurate result. The example will relate to the likelihood that someone will close their account for your online software product.

These are the probabilities we know:

7% (0.07) of people are likely to close their account next month
15% (0.15) of people with accounts are unhappy with your product (you don't know who though!)
35% (0.35) of people who are likely to close their account are unhappy with your product
"""

# Assign probabilities to variables 
p_unhappy = 0.15
p_unhappy_close = 0.35

# Probabiliy someone will close
p_close = 0.07

# Probability unhappy person will close
p_close_unhappy = ( 0.35 * 0.07) / 0.15
print(p_close_unhappy)



"""
Bayesian Hyperparameter tuning with Hyperopt
In this example you will set up and run a Bayesian hyperparameter optimization process using the package Hyperopt (already imported as hp for you). You will set up the domain (which is similar to setting up the grid for a grid search), then set up the objective function. Finally, you will run the optimizer over 20 iterations.

You will need to set up the domain using values:

max_depth using quniform distribution (between 2 and 10, increasing by 2)
learning_rate using uniform distribution (0.001 to 0.9)
Note that for the purpose of this exercise, this process was reduced in data sample size and hyperopt & GBM iterations. If you are trying out this method by yourself on your own machine, try a larger search space, more trials, more cvs and a larger dataset size to really see this in action!

"""

# Set up space dictionary with specified hyperparameters
space = {'max_depth': hp.quniform('max_depth', 2, 10, 2),'learning_rate': hp.uniform('learning_rate', 0.001, 0.9)}

# Set up objective function
def objective(params):
    params = {'max_depth': int(params['max_depth']),'learning_rate': params['learning_rate']}
    gbm_clf = GradientBoostingClassifier(n_estimators=100, **params) 
    best_score = cross_val_score(gbm_clf, X_train, y_train, scoring='accuracy', cv=2, n_jobs=4).mean()
    loss = 1 - best_score
    return loss

# Run the algorithm
best = fmin(fn=objective,space=space, max_evals=20, rstate=np.random.RandomState(42), algo=tpe.suggest)
print(best)




"""
Genetic algorithm


In genetic evolution in the real world, we have the following process:

1. There are many creatures existing('offspring')

2. The strongest creatures survive and pair off.

3. There is some 'crossover' as they form offspring

4. There are random mutations to some of the offspring
		These mutations sometimes help give some offspring on advantage

	5. Go back to (1)!


	We can apply the same idea to hyperparameter tunning:

	1. We can create some models(that have hyperparameter settings)

	2. We can pick the best(by our scoring function)

			These are the ones that'survive'

		3. We can create new models that are similar to the best ones

		4. We add in some randomness so we don't reach a local optimum

		5. Repeat until we are happy		


This is an informed search that has a number  of advantages:

* It allows us to learn from previous iterations, just like bayesian hyperparameter tuning

* It has the additional advantage of some randomness

* (The package we'll use) takes care of many tedious aspects of machine learning


Introducing TPOT

A useful library for genetic hyperparameter tunning is TPOT:

"Consider TPOT your Data Science Assistant. TPOT is a Python Automated Machine Learning tool that optimizes machine learning pipelines using genetic programming"


Pipelines not only include the model(or multiple models) but also work on features and other aspect of the process.Plust it returns the Python code of the pipeline for you

TPOT components

The key arguments to a TPOT classifier are:

* generations: Iterations to run training for.

* population_size: The number of models to keep after each iteration

* offspring_size: Number of models to produce in each iteration

* mutation_rate: The proportion of pipelines to apply randomness to

* crossover_rate: The proportion of pipelines to breed each iteration.

* scoring: The function to determine the best models

* cv: Cross-validation strategy to use

from tpot import TPOTClassifier

tpot = TPOTClassifier(generations=3, population_size=5,
					 verbosity=2, offspring_size=10,
					 scoring='accuracy', cv = 5
)

tpot.fit(X_train, y_train)

print(tpot.score(X_test, y_test))

We will keep default values for mutation_rate and crossover_rate as they are best left to the default without deeper knowlegge on genetic programming.


Notice: No algorithm-specific hyperparameter?

"""

"""
Genetic Hyperparameter Tuning with TPOT
You're going to undertake a simple example of genetic hyperparameter tuning. TPOT is a very powerful library that has a lot of features. You're just scratching the surface in this lesson, but you are highly encouraged to explore in your own time.

This is a very small example. In real life, TPOT is designed to be run for many hours to find the best model. You would have a much larger population and offspring size as well as hundreds more generations to find a good model.

You will create the estimator, fit the estimator to the training data and then score this on the test data.

For this example we wish to use:

3 generations
4 in the population size
3 offspring in each generation
accuracy for scoring
A random_state of 2 has been set for consistency of results.
"""

# Assign the values outlined to the inputs
number_generations = 3
population_size = 4
offspring_size = 3
scoring_function = "accuracy"

# Create the tpot classifier
tpot_clf = TPOTClassifier(generations=number_generations, population_size=population_size,
                          offspring_size=offspring_size, scoring=scoring_function,
                          verbosity=2, random_state=2, cv=2)

# Fit the classifier to the training data
tpot_clf.fit(X_train, y_train)

# Score on the test set
print(tpot_clf.score(X_test,y_test ))


"""
Analysing TPOT's stability
You will now see the random nature of TPOT by constructing the classifier with different random states and seeing what model is found to be best by the algorithm. This assists to see that TPOT is quite unstable when not run for a reasonable amount of time.

"""

# Create the tpot classifier 
tpot_clf = TPOTClassifier(generations=2, population_size=4, offspring_size=3, scoring='accuracy', cv=2,
                          verbosity=2, random_state=42)

# Fit the classifier to the training data
tpot_clf.fit(X_train, y_train)

# Score on the test set
print(tpot_clf.score(X_test, y_test))


"""
Well done! You can see that TPOT is quite unstable when only running with low generations, population size and offspring. The first model chosen was a Decision Tree, then a K-nearest Neighbor model and finally a Random Forest. Increasing the generations, population size and offspring and running this for a long time will assist to produce better models and more stable results. Don't hesitate to try it yourself on your own machine!

"""