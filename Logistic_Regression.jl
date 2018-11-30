# Logistic Regression in Julia

# import Pkg
using Pkg


using RDatasets: dataset

iris = dataset("datasets", "iris")


# get df size
using DataFrames

# number of columns
n_col = size(iris)[2]
println("Number of columns: $n_col")

# number of rows
n_row = size(iris)[1]
println("Number of rows: $n_row")


X = convert(Array, iris[[:SepalLength, :SepalWidth, :PetalLength, :PetalWidth]])
y = convert(Array, iris[:Species])


using MLDataUtils

# shuffle the data so its not in order when we split it up
Xs, Ys = shuffleobs((transpose(X), y))

#now split the data into training sets and validation sets
(X_train1, y_train1), (X_test1, y_test1) = splitobs((Xs, Ys); at = 0.67)
    
# need to convert the split data back into arrays
X_train = Array(transpose(X_train1))
y_train = Array(y_train1)
X_test = Array(transpose(X_test1))
y_test = Array(y_test1)


using ScikitLearn

@sk_import linear_model: LogisticRegression


using ScikitLearn.GridSearch: GridSearchCV

# fit on training data
gridsearch = GridSearchCV(LogisticRegression(), Dict(:C => 0.1:0.1:2.0))
fit!(gridsearch, X_train, y_train)
println("Best parameters: $(gridsearch.best_params_)")


# cross-validate model
using ScikitLearn.CrossValidation: cross_val_score

cross_val_scores = cross_val_score(LogisticRegression(C = 1.3), X, y; cv=5)  # 5-fold


# print the mean cross validation score
using Statistics

mean_cross_val = mean(cross_val_scores)
println("Average cross validation score: $mean_cross_val")


using PyPlot, Statistics

plot([cv_res.parameters[:C] for cv_res in gridsearch.grid_scores_],
     [Statistics.mean(cv_res.cv_validation_scores) for cv_res in gridsearch.grid_scores_])