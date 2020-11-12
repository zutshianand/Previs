# Models and Implementation

We provide different implementations of training different models in this subdirectory here.
Machine learnign and deep learning uses both traditional and modern models.

Traditional models include models like SVM, Random forest etc. These models, although
easy to implement in code, takes up some time for the user to search online and implement.
Also, there are certain issues when implementing these models. These include deciding
which one to implement in case of regression and also classification.

We try to solve these problmes by providing a well accessible codebase for almost all the
traditional and modern models.

In case of regression as well as classification.

## Regression

We provide the following for regresison use cases:
* Catboost 
* Linear regression
* Ridge regression
* Lasso regression
* Elastic net regression
* XGBoost regression

We also provide how to use the state of the art model **AutoGluon** in Python.

## Baseline text models

We provide to you logistic regression on a text based dataset using different models.
Please refer to the code for more undertsanding. It fits multiple models to the dataset using 
different feature extractors like TFIDF and CountVectoriser. Post that, it outputs the scores for each 
one of them.

## Stacking ensembler

We provide to you a stacking ensembler as well. Please refer to the ```parameters_for_ensembling_models```
for the different parameters and models to be used. 

We perform the ```ensembling_cross_val_classification_first_step``` to get the results
of the different models. Out of the best couple of models, we select the ones which are to be used
in the ensemble technique. 

Then we do a grid search to find the best model and also plot their curves to visualise their
learnings via ```grid_search_find_best_models```. We can also plot the different feature importances 
of the tree based models if we require that using ```plot_feature_importance_of_tree_based_models```.

In the end, we can output the classifier results using the ```plot_ensemble_classifier_results```.