import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier, \
    GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV, learning_curve
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

random_state = 2
classifiers = [SVC(random_state=random_state),
               DecisionTreeClassifier(random_state=random_state),
               AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state), random_state=random_state,
                                  learning_rate=0.1),
               RandomForestClassifier(random_state=random_state),
               ExtraTreesClassifier(random_state=random_state),
               GradientBoostingClassifier(random_state=random_state),
               MLPClassifier(random_state=random_state),
               KNeighborsClassifier(),
               LogisticRegression(random_state=random_state),
               LinearDiscriminantAnalysis(),
               MultinomialNB()]

classifiers_for_ensembling = [
    AdaBoostClassifier(DecisionTreeClassifier, random_state=7),
    ExtraTreesClassifier(),
    RandomForestClassifier(),
    GradientBoostingClassifier(),
    SVC(probability=True),
    MultinomialNB()
]

parameters_for_ensembling_models = [
    {
        # For Adaboost classifier
        "base_estimator__criterion": ["gini", "entropy"],
        "base_estimator__splitter": ["best", "random"],
        "algorithm": ["SAMME", "SAMME.R"],
        "n_estimators": [1, 2],
        "learning_rate": [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 1.5]
    },
    {
        # For ExtraTreeClassifier
        "max_depth": [None],
        "max_features": [1, 3, 10],
        "min_samples_split": [2, 3, 10],
        "min_samples_leaf": [1, 3, 10],
        "bootstrap": [False],
        "n_estimators": [100, 300],
        "criterion": ["gini"]
    },
    {
        # Random forest classifier
        "max_depth": [None],
        "max_features": [1, 3, 10],
        "min_samples_split": [2, 3, 10],
        "min_samples_leaf": [1, 3, 10],
        "bootstrap": [False],
        "n_estimators": [100, 300],
        "criterion": ["gini"]
    },
    {
        # Gradient boosting
        'loss': ["deviance"],
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.1, 0.05, 0.01],
        'max_depth': [4, 8],
        'min_samples_leaf': [100, 150],
        'max_features': [0.3, 0.1]
    },
    {
        # SVM Classifier
        'kernel': ['rbf'],
        'gamma': [0.001, 0.01, 0.1, 1],
        'C': [1, 10, 50, 100, 200, 300, 1000]
    },
    {
        # Multinomial Naive Bayes classifier
        'nb__alpha': [0.001, 0.01, 0.1, 1, 10, 100]
    }
]


def ensembling_cross_val_classification_first_step(X_train, Y_train):
    """This method takes as input the training dataset and
    plots the cross validation scores of different models on the
    classification task
    @param X_train: Training input
    @param Y_train: Training output
    """
    kfold = StratifiedKFold(n_splits=10)

    cv_results = []
    for classifier in classifiers:
        cv_results.append(cross_val_score(classifier,
                                          X_train, y=Y_train,
                                          scoring="accuracy",
                                          cv=kfold,
                                          n_jobs=4))

    cv_means = []
    cv_std = []
    for cv_result in cv_results:
        cv_means.append(cv_result.mean())
        cv_std.append(cv_result.std())

    cv_res = pd.DataFrame(
        {
            "CrossValMeans": cv_means,
            "CrossValerrors": cv_std,
            "Algorithm": ["SVC", "DecisionTree", "AdaBoost",
                          "RandomForest", "ExtraTrees", "GradientBoosting",
                          "MultipleLayerPerceptron", "KNeighboors", "LogisticRegression",
                          "LinearDiscriminantAnalysis"]
        }
    )

    g = sns.barplot("CrossValMeans",
                    "Algorithm",
                    data=cv_res,
                    palette="Set3",
                    orient="h",
                    **{'xerr': cv_std})
    g.set_xlabel("Mean Accuracy")
    g.set_title("Cross validation scores")


def plot_learning_curve(estimator, title,
                        X, y, ylim=None,
                        cv=None, n_jobs=-1,
                        train_sizes=np.linspace(.1, 1.0, 5)):
    """Generate a simple plot of the test and training learning curve"""
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(estimator,
                                                            X,
                                                            y,
                                                            cv=cv,
                                                            n_jobs=n_jobs,
                                                            train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()
    plt.fill_between(train_sizes,
                     train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std,
                     alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes,
                     test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std,
                     alpha=0.1,
                     color="g")
    plt.plot(train_sizes,
             train_scores_mean,
             'o-',
             color="r",
             label="Training score")
    plt.plot(train_sizes,
             test_scores_mean,
             'o-',
             color="g",
             label="Cross-validation score")
    plt.legend(loc="best")
    return plt


def grid_search_find_best_models(X_train, Y_train,
                                 kfold):
    """This method does a grid search on the selected models
    in the above defined list of models and outputs the
    best model paramters and also plots the different learning
    curves for them
    @param X_train: Training input
    @param Y_train: Training output
    @param kfold: KFole cross validator
    @return: List of best models
    """
    best_models = []

    for i in range(len(classifiers_for_ensembling)):
        model = classifiers_for_ensembling[i]
        params = parameters_for_ensembling_models[i]
        grid_search_model = GridSearchCV(model,
                                         param_grid=params,
                                         cv=kfold,
                                         scoring="accuracy",
                                         n_jobs=4,
                                         verbose=1)
        grid_search_model.fit(X_train, Y_train)
        best_models.append(grid_search_model.best_estimator_)
        plot_learning_curve(grid_search_model.best_estimator_,
                            "Learning curve for best model",
                            X_train,
                            Y_train,
                            cv=kfold)

    return best_models


def plot_feature_importance_of_tree_based_models(names_classifiers, X_train):
    """This model plots the feature importances of the different features
    by the tree based models.
    @param names_classifiers: The list of classifiers in the following format
    names_classifiers = [("AdaBoosting", ada_best), ("ExtraTrees", ExtC_best), ("RandomForest", RFC_best),
                        ("GradientBoosting", GBC_best)]
    @param X_train: The input training dataset
    """

    nrows = ncols = 2
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex="all", figsize=(15, 15))

    nclassifier = 0
    for row in range(nrows):
        for col in range(ncols):
            name = names_classifiers[nclassifier][0]
            classifier = names_classifiers[nclassifier][1]
            indices = np.argsort(classifier.feature_importances_)[::-1][:40]
            g = sns.barplot(y=X_train.columns[indices][:40],
                            x=classifier.feature_importances_[indices][:40],
                            orient='h',
                            ax=axes[row][col])
            g.set_xlabel("Relative importance", fontsize=12)
            g.set_ylabel("Features", fontsize=12)
            g.tick_params(labelsize=9)
            g.set_title(name + " feature importance")
            nclassifier += 1


def plot_ensemble_classifier_results(test, classifiers,
                                     X_train, Y_train):
    """This method plots the different results by the ensembled
    classifiers
    @param test: The testing input
    @param classifiers: The classifier list in the following format
    names_classifiers = [("AdaBoosting", ada_best), ("ExtraTrees", ExtC_best), ("RandomForest", RFC_best),
                        ("GradientBoosting", GBC_best)]
    @param X_train: The training input
    @param Y_train: The training output
    @return: The ensembled voting classifier
    """

    class_res = []
    for classifier in classifiers:
        class_res.append(pd.Series(classifier[1].predict(test), name=classifier[0]))
    ensemble_results = pd.concat(class_res, axis=1)
    sns.heatmap(ensemble_results.corr(), annot=True)
    votingC = VotingClassifier(estimators=classifiers,
                               voting='soft',
                               n_jobs=4)
    return votingC.fit(X_train, Y_train)
