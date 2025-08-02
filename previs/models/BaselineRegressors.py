import numpy as np
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn import metrics
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV


def cat_boost_regressor(X_train, y_train,
                        X_val, y_val,
                        categorical_features_indices):
    """This trains the cat boost regressor on a training data set
    @param X_train: Training dataset input
    @param y_train: Training dataset output
    @param X_val: Validation dataset input
    @param y_val: Validation dataset output
    @param categorical_features_indices: List of indices
    which are categorical features in the X_train
    @return: The trained model
    """
    model = CatBoostRegressor(iterations=50,
                              depth=3,
                              learning_rate=0.1,
                              loss_function='RMSE')
    model.fit(X_train, y_train,
              cat_features=categorical_features_indices,
              eval_set=(X_val, y_val),
              plot=True)

    return model


def linear_regression(X_train, y_train,
                      X_val, y_val):
    """This trains the linear regression model
    @param X_train: Training dataset input
    @param y_train: Training dataset output
    @param X_val: Validation dataset input
    @param y_val: Validation dataset output
    @return: The trained model, MAE, MSE and sqrt(MSE)
    """
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_val).flatten()
    y_actual = y_val.flatten()

    return (
        regressor,
        metrics.mean_absolute_error(y_actual, y_pred),
        metrics.mean_squared_error(y_actual, y_pred),
        np.sqrt(metrics.mean_squared_error(y_actual, y_pred))
    )


def ridge_regression(X_train, y_train):
    """This trains the ridge regressor
    @param X_train: Training dataset input
    @param y_train: Training dataset output
    @return: Trained model, the best parameters after grid search and the
    best score of the model
    """
    ridge = Ridge()
    parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]}
    ridge_regressor = GridSearchCV(ridge,
                                   parameters,
                                   scoring='neg_mean_squared_error',
                                   cv=5)
    ridge_regressor.fit(X_train, y_train)

    return ridge_regressor.best_estimator_, ridge_regressor.best_params_, ridge_regressor.best_score_


def lasso_regression(X_train, y_train):
    """This trains the lasso regressor
    @param X_train: Training dataset input
    @param y_train: Training dataset output
    @return: Trained model, the best parameters after grid search and the
    best score of the model
    """
    lasso = Lasso()
    parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]}
    lasso_regressor = GridSearchCV(lasso,
                                   parameters,
                                   scoring='neg_mean_squared_error',
                                   cv=5)
    lasso_regressor.fit(X_train, y_train)

    return lasso_regressor.best_estimator_, lasso_regressor.best_params_, lasso_regressor.best_score_


def elastic_net_regression(X_train, y_train):
    """This trains the elastic net regressor
    @param X_train: Training dataset input
    @param y_train: Training dataset output
    @return: Trained model, the best parameters after grid search and the
    best score of the model
    """
    elastic = ElasticNet(normalize=True)
    search = GridSearchCV(estimator=elastic,
                          param_grid={'alpha': np.logspace(-5, 2, 8),
                                      'l1_ratio': [.2, .4, .6, .8]},
                          scoring='neg_mean_squared_error',
                          n_jobs=1,
                          refit=True,
                          cv=10)
    search.fit(X_train, y_train)

    return search.best_estimator_, search.best_params_, search.best_score_


def xgboost_regression(X_train, y_train,
                      X_val, y_val):
    """This trains the xgboost regressor model
    @param X_train: Training dataset input
    @param y_train: Training dataset output
    @param X_val: Validation dataset input
    @param y_val: Validation dataset output
    @return: The trained model, MAE, MSE and RMSE
    """
    """
    objective: Determines the loss function to be used like reg:linear
    for regression problems, reg:logistic for classification problems with
    only decision, binary:logistic for classification problems with probability.
    """
    xg_reg = xgb.XGBRegressor(objective='reg:linear',
                              colsample_bytree=0.3,
                              learning_rate=0.1,
                              max_depth=5,
                              alpha=10,
                              n_estimators=10)

    xg_reg.fit(X_train, y_train)
    y_pred = xg_reg.predict(X_val).flatten()
    y_actual = y_val.flatten()

    return (
        xg_reg,
        metrics.mean_absolute_error(y_actual, y_pred),
        metrics.mean_squared_error(y_actual, y_pred),
        np.sqrt(metrics.mean_squared_error(y_actual, y_pred))
    )


def stacking_regressor(regressor_list, X_train,
                       y_train):
    """This method stacks different regressors together
    into an ensemble model to do the predictions
    @param regressor_list: List of regression models
    @param X_train: Training dataframe
    @param y_train: Test dataframe
    @return: The trained ensembled model
    """
    model = StackingRegressor(estimators=regressor_list,
                              final_estimator=LinearRegression())
    model.fit(X_train, y_train)
    return model

