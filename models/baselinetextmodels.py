import numpy as np
import xgboost
from sklearn import decomposition

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB


def multiclass_logloss(actual, predicted, eps=1e-15):
    """Multi class version of Logarithmic Loss metric.
    :param actual: Array containing the actual target classes
    :param predicted: Matrix with class predictions, one probability per class
    """
    # Convert 'actual' to a binary array if it's not already:
    if len(actual.shape) == 1:
        actual2 = np.zeros((actual.shape[0], predicted.shape[1]))
        for i, val in enumerate(actual):
            actual2[i, val] = 1
        actual = actual2

    clip = np.clip(predicted, eps, 1 - eps)
    rows = actual.shape[0]
    vsota = np.sum(actual * np.log(clip))
    return -1.0 / rows * vsota


def logistic_regression_using_different_models(xtrain, xvalid, ytrain, yvalid):
    """This method extracts two different types of features
    These are the TFIDF and CountVectoriser and fits four different
    baseline models on it. It then computes the multiclass log loss for the same.
    @param xtrain: Training dataframe
    @param xvalid: Validation dataframe
    @param ytrain: Training output
    @param yvalid: Validation output
    """
    tfv = TfidfVectorizer(min_df=3,
                          max_features=None,
                          strip_accents='unicode',
                          analyzer='word',
                          token_pattern=r'\w{1,}',
                          ngram_range=(1, 3),
                          use_idf=1,
                          smooth_idf=1,
                          sublinear_tf=1,
                          stop_words='english')

    ctv = CountVectorizer(analyzer='word',
                          token_pattern=r'\w{1,}',
                          ngram_range=(1, 3),
                          stop_words='english')

    logreg = LogisticRegression(C=1.0)
    multnb = MultinomialNB()
    svd = decomposition.TruncatedSVD(n_components=120)
    xgb_clf = xgboost.XGBClassifier(max_depth=7,
                                    n_estimators=200,
                                    colsample_bytree=0.8,
                                    subsample=0.8,
                                    nthread=10,
                                    learning_rate=0.1)

    features = [tfv, ctv]
    models = [logreg, multnb, svd, xgb_clf]

    for feature in features:
        for model in models:
            feature.fit(list(xtrain) + list(xvalid))
            xtrain_fit = feature.transform(xtrain)
            xvalid_fit = feature.transform(xvalid)

            model.fit(xtrain_fit, ytrain)
            predictions = model.predict_proba(xvalid_fit)
            print("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))
