from autogluon import TabularPrediction as task


def train_autogluon_model(training_data, test_data, label_to_be_predicted):
    """This method trains the autogluon model on the dataset.
    @param training_data: This is the training dataset. It is in the form of a dataframe
    and has the predicted feature as well present in it.
    @param test_data: This is the test data having the same format as that of the
    training data
    @param label_to_be_predicted: This is in string format and is the name of target feature
    @return: Returns the model and the resultant predictions in a dataframe format
    """
    predictor = task.fit(train_data=training_data, label=label_to_be_predicted)
    y_pred = predictor.predict(test_data)

    return predictor, y_pred
