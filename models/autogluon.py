from autogluon import TabularPrediction as task


def train_autogluon_model(training_data_path, test_data_path, label_to_be_predicted):
    """This method trains the autogluon model on the dataset.
    @param training_data_path: This is the path to the training dataset
    @param test_data_path: This is the path to the test dataset
    @param label_to_be_predicted: This is in string format and is the name of target feature
    @return: Returns the model and the resultant predictions in a dataframe format
    """
    training_data = task.Dataset(file_path=training_data_path, label=label_to_be_predicted)
    test_data = task.Dataset(file_path=test_data_path)
    predictor = task.fit(train_data=training_data, label=label_to_be_predicted)
    y_pred = predictor.predict(test_data)

    return predictor, y_pred
