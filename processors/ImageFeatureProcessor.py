from torch.nn import Sequential


class ImageFeatureProcessor(object):

    def __init__(self, model=None, image_dim=None):
        """
        This initialises the variables used for generating the
        features and embeddings for the image
        :param model: This is the model in torchvision.models
        :param image_dim: This is the dimension of the image which is
        input into the model. The dimension can be both integer or tuple.
        """
        self.model = model
        self.image_dim = image_dim

    def __call__(self, sample):
        """
        This performs the processing on the image and generates the
        embeddings and features
        :param sample: This is the image of the form of an ndarray
        :return: Returns a torch tensor of the dimension outputed by the model
        """
        feature_extractor = Sequential(*list(self.model.children())[:-1])
        resultant_sample = feature_extractor(sample.unsqueeze(0))
        return resultant_sample.squeeze(0).squeeze(1).squeeze(1)
