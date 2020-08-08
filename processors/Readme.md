# Processors

This directory contains all the different processors which will be required
by this repository. These processors can be used either along the different
models and datasets or without them.

These are explained below.

## ImageFeatureProcessor

This processors is used to process an input image in the form of an ndarray.
This outputs the embeddings and features of the image as per the model passed.

You can refer to its implementation and use in the datasets directory.

## TabularProcessor

This is used to process a dataset in tabular format. While designing this
processor, it was assumed that the dataset would all be either numerical or
categorical. However, since the dataset can also have textual features in it, 
we will either have to process the textual features beforehand using the 
```featuregeneration``` methods and techniques or incorporate the ```TextProcessor```
in this processor.

## TextProcessor

This is used to process the text as per the requirement. It has two parts in it.
Let us go over hwo to use both of them in detail.

### Part 1 : Check coverage of text and words

This is used to check the coverage of the words. This is used to check if we need
to remove some words from our vocabulary or not. In case we are using pretrained embeddings
in our model like GLoVe and the embeddings do not cover a large part of our text dataset,
then training it using those embeddings is a waste of time. Thus, we need to ensure
proper coverage of the same. We do this by using the method ``vocab_check_coverage``

Before doing this, we use the method ```load_embed``` to load the embeddings we desire.
Since embeddings are large in size, we do not provide them in our repository, therefore
we encourage you to download them from the internet and use them here in this method.

Upon doing this, you would get a good idea about the different words which are 
not getting covered by your embeddings and you can either remove them or add some
unique embeddings for them.

### Part 2 : Preprocess your text

This is used to preprocess your text based on different preprocessing techniques.
All the preprocessing techniques which are listed here are aimed at being comprehensive,
however you can either change the order as per your convenience or add more if you want.
As of now, we have commented out lemmatisation and stemming since they conflict with the
POS generation done in the featuregeneration directory, however it is upto your problem 
statement whether to include it or not. 