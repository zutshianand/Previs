# Utilities

This directory contains the different utilities used for the main processing
operations in this repository. However, there are certain important
operations which are defined in this directory which can be used extensively
for a machine learning problem. We will explain them here and their usages.

## Distance between two strings

Given a dataframe and the text column mame as the dataset corpus, and two 
sentences in string format, we use fastText to predict the distance between two strings.

The method ```get_distance_between_sentences``` in ```TextProcessingUtils``` is 
used in doing so. Gensim provides other metrics for giving the distance between two
sentences and pieces of text, however fasttext is used here since this provides us with
a better result of similarity,

## Find most similar text from a corpus

Given a dataframe and the text column name as the dataset corpus, and a target sentence
we have to find out the set of strings ordered in descending order of having the most 
similarity to the target sentence. 

This is done by using the method ```find_most_similar_texts``` in ```TextProcessingUtils```.

This uses the LsiModel for doing this. Gensim also provides the LDAModel amongst many others
for doing this. Feel free to use them as well. However for simplicity we have used this.
