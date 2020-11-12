# Feature engineering

Feature engineering is very important in machine learning and deep learning problems.
Generating features usually takes a lot of time and depth of knowledge of the
problem statement. However, we can also use sophisticated techniques to do that.
We will be going over some techniques and tools which you can use to make this 
task of feature engineering a little easier.

## Feature tools

Featuretools is an open source library for performing automated feature engineering. 
It is a great tool designed to fast-forward the feature generation process, thereby 
giving more time to focus on other aspects of machine learning model building. 
In other words, it makes your data “machine learning ready”.

There are three major components of the package that we should be aware of:

* Entities
* Deep Feature Synthesis (DFS)
* Feature primitives

An **Entity** can be considered as a representation of a Pandas DataFrame. 
A collection of multiple entities is called an Entityset.

**Deep Feature Synthesis** (DFS) has got nothing to do with deep learning. 
Don’t worry. DFS is actually a Feature Engineering method and is the backbone of 
Featuretools. It enables the creation of new features from single, as well as multiple dataframes.

DFS create features by applying **Feature primitives** to the Entity-relationships in 
an EntitySet. These primitives are the often-used methods to generate features manually. 
For example, the primitive “mean” would find the mean of a variable at an aggregated level.

To install Feature Tools, simply use *pip* fir the same
```
pip install featuretools
```
Let us say we have a dataframe **DF**. We need to generate more features from this.
Always remember to use this once you have cleaned your dataset completely.

```python
from previs.featuregeneration.featuregeneration import engineer_features_using_feature_tools

df_new = engineer_features_using_feature_tools(DF, 'id', key_variable_list_map)
```
The **key_variable_list_map** is defined as a dictionary having the following key 
value pairs:
    
    secondary_key_1 -> ['col1', 'col2', 'col3'],
    secondary_key_2 -> ['col4', 'col5', 'col6']
    
There can be a scenario where the dataframe constitutes of more than one datasets.
These multiple datasets can be further combined by the different secondary keys. 

## Text feature generation

This is used to generate important and essential features for the textual data.
We have ensured to incorporate almost all the essential text preprocessing tools
and techniques in this method. 

There are two parts of this text feature preprocessing and we advise you to follow
the follow the following steps.

### Step 1 : Generate the features as per your requirement
We advise you to use the method ```generate_features_from_text_column``` to 
generate the textual features. As you can see, there are different features
which are generated in this method, however dependeing upon your problem, you 
may not be required to generate all of them. Therefore, it is essential for you
to peruse and go through the method and modify it as per your requirements.

The method is divided into three parts.

#### Part 1 : Before ```generate_lsi_features```
This part performs feature generation on the raw textual data. This is important
since there are certain textual features which we want to be generated from the
raw text rather than the processed text. Therefore, certain features such as captial words
et cetera as generated here. You can either add your own here, remove some from here,
or change the order of the feature generation from the ones below.

#### Part 2 : ```generate_lsi_features``` method
This method does two things. First it internally invokes ```build_clean_corpus```
to clean the corpus by performing a lot of text processing operations. Please note 
that it **does not** perform *stemming* and *lemmatisation* since we are generating
features based on adjective and verb count.

If we are not generating features based on POS tags, then in that case, it is 
a good idea to perform stemming and lemmatisation as well.

Post this, it uses LSI and BOW model to perform dimensionality reduction on the
columns of textual dataset. We are using gensim here to do this. However, you
can use other models from gensim as well. We have used the LSI model and used an efficient
implementation of the same. Thsi implementation ensures everything is done at
run time and does not load the complete dataset at once in memory. This is done
to ensure speed and accuracy as well.

#### Part 3 : After ```generate_lsi_features```
We generate multiple features on the cleaned dataset post the preprocessing.
Some of them include POS tags and their counts. Upon performing these steps, 
we would then perform the next Step in our text processing pipeline. 

### Step 2 : Visualisations and analysis
Once the features are ready, it is important to perform some analysis on the same.
We need to analyse the importance of the features, for this we use a correlation
matrix. 

We use the following to plot the correlation matrix
```python
from previs.visualisations.BasicPlots import correlation_matrix

correlation_matrix(dataframe, ['text_col_name', 'feat_cat_1'])
```

Now, we have got a good understanding of the correlation matrix and the
different feature importances and their relationships. Now, we should plot
the frequency of the **n-grams** as well to get more understanding as well.

Use ```plot_n_grams``` to do this. Select **n-grams** to range from 1 to 3.
This will give you more understanding of the dataset as well.

Post this, plot a word cloud to see whether you get a more understanding of 
the **n-gram** counts from the plot as well as the word cloud.

These steps should ideally help you in performing the text processing you
require for your dataset. 
