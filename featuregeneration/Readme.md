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
from featuregeneration.featuregeneration import engineer_features_using_feature_tools

df_new = engineer_features_using_feature_tools(DF, 'id', key_variable_list_map)
```
The **key_variable_list_map** is defined as a dictionary having the following key 
value pairs:
    
    secondary_key_1 -> ['col1', 'col2', 'col3'],
    secondary_key_2 -> ['col4', 'col5', 'col6']
    
There can be a scenario where the dataframe constitutes of more than one datasets.
These multiple datasets can be further combined by the different secondary keys. 