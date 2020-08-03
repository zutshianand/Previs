# Visualisations

Visualisation tools are of extreme importance when it comes to EDA and 
understanding the dataset. These are also very complex and difficult when it 
comes to implement as well. There are different problem categories when
one wants to visualise the data. These can be very vast and we do not intend to
cover them all here. However, we will cover most of them so that it gives a person
the confidence to explore and go ahead to implement new ones themselves.

| Relationship between A | Relationship between B | What to use? | Remarks |
| ---------------------- | ---------------------- | ------------ | ------- |
| Single numerical       | Single numerical       | ```numerical_vs_numerical_or_categorical,numerical_vs_numerical``` |
| Single numerical       | Single categorical     | ```numerical_vs_numerical_or_categorical,categorical_vs_numerical_bubble,categorical_vs_numerical_bar``` | Pass the categorical variable name in the second numerical attribute |
| Two or three numerical | Single categorical     | ```numerical_vs_numerical_or_categorical``` |
| Single numerical       | Time series            | ```numerical_vs_time``` |
| Single numerical       | More than one categorical    | ```numerical_vs_multiple_cat``` |
| Single continuous variable    | Single categorical | ```continuous_var_distribution_vs_single_cat_var,continous_var_vs_single_cat_vars_waves``` |
| Two continuous variable | Single categorical | ```two_continuous_var_vs_single_cat_vars_heatmap``` |
| One categorical | Multiple categorical | ```multiple_cat_vars``` |

Apart from these, there are 3d graphs as well which we have not added. 
For the guide to read and how to use these graphs and plotting techniques, please
refer to the above table and the attached pdf in this directory. The pdf has very useful
images and details for the different use cases and the images which will aid you
in taking the right decision when it comes to building the plots.