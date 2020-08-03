from random import random

import matplotlib
import numpy as np

import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff


def numerical_vs_numerical_or_categorical(dataframe, numerical_col_1,
                                          numerical_col_2, title_of_plot,
                                          x_axis_title, y_axis_title,
                                          categorical_col=None, numerical_col_3=None):
    """This method plots the a scatter plot between a numerical value,
    and a categorical value. It also supports when there are 1 or 2 numerical
    values along with a categorical value.
    @param dataframe: The dataframe
    @param numerical_col_1: The first numerical value
    @param numerical_col_2: The second numerical value
    @param title_of_plot: Title of the plot
    @param x_axis_title: X axis title
    @param y_axis_title: Y axis title
    @param categorical_col: Categorical column name (optional)
    @param numerical_col_3: The third numerical value (optional)
    """
    fig = px.scatter(dataframe, x=numerical_col_1, y=numerical_col_2, color=categorical_col, size=numerical_col_3)
    fig.update_layout(title=title_of_plot, xaxis_title=x_axis_title, yaxis_title=y_axis_title)
    fig.show()


def numerical_vs_numerical(dataframe, numerical_col_1,
                           numerical_col_2, title_of_plot,
                           x_axis_title, y_axis_title):
    """This method does a scatter plot between a numerical
    and another numerical value
    @param dataframe: The dataframe
    @param numerical_col_1: The first numerical value
    @param numerical_col_2: The second numerical value
    @param title_of_plot: Title of the plot
    @param x_axis_title: X axis title
    @param y_axis_title: Y axis title
    """
    fig = go.Figure(data=go.Scatter(x=dataframe[numerical_col_1],
                                    y=dataframe[numerical_col_2],
                                    mode='markers',
                                    marker_color=dataframe[numerical_col_2]))  # hover text goes here
    fig.update_xaxes(range=[0, 6])
    fig.update_layout(title=title_of_plot, xaxis_title=x_axis_title,
                      yaxis_title=y_axis_title)
    fig.show()


def numerical_vs_time(dataframe, numerical_col_list,
                      date_col_1, title_of_plot,
                      x_axis_title, y_axis_title):
    """This plots a line graph between numerical value
    and series data - time.
    @param dataframe: The dataframe
    @param numerical_col_list: The different names of the numerical cols
    @param date_col_1: The col name of a series column
    @param title_of_plot: Title of the plot
    @param x_axis_title: X axis title
    @param y_axis_title: Y axis title
    """
    hex_colors_names = []
    for name, hex in matplotlib.colors.cnames.items():
        hex_colors_names.append(name)

    valid_markers = ([item[0] for item in matplotlib.markers.MarkerStyle.markers.items() if
                      item[1] is not 'nothing' and not item[1].startswith('tick')
                      and not item[1].startswith('caret')])
    markers = np.random.choice(valid_markers, len(numerical_col_list), replace=False)

    fig = go.Figure()
    dataframes_list = []
    for col_name in numerical_col_list:
        temp_dataframe = dataframe[[col_name, date_col_1]].groupby(date_col_1).sum().reset_index()
        dataframes_list.append(temp_dataframe)

    for i in range(len(dataframes_list)):
        df = dataframes_list[i]
        col_name = numerical_col_list[i]
        fig.add_trace(go.Scatter(x=df[date_col_1], y=df[col_name], name=col_name,
                                 line=dict(color=random.choice(hex_colors_names),
                                           width=4, dash=markers[i])))

    fig.update_xaxes(rangeslider_visible=True)
    fig.update_layout(title=title_of_plot,
                      xaxis_title=x_axis_title,
                      yaxis_title=y_axis_title)
    fig.show()


def categorical_vs_numerical_bubble(dataframe, numerical_col_1,
                                    cat_col_1, x_axis_title,
                                    y_axis_title, plot_title):
    """This method plots the relationship between numerical
    and categorical data using bubbles
    @param dataframe: The dataframe
    @param numerical_col_1: The numerical col name
    @param cat_col_1: The categ col name
    @param x_axis_title: The x axis title
    @param y_axis_title: The y axis title
    @param plot_title: The plot title name
    """
    temp_dataframe = dataframe[cat_col_1].value_counts().to_frame().reset_index().rename(
        columns={'index': cat_col_1, cat_col_1: numerical_col_1})

    fig = go.Figure(data=[go.Scatter(x=temp_dataframe[cat_col_1], y=temp_dataframe[numerical_col_1],
                                     mode='markers',
                                     marker=dict(color=temp_dataframe[cat_col_1],
                                                 size=temp_dataframe[numerical_col_1],
                                                 showscale=True))])
    fig.update_layout(title=plot_title,
                      xaxis_title=x_axis_title,
                      yaxis_title=y_axis_title)
    fig.show()


def categorical_vs_numerical_bar(dataframe, numerical_col_1,
                                 cat_col_1, x_axis_title,
                                 y_axis_title, plot_title):
    """This method shows the relationship between a
    single categorical data and a numerical data. This
    plots a simple bar graph plot.
    @param dataframe: The dataframe
    @param numerical_col_1: The numerical col name
    @param cat_col_1: The category col name
    @param x_axis_title: The x axis title
    @param y_axis_title: The y axis title
    @param plot_title: The plot title
    """
    temp_dataframe = dataframe[cat_col_1].value_counts().to_frame().reset_index().rename(
        columns={'index': cat_col_1, cat_col_1: numerical_col_1})

    fig = go.Figure(go.Bar(x=temp_dataframe[cat_col_1], y=temp_dataframe[numerical_col_1],
                           marker={'color': temp_dataframe[numerical_col_1],
                                   'colorscale': 'Viridis'},
                           text=temp_dataframe[numerical_col_1],
                           textposition="outside", ))
    fig.update_layout(title_text=plot_title, xaxis_title=x_axis_title,
                      yaxis_title=y_axis_title)
    fig.show()


def numerical_vs_multiple_cat(dataframe, numerical_col_1,
                              cat_col_list, plot_title,
                              use_sunburst=False, use_heatmap=False):
    """This method plots the relationship between a single numerical
    value and multiple categorical values.

    Case 1 : You can use a bar graph to do this. But it will only support
    upto 4 categorical values. If there is more than 4, then use sunburst.

    Case 2 : It will support more than 4 categorical variables for one
    numerical variable.

    Case 3 : Heatmap is only possible when the categorical variables
    are 2. It shows a relationship between two categorical variables.

    @param dataframe: The dataframe
    @param numerical_col_1: The numerical col name
    @param cat_col_list: The categorical col names list
    @param plot_title: The title of the plot
    @param use_sunburst: Whether to use sunburst or not
    @param use_heatmap: Whether to use heatmap or not
    """
    # Sunburst support more than 4 categorical variables
    complete_col_list = cat_col_list
    complete_col_list.append(numerical_col_1)
    temp_dataframe = dataframe[complete_col_list].groupby(cat_col_list).agg('sum').reset_index()

    if use_sunburst:
        fig = px.sunburst(temp_dataframe, path=cat_col_list, values=numerical_col_1)
        fig.update_layout(title=plot_title, title_x=0.5)
        fig.show()
    elif use_heatmap is True and len(cat_col_list) == 2:
        numeric_data = []
        for i in dataframe[cat_col_list[0]].unique():
            numeric_data.append(
                dataframe[dataframe[cat_col_list[0]] == i][[cat_col_list[1], numerical_col_1]].groupby(
                    cat_col_list[1]).sum().reset_index()[numerical_col_1])
        fig = go.Figure(data=go.Heatmap(z=numeric_data,
                                        x=dataframe[cat_col_list[0]].unique(),
                                        y=dataframe[cat_col_list[1]].unique(),
                                        colorscale='Viridis'))
        fig.update_layout(title=plot_title,
                          xaxis_nticks=30)
        fig.show()
    else:
        fig = px.bar(temp_dataframe, x=cat_col_list[0],
                     y=numerical_col_1, color=cat_col_list[1],
                     barmode="group", facet_row=cat_col_list[2],
                     facet_col=cat_col_list[3], )
        fig.update_layout(title_text=plot_title)
        fig.show()


def continuous_var_distribution_vs_single_cat_var(dataframe, numerical_col_1,
                                                     cat_col_name, cat_col_list,
                                                     plot_title):
    """This method plots multiple box plots for a single
    continuous variable distribution vs multiple categorical
    variable distribution.
    @param dataframe: The dataframe
    @param numerical_col_1: The numerical col name
    @param cat_col_name: The categorical col name
    @param cat_col_list: The different values of the categorical variable
    that it takes
    @param plot_title: The plot title name
    """
    hex_colors_names = []
    for name, hex in matplotlib.colors.cnames.items():
        hex_colors_names.append(name)

    dataframe_list = []
    for col_name in cat_col_list:
        dataframe_list.append(dataframe[dataframe[cat_col_name] == col_name][numerical_col_1])

    fig = go.Figure()
    for i in range(len(dataframe_list)):
        df = dataframe_list[i]
        fig.add_trace(go.Box(y=df,
                             jitter=0.3,
                             pointpos=-1.8,
                             boxpoints='all',  # Display all points in plot
                             marker_color=hex_colors_names[i],
                             name=cat_col_name[i]))
    fig.update_layout(title=plot_title)
    fig.show()


def continous_var_vs_single_cat_vars_waves(dataframe, numerical_col_1,
                                       cat_col_name, cat_col_list,
                                       binsize, show_hist=False):
    """This method plots distplot and also waves without
    the histogram in the background when there is a single
    continuous variable vs a single categorical vairable
    @param dataframe: The dataframe
    @param numerical_col_1: The numerical col name
    @param cat_col_name: The categorical col name
    @param cat_col_list: The different values the categorical
    variable takes
    @param binsize: The bin size for the single categorical variable
    @param show_hist: Whether to show the histogram or not
    """
    hex_colors_names = []
    for name, hex in matplotlib.colors.cnames.items():
        hex_colors_names.append(name)
    hex_colors_names = hex_colors_names[:len(cat_col_list)]

    dataframe_list = []
    for col_name in cat_col_list:
        dataframe_list.append(dataframe[dataframe[cat_col_name] == col_name][numerical_col_1])

    binsize_list = []
    for i in range(len(cat_col_list)):
        binsize_list.append(binsize)

    fig = ff.create_distplot(dataframe_list,
                             cat_col_list,
                             show_hist=show_hist,
                             colors=hex_colors_names,
                             bin_size=binsize_list)
    fig.show()


def two_continuous_var_vs_single_cat_vars_heatmap(dataframe, numerical_col_1,
                                                  numerical_col_2, cat_col_list,
                                                  plt_title):
    """This method plots a heatmap for two continuous variables
    with one categorical variable. The relationship between the two
    continuous vairables is seen as the categorical variable changes.

    We can simply provide a list having two None objects for the cat_col_list
    if we want a heatmap for two continuous variables.
    @param dataframe: The dataframe
    @param numerical_col_1: The first numerical col name
    @param numerical_col_2: The second numerical col name
    @param cat_col_list: The col names of the categorical data
    @param plt_title: The plot title
    """
    fig = px.density_heatmap(dataframe,
                             x=numerical_col_1,
                             y=numerical_col_2,
                             facet_row=cat_col_list[0],
                             facet_col=cat_col_list[1])
    fig.update_layout(title=plt_title)
    fig.show()


def multiple_cat_vars(dataframe, cat_col_list,
                      cat_fourth_var, plt_title):
    """This method plots the relationship between more than
    one categorical variable. The list of categorical vairables
    has to be of length 3 or less
    @param dataframe: The dataframe
    @param cat_col_list: The list of column names
    @param cat_fourth_var: The fourth variable if required.
    @param plt_title: The title of the plot
    """
    fig = px.parallel_categories(dataframe,
                                 dimensions=cat_col_list,
                                 color=cat_fourth_var,
                                 color_continuous_scale=px.colors.sequential.Inferno)  # Color for Pclass
    fig.update_layout(title=plt_title)
    fig.show()
