import string
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as py
import regex as re
import seaborn as sns
from gensim import corpora, models
from plotly import tools
from wordcloud import STOPWORDS, ImageColorGenerator, WordCloud

from previs.data.badwords import BAD_WORDS
from previs.processors.TextProcessor import tag_part_of_speech
from previs.util.TextProcessingUtils import (CLEANED_TEXT_COL_NAME,
                                             convert_text_col_to_corpus,
                                             get_stopwords)

color = sns.color_palette()

py.init_notebook_mode(connected=True)

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999


def plot_wordcloud(text, mask=None, max_words=200, max_font_size=100, figure_size=(24.0, 16.0),
                   title=None, title_size=40, image_color=False):
    """This method plots the word cloud for the text column
    in the dataframe
    @param text: This is the dataframe['text_col_name']
    @param mask: This is not required.
    @param max_words: The maximum number fo words which we will plot
    @param max_font_size: The max font size
    @param figure_size: Fig size
    @param title: Title of the plot
    @param title_size: Size of the title
    @param image_color: Colour of the image
    """
    stopwords = get_stopwords()
    wordcloud = WordCloud(background_color='black',
                          stopwords=stopwords,
                          max_words=max_words,
                          max_font_size=max_font_size,
                          random_state=42,
                          width=800,
                          height=400,
                          mask=mask)
    wordcloud.generate(str(text))
    plt.figure(figsize=figure_size)
    if image_color:
        image_colors = ImageColorGenerator(mask)
        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear")
        plt.title(title, fontdict={'size': title_size, 'verticalalignment': 'bottom'})
    else:
        plt.imshow(wordcloud)
        plt.title(title, fontdict={'size': title_size, 'color': 'black', 'verticalalignment': 'bottom'})
    plt.axis('off')
    plt.tight_layout()


def generate_lsi_features(dataframe, text_col_name,
                          num_topics, batchsize):
    """This method cleans the dataset and generates the LSI
    features using Gensim. Refer to Readme.md for more details on this.
    @param dataframe: Dataframe name
    @param text_col_name: Name of the text data column
    @param num_topics: Number of topics which it will generate (10-20)
    @param batchsize: Batchsize it will use to generate per iteration.
    @return: the cleaned dataframe along with the topics and the lsi model
    """
    res_lists = {}
    for i in range(num_topics):
        res_lists[i] = []
    corpus, texts, dictionary = convert_text_col_to_corpus(dataframe, text_col_name, batchsize)
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    lsi_model = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=num_topics)
    corpus_lsi = lsi_model[corpus_tfidf]
    for i in range(num_topics):
        for attr in corpus_lsi:
            res_lists[i].append(attr[i][1])
    for i in range(batchsize, len(texts), batchsize):
        text = texts[i: i + batchsize]
        dictionary = corpora.Dictionary(text)
        corpus = [dictionary.doc2bow(txt) for txt in text]
        tfidf = models.TfidfModel(corpus)
        corpus_tfidf = tfidf[corpus]
        lsi_model.add_documents(corpus_tfidf)
        corpus_lsi = lsi_model[corpus_tfidf]
        for i in range(num_topics):
            for attr in corpus_lsi:
                res_lists[i].append(attr[i][1])
    for i in range(num_topics):
        dataframe[text_col_name + 'attr_' + str(i + 1)] = res_lists[i]

    return dataframe, lsi_model


def generate_ngrams(text, n_gram=1):
    """This method id used to genrate the n-grams which
    are used to plot in the plot_n_grams method
    @param text: The string text
    @param n_gram: Number of grams (1,2,3..)
    @return: List of n-grams
    """
    token = [token for token in text.lower().split(" ") if token != "" if token not in STOPWORDS]
    ngrams = zip(*[token[i:] for i in range(n_gram)])
    return [" ".join(ngram) for ngram in ngrams]


def horizontal_bar_chart(df, color):
    """Plots a bar chart of the word and word count.
    This is used by the plotting the n-grams of a dataframe text
    column.
    @param df: The dataframe name
    @param color: The color used
    @return: Trace which will be used to plot the graph
    """
    trace = go.Bar(
        y=df["word"].values[::-1],
        x=df["wordcount"].values[::-1],
        showlegend=False,
        orientation='h',
        marker=dict(
            color=color,
        ),
    )
    return trace


def plot_n_grams(dataframe, text_col_name, plot_title, n_gram):
    """This method plots the different n-grams for visualisation purpose.
    @param dataframe: The name of the dataframe
    @param text_col_name: The text column name
    @param plot_title: The title plot
    @param n_gram: The number of the n-grma (1,2,3...)
    """
    freq_dict = defaultdict(int)
    for sent in dataframe[text_col_name]:
        for word in generate_ngrams(sent, n_gram):
            freq_dict[word] += 1
    fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
    fd_sorted.columns = ["word", "wordcount"]
    trace = horizontal_bar_chart(fd_sorted.head(50), 'blue')

    fig = tools.make_subplots(rows=1, cols=1, vertical_spacing=0.04,
                              subplot_titles=[plot_title])
    fig.append_trace(trace, 1, 1)
    fig['layout'].update(height=1200, width=900, paper_bgcolor='rgb(233,233,233)', title="Word Count Plots")
    py.iplot(fig, filename='word-plots')


def word_count(text):
    """Returns the number of words in a text string
    after removing the stopwords from it.
    @param text: The string of text
    @return: Length of the string in int
    """
    try:
        text = text.lower()
        regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]')
        txt = regex.sub(" ", text)
        words = [w for w in txt.split(" ") if w not in STOPWORDS and len(w) > 3]
        return len(words)
    finally:
        return 0


def generate_features_from_text_column(dataframe, text_col_name):
    """This method performs the many processing and feature generation
    tasks for the textual dataset. Please refer to Readme.md for the operations
    and how to use this method.
    @param dataframe: The name of the dataframe.
    @param text_col_name: The columns name of the textual data
    """
    dataframe["num_stopwords"] = dataframe[text_col_name].apply(
        lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS]))
    dataframe["num_punctuations"] = dataframe[text_col_name].apply(
        lambda x: len([c for c in str(x) if c in string.punctuation]))
    dataframe["num_words_upper"] = dataframe[text_col_name].apply(
        lambda x: len([w for w in str(x).split() if w.isupper()]))
    dataframe["num_words_title"] = dataframe[text_col_name].apply(
        lambda x: len([w for w in str(x).split() if w.istitle()]))
    dataframe["mean_word_len"] = dataframe[text_col_name].apply(
        lambda x: np.mean([len(w) for w in str(x).split()]))
    dataframe['num_exclamation_marks'] = dataframe[text_col_name].apply(
        lambda x: x.count('!'))
    dataframe['num_question_marks'] = dataframe[text_col_name].apply(
        lambda x: x.count('?'))
    dataframe['num_symbols'] = dataframe[text_col_name].apply(
        lambda x: sum(x.count(w) for w in '*&$%'))
    dataframe['num_unique_words'] = dataframe[text_col_name].apply(
        lambda x: len(set(w for w in x.split())))

    dataframe, _ = generate_lsi_features(dataframe, text_col_name, 3, 50)
    text_col_name = CLEANED_TEXT_COL_NAME

    dataframe["num_words"] = dataframe[text_col_name].apply(
        lambda x: word_count(x))
    dataframe["num_unique_words"] = dataframe[text_col_name].apply(
        lambda x: len(set(str(x).split())))
    dataframe["num_chars"] = dataframe[text_col_name].apply(
        lambda x: len(str(x)))
    dataframe['words_vs_unique'] = dataframe['num_unique_words'] / dataframe['num_words']
    dataframe["badwordcount"] = dataframe[text_col_name].apply(
        lambda x: sum(x.count(w) for w in BAD_WORDS))
    dataframe["normword_badwords"] = dataframe["badwordcount"] / dataframe['num_words']

    dataframe['nouns'], dataframe['adjectives'], dataframe['verbs'] = zip(*dataframe[text_col_name].apply(
        lambda text: tag_part_of_speech(text)))
    dataframe['nouns_vs_length'] = dataframe['nouns'] / dataframe['num_words']
    dataframe['adjectives_vs_length'] = dataframe['adjectives'] / dataframe['num_words']
    dataframe['verbs_vs_length'] = dataframe['verbs'] / dataframe['num_words']
    dataframe['nouns_vs_words'] = dataframe['nouns'] / dataframe['num_words']
    dataframe['adjectives_vs_words'] = dataframe['adjectives'] / dataframe['num_words']
    dataframe['verbs_vs_words'] = dataframe['verbs'] / dataframe['num_words']
