from collections import defaultdict

from gensim import corpora, models, similarities
from gensim.models.fasttext import FastText as FT_gensim
from wordcloud import STOPWORDS

from processors.TextProcessor import TextProcessor

CLEANED_TEXT_COL_NAME = 'cleaned_text'


def convert_text_col_to_corpus(dataframe, text_col_name,
                               batchsize):
    """This method converts the text col
    of dataframe to a BOW corpus
    @param dataframe: Dataframe
    @param text_col_name: Text column name
    @param batchsize: Batch size to be used
    @return: Corpus, tokenized corpus and a dictionary
    """
    documents, text_col_name = build_clean_corpus(dataframe, text_col_name)
    texts = build_tokenised_corpus(documents)
    text = texts[0: batchsize]
    dic = corpora.Dictionary(text)
    corpus = [dic.doc2bow(txt) for txt in text]
    return corpus, texts, dic


def build_clean_corpus(dataframe, text_col_name):
    """This method builds a clean dataframe
    by applying different text processing techniques
    @param dataframe: Dataframe
    @param text_col_name: Text column name
    @return: Cleaned datafrmae and the cleaned text col name
    """
    text_processor = TextProcessor()
    documents = []
    dataframe[CLEANED_TEXT_COL_NAME] = dataframe[text_col_name].apply(lambda x: text_processor.process(x))
    text_col_name = CLEANED_TEXT_COL_NAME
    for sent in dataframe[text_col_name]:
        documents.append(sent)
    return documents, text_col_name


def build_tokenised_corpus(documents):
    """This method build a tokenised corpus
    @param documents: List of strings
    @return: tokenised corpus after removing frequent words
    """
    stop_list = get_stopwords()
    texts = [[word for word in document.lower().split() if word not in stop_list] for document in documents]
    return remove_less_frequent_words(2, texts)


def remove_less_frequent_words(freq, texts):
    """Removes less and more frequent words from the text
    @param freq: Frequency to be removed
    @param texts: list of strings
    @return: list of tokens for each string
    """
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1
    return [[token for token in text if frequency[token] > freq] for text in texts]


def get_stopwords():
    """Get the stopwords
    @return: Set of stopwords
    """
    stopwords = set(STOPWORDS)
    more_stopwords = {'one', 'br', 'Po', 'th', 'sayi', 'fo', 'Unknown'}
    return stopwords.union(more_stopwords)


def find_most_similar_texts(target_sent, dataframe,
                            text_col_name, num_topics):
    """Given a target sentence and d dataframe
    along with the text col name, this method returns
    the most similar sentences from the dataframe
    @param target_sent: Target sentence in string
    @param dataframe: Dataframe
    @param text_col_name: Text column name
    @param num_topics: Number of topics to be used for the model
    @return: List of similar texts
    """
    corpus, texts, diction = convert_text_col_to_corpus(dataframe, text_col_name, 50)
    documents, text_col_name = build_clean_corpus(dataframe, text_col_name)
    lsi_model = models.LsiModel(corpus, id2word=diction, num_topics=num_topics)
    vec_bow = diction.doc2bow(target_sent.lower().split())
    vec_lsi = lsi_model[vec_bow]
    index = similarities.MatrixSimilarity(lsi_model[corpus])
    sims = sorted(enumerate(index[vec_lsi]), key=lambda item: -item[1])
    similar_texts = []
    for i, s in enumerate(sims):
        similar_texts.append((s[1], documents[i]))
    return similar_texts


def get_distance_between_sentences(dataframe, text_col_name,
                                   sent1, sent2):
    """Given two sentences and d dataframe and text
    col name as dataset, we return the WMD distance between
    those two sentences.
    @param dataframe: Dataframe
    @param text_col_name: Text column name
    @param sent1: First sentence
    @param sent2: Second sentence
    @return: Distance between the two sentences
    """
    text_processor = TextProcessor()
    documents, text_col_name = build_clean_corpus(dataframe, text_col_name)
    model = FT_gensim(size=100)
    model.build_vocab(sentences=documents)
    model.train(sentences=documents,
                epochs=model.epochs,
                total_examples=model.corpus_count,
                total_words=model.corpus_total_words)
    sent1 = text_processor.process(sent1)
    sent2 = text_processor.process(sent2)
    return model.wmdistance(sent1, sent2)
