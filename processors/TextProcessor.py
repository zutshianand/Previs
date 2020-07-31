import numpy as np
import pickle

import os
import operator
from gensim.models import KeyedVectors
import re
from collections import defaultdict
import json


def build_vocab(text: list) -> dict:
    """
    Creates a vocabulary of the text, which can be used to check text coverage.

    :param text: list of strings. Usually it will be one big string.
    :return: dictionary with words and their counts
    """
    vocab = defaultdict(lambda: 0)
    for txt in text:
        sentence = txt
        for word in str(sentence).split():
            vocab[word] += 1

    return vocab


def load_embed(filepath):
    """
    Load embeddings.

    :param filepath: path to the embeddings
    :return:
    """

    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    if '.pkl' in filepath:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    if 'news' in filepath:
        embeddings_index = dict(get_coefs(*o.strip().split(" ")) for o in open(filepath) if len(o) > 100)
    elif '.bin' in filepath:
        embeddings_index = KeyedVectors.load_word2vec_format(filepath, binary=True)
    else:
        embeddings_index = dict(get_coefs(*o.strip().split(" ")) for o in open(filepath, encoding='utf-8'))

    return embeddings_index


def check_coverage(vocab: dict, embeddings_index) -> list:
    """
    Check word coverage of embedding. Returns words which aren't in embeddings_index

    :param vocab: Dictionary with words and their counts.
    :param embeddings_index: embedding index
    :return: list of tuples with unknown words and their count
    """
    known_words = {}
    unknown_words = {}
    nb_known_words = 0
    nb_unknown_words = 0
    for word in vocab.keys():
        if word in embeddings_index:
            known_words[word] = embeddings_index[word]
            nb_known_words += vocab[word]
        elif word.capitalize() in embeddings_index:
            known_words[word] = embeddings_index[word.capitalize()]
            nb_known_words += vocab[word]
        elif word.lower() in embeddings_index:
            known_words[word] = embeddings_index[word.lower()]
            nb_known_words += vocab[word]
        elif word.upper() in embeddings_index:
            known_words[word] = embeddings_index[word.upper()]
            nb_known_words += vocab[word]
        else:
            unknown_words[word] = vocab[word]
            nb_unknown_words += vocab[word]

    vocab_rate = len(known_words) / len(vocab)
    print(f'Found embeddings for {vocab_rate:.2%} of vocab')

    text_rate = nb_known_words / (nb_known_words + nb_unknown_words)
    print(f'Found embeddings for {text_rate:.2%} of all text')
    unknown_words = sorted(unknown_words.items(), key=operator.itemgetter(1))[::-1]

    return unknown_words


def vocab_check_coverage(text: list, *args) -> list:
    """
    Calculate word coverage for the passed dataframe and embeddings.
    Can do it for one or several embeddings.

    :param text: list of strings. Usually it will only be one string
    :param args: one or several embeddings
    :return: list of dicts with out of vocab rate and words
    """
    oovs = []
    vocab = build_vocab(text)

    for emb in args:
        oov = check_coverage(vocab, emb)
        oov = {"oov_rate": len(oov) / len(vocab), 'oov_words': oov}
        oovs.append(oov)

    return oovs


def remove_space(text: str, spaces: list, only_clean: bool = True) -> str:
    """
    Remove extra spaces and ending space if any.

    :param text: text to clean
    :param text: spaces
    :param only_clean: simply clean texts or also replace texts
    :return: cleaned text
    """
    if not only_clean:
        for space in spaces:
            text = text.replace(space, ' ')

    text = text.strip()
    text = re.sub('\s+', ' ', text)

    return text


def replace_words(text: str, mapping: dict) -> str:
    """
    Replaces unusual punctuation with normal.

    :param text: text to clean
    :param mapping: dict with mapping
    :return: cleaned text
    """
    for word in mapping:
        if word in text:
            text = text.replace(word, mapping[word])

    return text


def clean_number(text: str) -> str:
    """
    Cleans numbers.

    :param text: text to clean
    :return: cleaned text
    """
    text = re.sub(r'(\d+)([a-zA-Z])', '\g<1> \g<2>', text)
    text = re.sub(r'(\d+) (th|st|nd|rd) ', '\g<1>\g<2> ', text)
    text = re.sub(r'(\d+),(\d+)', '\g<1>\g<2>', text)
    text = re.sub(r'(\d+),', '\g<1>', text)
    text = re.sub(r'(\d+)(e)(\d+)', '\g<1> \g<3>', text)

    return text


def spacing_punctuation(text: str, punctuation: str) -> str:
    """
    Add space before and after punctuation and symbols.

    :param text: text to clean
    :param punctuation: string with symbols
    :return: cleaned text
    """
    for punc in punctuation:
        if punc in text:
            text = text.replace(punc, f' {punc} ')

    return text


def fixing_with_regex(text) -> str:
    """
    Additional fixing of words.

    :param text: text to clean
    :return: cleaned text
    """
    mis_connect_list = ['\b(W|w)hat\b', '\b(W|w)hy\b', '(H|h)ow\b', '(W|w)hich\b', '(W|w)here\b', '(W|w)ill\b']
    mis_connect_re = re.compile('(%s)' % '|'.join(mis_connect_list))

    text = re.sub(r" (W|w)hat+(s)*[A|a]*(p)+ ", " WhatsApp ", text)
    text = re.sub(r" (W|w)hat\S ", " What ", text)
    text = re.sub(r" \S(W|w)hat ", " What ", text)
    text = re.sub(r" (W|w)hy\S ", " Why ", text)
    text = re.sub(r" \S(W|w)hy ", " Why ", text)
    text = re.sub(r" (H|h)ow\S ", " How ", text)
    text = re.sub(r" \S(H|h)ow ", " How ", text)
    text = re.sub(r" (W|w)hich\S ", " Which ", text)
    text = re.sub(r" \S(W|w)hich ", " Which ", text)
    text = re.sub(r" (W|w)here\S ", " Where ", text)
    text = re.sub(r" \S(W|w)here ", " Where ", text)
    text = mis_connect_re.sub(r" \1 ", text)
    text = text.replace("What sApp", ' WhatsApp ')

    # Clean repeated letters.
    text = re.sub(r"(I|i)(I|i)+ng", "ing", text)
    text = re.sub(r"(-+|\.+)", " ", text)

    text = re.sub(r'[\x00-\x1f\x7f-\x9f\xad]', '', text)
    text = re.sub(r'(\d+)(e)(\d+)', r'\g<1> \g<3>', text)  # is a dup from above cell...
    text = re.sub(r"(-+|\.+)\s?", "  ", text)
    text = re.sub("\s\s+", " ", text)
    text = re.sub(r'ᴵ+', '', text)

    text = re.sub(r"(W|w)on(\'|\’)t ", "will not ", text)
    text = re.sub(r"(C|c)an(\'|\’)t ", "can not ", text)
    text = re.sub(r"(Y|y)(\'|\’)all ", "you all ", text)
    text = re.sub(r"(Y|y)a(\'|\’)ll ", "you all ", text)

    text = re.sub(r"(I|i)(\'|\’)m ", "i am ", text)
    text = re.sub(r"(A|a)in(\'|\’)t ", "is not ", text)
    text = re.sub(r"n(\'|\’)t ", " not ", text)
    text = re.sub(r"(\'|\’)re ", " are ", text)
    text = re.sub(r"(\'|\’)s ", " is ", text)
    text = re.sub(r"(\'|\’)d ", " would ", text)
    text = re.sub(r"(\'|\’)ll ", " will ", text)
    text = re.sub(r"(\'|\’)t ", " not ", text)
    text = re.sub(r"(\'|\’)ve ", " have ", text)

    text = re.sub(
        r'(by|been|and|are|for|it|TV|already|justhow|some|had|is|will|would|should|shall|must|can|his|here|there|them|these|their|has|have|the|be|that|not|was|he|just|they|who)(how)',
        '\g<1> \g<2>', text)

    return text


def load_preprocessing_data() -> dict:
    """
    Loads dict with various mappings and strings for cleaning.

    :return:
    """
    if os.path.exists('../data/mapping_dict.json'):
        path = '../data/mapping_dict.json'
    else:
        path = '../data/mapping_dict.json'

    with open(path, 'r') as f:
        mapping_dict = json.load(f)

    # combine several dicts into one
    replace_dict = {**mapping_dict['contraction_mapping'],
                    **mapping_dict['mispell_dict'],
                    **mapping_dict['special_punc_mappings'],
                    **mapping_dict['rare_words_mapping'],
                    **mapping_dict['bad_case_words'],
                    **mapping_dict['mis_spell_mapping']}

    mapping_dict = {'spaces': mapping_dict['spaces'],
                    'punctuation': mapping_dict['punctuation'],
                    'words_to_replace': replace_dict}

    return mapping_dict


def build_matrix(word_index, path: str, embed_size: int):
    embedding_index = load_embed(path)
    embedding_matrix = np.zeros((len(word_index) + 1, embed_size))
    unknown_words = []

    for word, i in word_index.items():
        try:
            embedding_matrix[i] = embedding_index[word]
        except KeyError:
            unknown_words.append(word)

    return embedding_matrix, unknown_words


class TextProcessor(object):

    def __init__(self):
        self.mapping_dict = load_preprocessing_data()

    def process(self, text: str) -> str:
        """
        Apply all preprocessing.

        :param text: text to clean.
        :return: cleaned text
        """
        text = remove_space(text, self.mapping_dict['spaces'], only_clean=False)
        text = clean_number(text)
        text = spacing_punctuation(text, self.mapping_dict['punctuation'])
        text = fixing_with_regex(text)
        text = replace_words(text, self.mapping_dict['words_to_replace'])

        for punct in "/-'":
            if punct in text:
                text = text.replace(punct, ' ')

        text = clean_number(text)
        text = remove_space(text, self.mapping_dict['spaces'])

        return text

    def __call__(self, sample):
        processed_text = []
        for txt in sample:
            processed_text.append(self.process(txt))
        return processed_text
