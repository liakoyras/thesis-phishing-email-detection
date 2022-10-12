"""
This module contains utilities for feature extraction and
feature selection.

It can extract both vectorized text features and features
that are based on the writing style (stylometric).
"""
import pandas as pd
import numpy as np
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, chi2
from gensim.models import Word2Vec

from string import punctuation
from readability import Readability
import language_tool_python

from preprocessing import tokenize

"""
Text Feature Extraction
"""
def tfidf_features(text_col_train, text_col_test=None, max_df=1.0, min_df=1, max_features=None):
    """
    Extract TF-IDF features from a series using scikit-learn.

    Each row of the input series should contain a list with the
    words of the tokenized email. The output dictionary contains
    the trained vectorizer and the feature arrays of the train
    dataset and also of the test dataset, if provided.
    
    Some parameters for the vectorizer can also be passed as
    arguments.

    Parameters
    ----------
    token_text_col_train : pandas.Series of list of str
        The series that contains the tokenized emails as word lists.
    token_text_col_test : pandas.Series of list of str or None
        The series of the test set that contains the tokenized emails
        as word lists, or None if a test set is not provided.
    max_df : float or int, default 1.0
        Ignore terms that have a frequency higher than this threshold,
        as a percentage of documents if float in range [0.0, 1.0] or
        an absolute count for int. To be used by TfidfVectorizer.
    min_df : int, default 1
        The minimum number of times a term has to appear in order to
        be included in the vocabulary. To be used by TfidfVectorizer.
    max_features : int, default None
        The maximum number of terms in the vocabulary. To be used by
        TfidfVectorizer.
    
    Returns
    -------
    dict
    {'vectorizer': sklearn.feature_extraction.text.TfidfVectorizer,
     'tfidf_train': pandas.DataFrame
     'tfidf_test': pandas.DataFrame or None}
        A dictionary that contains the vectorizer and the vectorized
        sets.
    """
    # lazy check to see if the input column consists of lists of strings
    if all(isinstance(elem, str) for elem in text_col_train[0]):
        text_train = text_col_train.map(' '.join)
    else:
        raise TypeError("The input column must contain lists of strings.")
    
    output = dict(); 
    
    tfidf_vec = TfidfVectorizer(max_df=max_df, min_df=min_df, max_features=max_features)
    
    tfidf_score_train = tfidf_vec.fit_transform(text_train).toarray()
    tfidf_features_train = pd.DataFrame(tfidf_score_train, columns=tfidf_vec.get_feature_names_out())
    
    output['vectorizer'] = tfidf_vec
    output['tfidf_train'] = tfidf_features_train
    
    if text_col_test is not None:
        text_test = text_col_test.map(' '.join)
        
        tfidf_score_test = tfidf_vec.transform(text_test).toarray()
        tf_idf_features_test = pd.DataFrame(tfidf_score_test, columns=tfidf_vec.get_feature_names_out())
        
        output['tfidf_test'] = tf_idf_features_test
    else:
        output['tfidf_test'] = None
    
    return output


def filter_vocab_words(wordlist, vocabulary):
    """
    Remove words not appearing in a vocabulary from a list.
    
    Parameters
    ----------
    wordlist : list of str
        The list of words to be filtered.
    vocabulary : list of str
        The vocabulary that will do the filtering.
        
    Returns
    -------
    list of str
        The filtered list.
    """
    return [word for word in wordlist if word in vocabulary]

def get_mean_vector(wordlist, word2vec_model):
    """
    Calculate the mean vector of a list of words.
    
    It takes the word vectors from the Word2Vec model and
    calculates the mean of those who appear in this specific
    list of words.
    
    Parameters
    ----------
    wordlist : list of str
        The list of words to be vectorized.
    word2vec_model : gensim.models.word2vec.Word2Vec
        The Word2Vec model that produced the word vectors.
        
    Returns
    -------
    numpy.ndarray
        An array containing the mean vector, or zeroes if
        the input wordlist was empty.
    """
    if len(wordlist) >= 1:
        return np.mean(word2vec_model.wv[wordlist], axis=0)
    else:
        return np.zeros(word2vec_model.vector_size)

def word2vec_features(text_col_train, text_col_test=None, vector_size=100, min_count=5, max_vocab_size=None, workers=1):
    """
    Extract Word2Vec embedding features using gensim.
    
    It uses the skip-gram model that Word2Vec provides. This is
    hardcoded (sg=1).
    
    Word2Vec represents each word in the corpus as a high-dimensional
    vector. Then, get_mean_vector() is used to get the averages of
    the vectors of all the words in an email, after removing the
    words that do not appear in the vocabulary that Word2Vec built.
    
    Some parameters for the vectorizer can also be passed as
    arguments.

    Parameters
    ----------
    token_text_col_train : pandas.Series of list of str
        The series that contains the tokenized emails as word lists.
    token_text_col_test : pandas.Series of list of str or None
        The series of the test set that contains the tokenized emails
        as word lists, or None if a test set is not provided.
    vector_size : int, default 100
        The size (dimensions) of the word vectors that will be
        produced. To be used by Word2Vec.
    min_count : int, default 5
        The minimum number of times a term has to appear in order to
        be included in the vocabulary. To be used by Word2Vec.
    workers : int, default 1
        How many threads to do the processing on. Note that if this is
        not set to 1, the processing witll be faster but the result
        will not be 100% reproducible. To be used by Word2Vec.
    max_vocab_size : int, default None
        The maximum number of terms in the vocabulary. To be used by
        Word2Vec.
    
    Returns
    -------
    dict
    {'vectorizer': gensim.models.word2vec.Word2Vec,
     'word2vec_train': pandas.DataFrame
     'word2vec_test': pandas.DataFrame or None}
        A dictionary that contains the vectorizer and the vectorized
        sets.
    
    See Also
    --------
    filter_vocab_words : Remove words not appearing in a vocabulary from a list.
    get_mean_vector : Calculate the mean vector of a list of words.
    """
    output = dict();
    
    model = Word2Vec(sentences=text_col_train,
                     min_count=min_count, vector_size=vector_size, max_final_vocab=max_vocab_size,
                     sg=1, workers=workers, seed=1746)
    
    vocab = list(model.wv.key_to_index.keys())
    
    filtered_col_train = text_col_train.apply(filter_vocab_words, vocabulary=vocab)
    col_with_means_train = filtered_col_train.apply(get_mean_vector, word2vec_model=model)    
    word2vec_features_train = pd.DataFrame(col_with_means_train.tolist())
    
    output['vectorizer'] = model
    output['word2vec_train'] = word2vec_features_train
    
    if text_col_test is not None:
        filtered_col_test = text_col_test.apply(filter_vocab_words, vocabulary=vocab)
        col_with_means_test = filtered_col_test.apply(get_mean_vector, word2vec_model=model)    
        word2vec_features_test = pd.DataFrame(col_with_means_test.tolist())
        
        output['word2vec_test'] = word2vec_features_test
    else:
        output['word2vec_features_test'] = None

    return output


"""
Style Feature Extraction
"""
"""
Simple Counts
"""
def count_chars(input_string):
    """
    Count characters of a string.
    
    Parameters
    ----------
    input_string : str
        The string that will be counted.

    Returns
    -------
    int
        The number of characters in input_string.
    """
    return len(input_string)

def count_newlines(input_string):
    """
    Count newlines in a string.
    
    Parameters
    ----------
    input_string : str
        The string that will be counted.

    Returns
    -------
    int
        The number of newlines in input_string.
    """
    return input_string.count('\n')

def count_special_chars(input_string):
    """
    Count special characters in a string.
    
    The list of special characters that is being used is
    python's string.punctuation.
    
    Parameters
    ----------
    input_string : str
        The string that will be counted.

    Returns
    -------
    int
        The number of special characters in input_string.
    """
    special_chars = [char for char in punctuation]
    
    return np.sum(input_string.count(c) for c in special_chars)

def count_words(input_tokens):
    """
    Count words (tokens) in a list.

    The input list is the output of preprocessing.tokenize.
    
    Parameters
    ----------
    input_tokens : list of str
        The tokenized text that will be counted.

    Returns
    -------
    int
        The number of words in input_tokens.
    """
    return len(input_tokens)

def count_unique_words(input_tokens):
    """
    Count unique words (tokens) in a list.

    In essence it counts the unique tokens returned by
    preprocessing.tokenize, using set().
    
    Parameters
    ----------
    input_tokens : list of str
        The tokenized text that will be counted.

    Returns
    -------
    int
        The number of unique words in input_tokens.
    """
    return len(set(input_tokens))

def count_sentences(input_sentences):
    """
    Count sentences and how many start with uppercase/lowercase.
    
    The input list is the output of nltk.sent_tokenize.
    
    Parameters
    ----------
    input_string : str
        The string that will be counted.

    Returns
    -------
    (int, int, int)
        A tuple containing the number of sentences, sentences that
        start with an uppercase letter and sentences that start with
        a lowercase letter.
    """
    upper, lower = 0, 0
    for sentence in input_sentences:
        if sentence[0].isupper():
            upper +=1
        else:
            lower +=1
            
    return (len(input_sentences), upper, lower)


"""
Word Size
"""
def small_words(input_tokens):
    """
    Count small words (tokens) in a list.
    
    A small word is defined as one having 3 or fewer characters.
    
    The input list is the output of preprocessing.tokenize.
    
    Parameters
    ----------
    input_tokens : list of str
        The tokenized text that will be counted.

    Returns
    -------
    (int, float)
        A tuple containing the number of small words and their relative
        frequencies in input_tokens.
    """
    count = 0
    for token in input_tokens:
        if (len(token) <= 3):
            count += 1
    
    frequency = count/len(input_tokens)
    
    return (count, frequency)

def big_words(input_tokens):
    """
    Count big words (tokens) in a list.
    
    A big word is defined as one having more than 6 characters.

    The input list is the output of preprocessing.tokenize.
    
    Parameters
    ----------
    input_tokens : list of str
        The tokenized text that will be counted.

    Returns
    -------
    (int, float)
        A tuple containing the number of big words and their relative
        frequencies in input_tokens.
    """
    count = 0
    for token in input_tokens:
        if (len(token) > 6):
            count += 1
    
    frequency = count/len(input_tokens)
    
    return (count, frequency)

def huge_words(input_tokens):
    """
    Count huge words (tokens) in a list.
    
    A huge word is defined as one having more than 15 characters.

    The input list is the output of preprocessing.tokenize.
    
    Parameters
    ----------
    input_tokens : list of str
        The tokenized text that will be counted.

    Returns
    -------
    (int, float)
        A tuple containing the number of huge words and their relative
        frequencies in input_tokens.
    """
    count = 0
    for token in input_tokens:
        if (len(token) > 15):
            count += 1
    
    frequency = count/len(input_tokens)
    
    return (count, frequency)

def average_word_length(input_tokens):
    """
    Calculate the mean length of words (tokens) in a list.
    
    It uses np.mean() for fast calculation.
    The input list is the output of preprocessing.tokenize.
    
    Parameters
    ----------
    input_tokens : list of str
        The tokenized text that will be counted.

    Returns
    -------
    float
        The mean length of words in input_tokens.
    """
    return np.mean([len(word) for word in input_tokens])


"""
Sentence Size
"""
def average_sentence_length(sentences):
    """
    Calculate the mean length of sentences in a list.
    
    It uses np.mean() for fast calculation.
    The input list is the output of nltk.sent_tokenize.
    
    It uses both characters and words as a unit of length.
    
    Parameters
    ----------
    sentences : list of str
        The sentences that will be counted.

    Returns
    -------
    (float, float)
        A tuple containing the mean length of the input sentences
        in units of characters and words.
    """
    avg_chars = np.mean([len(sentence) for sentence in sentences])
    avg_words = np.mean([len(tokenize(sentence)) for sentence in sentences])
    
    return (avg_chars, avg_words)

def std_sentence_length(sentences):
    """
    Calculate the standard deviation of length of sentences in a list.
    
    It uses np.std() for fast calculation.
    The input list is the output of nltk.sent_tokenize.
    
    It uses both characters and words as a unit of length.
    
    Parameters
    ----------
    sentences : list of str
        The sentences that will be counted.

    Returns
    -------
    (float, float)
        A tuple containing the std of the length of the input
        sentences in units of characters and words.
    """
    std_chars = np.std([len(sentence) for sentence in sentences])
    std_words = np.std([len(tokenize(sentence)) for sentence in sentences])
    
    return (std_chars, std_words)

def min_sentence_length(sentences):
    """
    Calculate the minimum length of sentences in a list.
    
    It uses np.min() for fast calculation.
    The input list is the output of nltk.sent_tokenize.
    
    It uses both characters and words as a unit of length.
    
    Parameters
    ----------
    sentences : list of str
        The sentences that will be counted.

    Returns
    -------
    (float, float)
        A tuple containing the min length of the input sentences
        in units of characters and words.
    """
    min_chars = np.min([len(sentence) for sentence in sentences])
    min_words = np.min([len(tokenize(sentence)) for sentence in sentences])
    
    return (min_chars, min_words)

def max_sentence_length(sentences):
    """
    Calculate the maximum length of sentences in a list.
    
    It uses np.max() for fast calculation.
    The input list is the output of nltk.sent_tokenize.
    
    It uses both characters and words as a unit of length.
    
    Parameters
    ----------
    sentences : list of str
        The sentences that will be counted.

    Returns
    -------
    (float, float)
        A tuple containing the max length of the input sentences
        in units of characters and words.
    """
    max_chars = np.max([len(sentence) for sentence in sentences])
    max_words = np.max([len(tokenize(sentence)) for sentence in sentences])
    
    return (max_chars, max_words)


"""
Ratios
"""
def series_ratio(series_1, series_2):
    """
    Create a Series with the ratio of two others.
    
    Parameters
    ----------
    series_1 : pandas.Series
        The Series with the nominators.
    series_2 : pandas.Series
        The Series with the denominators.
        
    Returns
    -------
    pandas.Series of float
        A Series that contains the ratios of series_1/series_2.
    """
    return series_1/series_2

def character_to_chars(input_string, character):
    """
    Calculate the ratio of a character to all characters.
    
    Parameters
    ----------
    input_string : str
        The string that will be counted.
    character : str
        A single-characte string with the character to count.

    Returns
    -------
    float
        The ratio of the specified character in input_string.
    """
    return input_string.count(character) / len(input_string)

def chars_to_lines(input_string):
    """
    Calculate the ratio of characters to lines of a string.
    
    Parameters
    ----------
    input_string : str
        The string that will be counted.

    Returns
    -------
    float
        The ratio of characters to lines in input_string.
    """
    num_lines = len(input_string.splitlines())
    return len(input_string) / num_lines

def alpha_tokens_ratio(input_tokens):
    """
    Calculate the ratio of alphabetic tokens.
    
    Parameters
    ----------
    input_tokens : list of str
        The list of tokens to calculate the ratio of.
        
    Returns
    -------
    float
        The ratio of alphabetic tokens to all tokens.
    """
    alpha_tokens = [token for token in input_tokens if token.isalpha()]
    return len(alpha_tokens) / len(input_tokens)


"""
Readability and Spelling
"""
def readability(input_text):
    """
    Calculate several readability scores for a text.
    
    The scores are Flesch Kincaid Grade Level, Flesch Reading Ease,
    Dale Chall Readability, Automated Readability Index (ARI),
    Coleman Liau Index, Gunning Fog, Spache and Linsear Write, using
    the implementation of the readability (py-readability-metrics)
    module.
    
    Since this implementation needs at least 100 words to function,
    the messages with less than this amount (the thrown error will be
    caught by try/except) will have numpy.NaN as the value for the
    readability scores.
    
    Parameters
    ----------
    input_text : str
        The string that will be scored.

    Returns
    -------
    (float, float, float, float, float, float, float, float)
        A tuple containing the different readability scores in order.
    """
    r = Readability(input_text)
    
    try:
        f_k_score = r.flesch_kincaid().score
    except:
        f_k_score = np.NaN
    try:
        fle_score = r.flesch().score
    except:
        fle_score = np.NaN
    try:
        fog_score = r.gunning_fog().score
    except:
        fog_score = np.NaN
    try:
        col_score = r.coleman_liau().score
    except:
        col_score = np.NaN
    try:
        dal_score = r.dale_chall().score
    except:
        dal_score = np.NaN
    try:
        ari_score = r.ari().score
    except:
        ari_score = np.NaN
    try:
        l_w_score = r.linsear_write().score
    except:
        l_w_score = np.NaN
    # there are not enough messages with more than 100 sentences
    #try:
    #    smg_score = r.smog().score
    #except:
    #    smg_score = np.NaN
    try:
        spa_score = r.spache().score
    except:
        spa_score = np.NaN
    
    return (f_k_score, fle_score, fog_score, col_score, dal_score, ari_score, l_w_score, spa_score)


def errors_check(input_text, tool):
    """
    Count the number of spelling and grammatical errors.
    
    It uses the language_tool_python module as the back-end, which
    is a wrapper for Language Tool.
    
    At the beginning, a simple cleaning of <emailaddress> and
    <urladdress> tokens is performed using regular expressions.
    This is beneficial not only because they should not be
    considered to be spelling mistakes in the first place, but also
    for performance reasons.
    
    Taking the tool as a parameter instead of initializing it here
    saves an enormous amount of time during the execution, since
    the initialization process takes several seconds.
    
    In case an email contains so many errors that it makes the
    server unresponsive, the return value will be numpy.NaN and
    the program will continue.
    
    Parameters
    ----------
    input_text : str
        The string that will be checked for errors.
    tool : language_tool_python.LanguageTool
        The initialized LanguageTool instance.
        
    Returns
    -------
    int
        The number of errors found in the text.
    """
    clean_text = re.sub(r'<emailaddress>', 'email address', input_text)
    clean_text = re.sub(r'<urladdress>', 'web address', clean_text)
    
    try:
        matches = tool.check(clean_text)
        errors = len(matches)
    except:
        errors = np.NaN
        
    return errors



"""
Feature Selection
"""
def chi2_feature_selection(features_train, class_col_train, features_test=None, percentile=50):
    """
    Select top features of a set using chi2 of scikit-learn.

    The output dictionary contains the trained selector and the
    feature arrays of the train dataset and also of the test
    dataset, if provided.
    
    The percentage of features to be selected can be passed as an
    argument, that will be passed to sklearn.SelectPercentile.

    Parameters
    ----------
    features_train : pandas.DataFrame
        The DataFrame containing the feature array of the train set.
    class_col_train : pandas.Series
        The series containing the classes of the train set emails.
    features_test : pandas.DataFrame or None
        The DataFrame containing the feature array of the test set,
        or None if a test set is not provided.
    percentile : int, default 50
        An integer showing the percent of features to keep.
        To be used by SelectPercentile.

    Returns
    -------
    dict
    {'selector': sklearn.feature_selection._univariate_selection.SelectPercentile,
     'features_train': pandas.DataFrame
     'features_test': pandas.DataFrame or None}
        A dictionary that contains the selector model and the
        reduced sets.
    """
    output = dict(); 
    
    chi2_selector = SelectPercentile(chi2, percentile=percentile)
    
    chi2_features_train = chi2_selector.fit_transform(features_train, class_col_train)
    selected_features_train = pd.DataFrame(chi2_features_train, columns=chi2_selector.get_feature_names_out())
    
    output['selector'] = chi2_selector
    output['features_train'] = selected_features_train
    
    if features_test is not None:
        chi2_features_test = chi2_selector.transform(features_test)
        selected_features_test = pd.DataFrame(chi2_features_test, columns=chi2_selector.get_feature_names_out())
        
        output['features_test'] = selected_features_test
    else:
        output['features_test'] = None
        
    return output
