"""
This module contains utilities for feature extraction and
feature selection.

It can extract both vectorized text features and features
that are based on the writing style (stylometric).
"""
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, chi2

from gensim.models import Word2Vec

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
                     workers=workers, seed=1746)
    
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
Word Size Features
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
