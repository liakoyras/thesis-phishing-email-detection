"""
This module contains utilities for feature extraction and
feature selection.
"""
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

"""
Feature Extraction
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
    max_df : float or int, default=1.0
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
