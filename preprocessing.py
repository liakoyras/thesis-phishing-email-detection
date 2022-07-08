"""
This module contains utility functions that process the data extracted
from the emails into a format more suitable for use with machine
learning algorithms.
"""
import pandas as pd

def strip_characters(input_series):
    """
    Strips characters from the text in a series to keep only plaintext.

    This is achieved by using regex to transform HTML characters to
    newlines and spaces, stripping HTML tags and extra whitespace.

    Parameters
    ----------
    input_series : pandas.Series
        The series to be converted.

    Returns
    -------
    pandas.Series
        The converted series.
    """
    output_series = input_series.str.replace(r'<br>', '\n', regex=True, case=False)
    output_series = output_series.str.replace(r'&nbsp;', ' ', regex=True, case=False)
    output_series = output_series.str.replace(r'<[^<>]*>', '', regex=True)
    output_series = output_series.str.strip().replace(r'\s+', ' ', regex=True)
    
    return output_series


def deduplicate_text(input_text):
    """
    Checks a string for duplicated text and returns it deduplicated.

    This is simply checked with a direct string comparison of the
    two halves of the string.
    
    It is needed because multipart/alternative emails usually
    contain the same text in both HTML and plaintext version, and
    the processing of the raw data leads to concatenation of these
    two strings that essentially convey the same information.
    
    Due to previous processing with strip_characters(), any html
    text will be very closely resembling plaintext and can thus
    be directly compared to the plaintext part.
    
    It is not 100% foolproof, but it works in most cases.

    Parameters
    ----------
    input_series : str
        The text to be checked.

    Returns
    -------
    str
        The the input or the deduplicated version if applicable.
    """
    s1 = input_text[:len(input_text)//2]
    s2 = input_text[(len(input_text)//2) + 1:]
    
    if s1 == s2:
        return s1
    else:
        return input_text
    
