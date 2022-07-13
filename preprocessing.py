"""
This module contains utility functions that process the data extracted
from the emails into a format more suitable for use with machine
learning algorithms.
"""
import pandas as pd
from bs4 import BeautifulSoup
import re

def strip_characters(input_string):
    """
    Strips characters from a string to convert it to plaintext.

    This is achieved by using regex to transform HTML line breaks
    and Unicode non-breaking space to newlines and spaces,
    stripping HTML tags with BeautifulSoup and removing extra
    and trailing whitespace with regex.
    
    Apart from the removal of HTML, the rest of the processing is
    done in order for the whitespace of an un-htmlified string to
    be as standardized as possible and close to that of the
    corresponding plaintext, in order for deduplicate_text() to
    work properly.

    Parameters
    ----------
    input_string : str
        The string to be converted.

    Returns
    -------
    str
        The converted string (or the input string, if none of the
        transformations are applicable).
        
    See Also
    --------
    deduplicate_text : Checks a string for duplicated text and returns it deduplicated.
    """
    converted_string = re.sub(r'<br>', ' ', input_string, flags=re.IGNORECASE)
    converted_string = re.sub(r'&nbsp;', ' ', converted_string, flags=re.IGNORECASE)
    
    soup = BeautifulSoup(converted_string, 'lxml')
    text = soup.get_text()
    
    output_text = re.sub(r'\s+', ' ', text)
    output_text = output_text.strip()
    
    return output_text


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
    
    It is not perfect, but it works in most cases.

    Parameters
    ----------
    input_text : str
        The text to be checked.

    Returns
    -------
    str
        The input_text or the deduplicated version if applicable.
    """
    s1 = input_text[:len(input_text)//2]
    s2 = input_text[(len(input_text)//2) + 1:]
    
    if s1 == s2:
        return s1
    else:
        return input_text
    
