"""
This module contains basic utility functions that help to process the
raw email data into more usable formats.
"""
import os
import mailbox
import pandas as pd

def check_text_types(message):
    """
    Check if a message contains text data.

    This is achieved by looking at the Content-Type header and checking
    for the most common MIME types that are used for text.

    Parameters
    ----------
    message : email.message.EmailMessage
        The message to check.

    Returns
    -------
    bool
        True if text types are detected, False otherwise.
    """
    type_list = ['text/plain', 'text/html']
    content_type = message.get_content_type()
    if any(types in content_type for types in type_list):
        return True
    else:
        return False

    
def mbox_to_df(filename, filepath, text_only=True):
    """
    Convert the text from emails in a .mbox file to a Pandas DataFrame.

    Each row of the output DataFrame contains a representation of an
    email, with each header (and the body) in different columns.

    Parameters
    ----------
    filename : str
        The name of the file.
    filepath : str
        The location of the file.
    text_only : bool, default True
        True if only the text needs to be extracted, False otherwise.

    Returns
    -------
    pandas.DataFrame
        The resulting DataFrame.
    """
    file = os.path.join(filepath,filename)
    mbox = mailbox.mbox(file)
        
    data = []
    
    for key in mbox.iterkeys():
        try:
            message = mbox[key]
        except UnicodeDecodeError:
            print("One email skipped: A header contains non-ascii characters, or the email is in some other way corrupted.")
            continue
        
        row = {}    
        
        if (not text_only):
            # TODO: Implement header (and other feature) extraction
            pass
        
        # Extracting body text
        if message.is_multipart():
            content = ''
            for part in message.get_payload():
                if check_text_types(part):
                    content += (str(part.get_payload(decode=True)) + '\n')

            content = content[:-1] # Strip the final newline character
        else:
            content = message.get_payload(decode=True)

        row['Body'] = content

        data.append(row)
        
    dataframe = pd.DataFrame(data)
    
    return dataframe


def read_dataset(path, exceptions, text_only=True):
    """
    Read .mbox files inside a directory into a Pandas DataFrame.

    It uses mbox_to_df to convert every file to a DataFrame and then
    concatenates those DataFrames.

    Parameters
    ----------
    path : str
        The location of the files.
    exceptions : list of str
        The list with the filenames to be ignored.
    text_only : bool, default True
        Gets passed to mbox_to_dfφ.
        
    Returns
    -------
    pandas.DataFrame
        The resulting DataFrame.
    
    See Also
    --------
    mbox_to_df : Convert the text from emails in a .mbox file to a Pandas DataFrame.
        
    """
    mbox_files = os.listdir(path)
    mbox_files = [name for name in mbox_files if name not in exceptions]
    
    dataset = pd.DataFrame()
    for file in mbox_files:
        print("Now reading file:", file)
        file_data = mbox_to_df(file, path, text_only)
        dataset = pd.concat([dataset, file_data], ignore_index=True)
    
    return dataset


def save_to_csv(data, path, filename):
    """
    Save a DataFrame to a .csv file.

    The operation happens only if the file does not exist already. If it
    does, it will overwrite only if the user authorizes it.

    Parameters
    ----------
    data : pandas.DataFrame
        The DataFrame to be saved.
    path : str
        The location of the output files.
    filename : str
        The name of the .csv file.
    """
    attempted_filename = os.path.join(path, filename)
    if os.path.exists(attempted_filename):
        print("File", attempted_filename, "already exists.")
        overwrite = input("Do you want to overwrite it? (y/n) ")
        if (overwrite == 'Y' or overwrite == 'y'):
            print("File", attempted_filename, "will be overwritten.")
            data.to_csv(os.path.join(path, filename))
        else:
            print("Aborting, data will not be written.")
    else:
        print("Saving to", attempted_filename)
        data.to_csv(os.path.join(path, filename))
        
