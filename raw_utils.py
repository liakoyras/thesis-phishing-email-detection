"""
This module contains basic utility functions that help to process the
raw email data into more usable formats.
"""
import os
import random
import mailbox
import email as eml
import pandas as pd
import re

from bs4 import BeautifulSoup

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
    str or None
        The content type if text types are detected, None otherwise.
    """
    type_list = ['text/plain', 'text/html']
    content_type = message.get_content_type()
    if content_type in type_list:
        return content_type
    else:
        return None
    
def parse_html(input_string):
    """
    Parse an HTML string and extract the text.
    
    This is done with BeautifulSoup. A list of inline tags that could
    contain text is being merged with their parent tags, so that there
    won't be any needless newline delimiters by get_text(). After the
    tree is pruned, it is being parsed again in order for the text to
    be properly merged without actually belonging on a different node.
    
    Before this process, the hyperlink is extracted from all <a> tags
    so that there is more parity between the plaintext and HTML version
    of multipart emails.
    
    The returned string has a newline character as a delimiter between
    the text extracted from different (block) HTML elements.

    Parameters
    ----------
    input_string : str
        The string to be parsed.

    Returns
    -------
    str
        The converted string (or the input string, if none of the
        transformations were applicable).
    """
    soup = BeautifulSoup(input_string, 'lxml')
        
    inline_tag_names = ['a','abbr','acronym','b','bdo','button','cite','code',
                       'dfn','em','i','kbd','label','output','q','samp','small',
                       'span','strong','sub','sup','time','var']
    
    inline_tags = soup.find_all(inline_tag_names)
        
    if inline_tags:
        for tag in inline_tags:
            if tag.name=='a':
                url = tag.get('href')
                if url:
                    tag.append("<" + url + ">")
                    
            tag.unwrap()

        new_soup = BeautifulSoup(str(soup), 'lxml')        
        text = new_soup.get_text('\n', strip=True)
    else:
        text = soup.get_text('\n', strip=True)
    return text
    
def mbox_to_df(filename, filepath, text_only=True):
    """
    Convert the text from emails in a .mbox file to a Pandas DataFrame.
    
    It choses only text MIME types, specifically 'text/plain' and
    'text/html' and tries to parse any HTML with parse_html().
    
    Afterwards, it tries to do a very simple deduplication, to avoid
    getting the same text twice from multipart/alternative emails.
    This is achieved by standardizing whitespace with the use of
    regular expressions.
    
    During this process, it assumes however that the plaintext version
    will be the better choice (since we care about the text information
    only) and that the plaintext part is first (which is usually the
    case). This is not a big problem since it only affects the
    duplicate texts, so the version that will be kept in the end does
    not matter that much (provided the HTML parsing was decent enough).

    Each row of the output DataFrame contains a representation of an
    email, with the body (and other headers in the future) representing
    a column.

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
        
    See Also
    --------
    parse_html : Parse an HTML string and extract the text.
    check_text_types : Check if a message contains text data.
    """
    file = os.path.join(filepath,filename)
    mbox = mailbox.mbox(file)

    data = []
    skip_counter = 0
    for key in mbox.iterkeys(): # iterating through the mbox file
        try:
            message = mbox[key]
        except UnicodeDecodeError:
            skip_counter += 1
            continue

        row = {}

        if (not text_only):
            # TODO: Implement header (and other feature) extraction
            pass

        # Extracting body text
        content = []
        for part in message.walk(): # iterating through the message parts
            if part.is_multipart():
                continue
            
            ctype = check_text_types(part)
            if ctype:
                try:
                    new_content = part.get_payload(decode=True).decode()
                except UnicodeDecodeError:
                    new_content = part.get_payload(decode=True).decode('latin-1')
                
                if ctype == 'text/html':
                    content.append(parse_html(new_content))
                elif ctype == 'text/plain':
                    content.append(new_content)
        
        # rudimentary deduplication
        joined = '\n'.join(content)
        stripped = re.sub(r'\s+', '', joined)

        if stripped[:len(stripped)//2] == stripped[(len(stripped)//2):]:
            if content:
                row['body'] = content[0]
            else:
                row['body'] = content
        else:
            row['body'] = joined

        data.append(row)

    if (skip_counter > 0):
        print(skip_counter, "emails skipped: Headers contain non-ascii "\
                "characters, or otherwise corrupted email data.")

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
        Gets passed to mbox_to_df.

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


def sample_enron_to_mbox(path, amount, mode='number', overwrite=True):
    """
    Sample a number of emails from the Enron Dataset to create .mbox files

    The function runs through the folders in the Enron Dataset and
    randomly selects a number of emails (either absolute number or percentage)
    to add to a .mbox file for easier proccessing. The output file will be
    named with the number of emails contained.

    The dir specified in path must contain a folder named maildir as
    downloaded from the official enron dataset.

    It also prints a lot of information during this process.

    Parameters
    ----------
    path : str
        The path to the dataset, without '/maildir' at the end.
    amount : float
        The number of emails or the percentage of the dataset to sample.
    mode : {'number', 'percentage'}
        How will the amount parameter be interpreted.
    overwrite: bool, default True
        Wether or not to overwrite existing .mbox files.

    Returns
    -------
    str or None
        The name of the file created or None if the operation is not completed.

    Raises
    ------
    ValueError
        If the specified mode is not implemented.
    ValueError
        If moe is 'number' and the amount is negative.
    ValueError
        If mode is 'percentage' and amount is not between 0 and 1.
    """
    # Previous research has shown that these folders are mostly duplicates
    # and/or computer generated.
    ignore_folders = ['discussion_threads', '_sent_mail', 'all_documents']
    maildir_path = os.path.join(path, 'maildir')

    # Create a list with all the folders containing emails
    folders = [root for root, dirs, files in os.walk(maildir_path)
               if files and not any([ignored in root for ignored in ignore_folders])]
    print(len(folders), "folders will be checked.")

    # Create a list with all the email filenames
    email_list = [os.path.join(folder, file) for folder in folders
                  for file in os.scandir(folder) if not file.is_dir()]
    print(len(email_list), "emails found.")

    if (mode == 'percentage'):
        if (amount > 1 or amount <= 0):
            raise ValueError("The percentage must be between 0 and 1.")
        else:
            email_number = int(amount*len(email_list))
    elif (mode == 'number'):
        if (amount <= 0):
            raise ValueError("The number of emails must be positive.")
        else:
            email_number = int(amount)
    else:
        raise ValueError("This mode does not exist:", mode)

    print("Extracting", email_number, "random emails.")

    # Create output file
    mbox_folder = os.path.join(path, 'mbox')
    if not os.path.exists(mbox_folder):
        os.makedirs(mbox_folder)

    mbox_file = os.path.join(mbox_folder,'enron_'+str(email_number)+'.mbox')
    if os.path.exists(mbox_file):
        if(overwrite):
            print("File", mbox_file, "will be overwritten.")
            os.remove(mbox_file)
        else:
            print("File", mbox_file, "already exists but 'overwrite' "\
                "is set to False. The operation will stop.")
            return None
    else:
        print("Creating output file", mbox_file)

    mbox = mailbox.mbox(mbox_file)
    mbox.lock()

    # Writing emails
    random.shuffle(email_list)
    skip_counter = 0
    for email_file in email_list:
        if (email_number == 0):
            break
        else:
            try:
                email = eml.message_from_file(open(email_file))
                mbox.add(email)
                mbox.flush()
            except UnicodeDecodeError:
#                 print(email_file, ':')
                skip_counter += 1
                continue
            else:
                email_number -= 1

    if (skip_counter > 0):
        print(skip_counter, "emails skipped: Headers contain non-ascii "\
            "characters, or otherwise corrupted email data.")

    mbox.unlock()
    mbox.close()
    print(mbox_file, "was created successfully.")

    return os.path.basename(mbox_file)
