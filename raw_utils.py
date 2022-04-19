"""
This module contains basic utility functions that help to process the
raw email data into more usable formats.
"""
import os
import random
import mailbox
import email as eml
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
            print("Email skipped: A header contains non-ascii characters, "\
                "or the email is in some other way corrupted.")
            continue

        row = {}

        if (not text_only):
            # TODO: Implement header (and other feature) extraction
            pass

        # Extracting body text
        content = ''
        for part in message.walk():
            if part.is_multipart():
                continue

            if check_text_types(part):
                content += (part.get_payload(decode=True).decode('latin-1') + '\n')

        content = content[:-1] # strips the final newline character

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
                print("Email skipped: A header contains non-ascii characters, "\
                "or the email is in some other way corrupted.")
                continue
            else:
                email_number -= 1

    mbox.unlock()
    mbox.close()
    print(mbox_file, "was created successfully.")

    return os.path.basename(mbox_file)
