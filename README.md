# phishing-email-detection

## Data
In order to download the phishing data:
```
cd ~/projects/phishing/data/phishing/nazario/
wget -r --no-parent --level=1 --reject "index.html*" --wait=1 https://monkey.org/~jose/phishing/
cp ./monkey.org/~jose/phishing/* ./
```
Make sure the `--level=1` flag is used, otherwise `wget` will follow every phishing link found in the
emails.

For the legitimate Enron data:
```
cd ~/projects/phishing/data/legitimate/enron
wget https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz
tar -x -f enron_mail_20150507.tar.gz
```

In case this directory structure is not followed, make sure the path/filename variables in the code
are changed as well.

## Requirements
```
pip install -r requirements.txt
```

## Project Structure
The code is structured as a series of Jupyter Notebooks.

Each notebook contains a part of the data processing and detection process, and outputs the results
in .csv files, in order to avoid running the entire code (a process that can be lengthy) every time
a change is being done in a specific part of the process.

The notebooks should be ran in the following order:
- `Import Text Data.ipynb` It reads the raw data files and converts them to `pandas.DataFrame`.
- `Text Dataset Cleanup.ipynb` Some rudimentary processing (like removing empty rows) and creation
of two datasets with 1:1 and 1:10 phishing to legitimate ratios.
- `Text Data Preprocessing.ipynb` Conversion of the email strings to lemmatized lists of words.
- `Text Feature Extraction and Feature Selection.ipynb` Vectorization of the text features.
- `Text Data Classification.ipynb` Algorithm training, predictions and evaluation metrics.

The python files (`raw_utils.py`, `preprocessing.py`, `features.py` and `machine_learning.py`) contain utility functions that are being used inside
the notebooks.
