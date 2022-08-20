"""
This module contains functions for a basic machine-learning
binary classification pipeline, including training models,
making predictions on test data and evaluating those
predictions.
Based on scikit-learn.
"""
import pandas as pd

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

"""
Preprocessing
"""
def separate_features_target(dataframe, num_cols_ignore=2, class_col_name='email_class'):
    """
    Separate feature columns from the target column.
    
    It assumes that any non-feature columns are in the
    beginning of the dataset (right after the index).
    
    Parameters
    ----------
    dataframe : pandas.DataFrame
        The DataFrame with data to split.
    num_cols_ignore : int
        The number of non-feature columns to skip.
    class_col_name : str
        The name of the target column.
        
    Returns
    -------
    dict
    {'features': pandas.DataFrame,
     'target': pandas.Series}
        A dictionary containing the features and target.
    """
    return {'features': dataframe[dataframe.columns[num_cols_ignore:]],
            'target': dataframe[class_col_name]}


"""
Training
"""
def fit_model(model, features, target, show_train_accuracy=False):
    """
    Fit a classifier.
    
    The input model should be a scikit-learn classifier
    supporting the .fit() method.
    
    Parameters
    ----------
    model : sklearn classifier object
        The classifier to use.
    features : pandas.DataFrame
        The DataFrame containing the features that the
        classifier will be fitted on.
    target : pandas.Series
        The Series with the target class variable.
    show_train_accuracy : bool, default False
        If True, it prints the accuracy of the model
        on the training data.
        
    Returns
    -------
    sklearn classifier object
        The fitted classifier model.
    """
    fitted_model = model.fit(features, target)
    
    if show_train_accuracy:
        predictions = fitted_model.predict(features)
        print("Train accuracy:", accuracy_score(target, predictions))
    
    return fitted_model


"""
Results Evaluation
"""
def confusion_matrix_rates(true, predicted):
    """
    Calculate confusion matrix rates.
    
    Parameters
    ----------
    true : pandas.Series
        The Series with the correct class labels.
    predicted : pandas.Series
        The Series with the predicted class labels.
        
    Returns
    -------
    tuple of float
        A tuple containing the rates.
    """
    samples = true.shape[0]
    cm = confusion_matrix(true, predicted)
              
    tnr = cm[0][0]/samples
    fpr = cm[0][1]/samples
    fnr = cm[1][0]/samples
    tpr = cm[1][1]/samples
    
    return (tnr, fpr, fnr, tpr)

def metrics(true, predicted):
    """
    Calculate evaluation metrics for a set of predictions.
    
    The used metrics are Accuracy, Precision, Recall,
    F1 Score, False Positive and False Negative
    Rates, and Area Under ROC Curve.
    
    For FPR and FNR, confusion_matrix_rates() is used.
    
    Parameters
    ----------
    true : pandas.Series
        The Series with the correct class labels.
    predicted : pandas.Series
        The Series with the predicted class labels.
    
    Returns
    -------
    pandas.DataFrame
        A single-row DataFrame containing the metrics
        for this set of predictions.
        
    See Also
    --------
    confusion_matrix_rates : Calculate confusion matrix rates.
    """
    cm_rates = confusion_matrix_rates(true, predicted)
    
    acc = accuracy_score(true, predicted)
    pre = precision_score(true, predicted)
    rec = recall_score(true, predicted)
    f1  = f1_score(true, predicted)
    fpr = cm_rates[1]
    fnr = cm_rates[2]
    auc = roc_auc_score(true, predicted)
    
    return pd.DataFrame({'Accuracy': [acc],
                          'Precision': [pre],
                          'Recall': [rec],
                          'F1 Score': [f1],
                          'False Positive Rate': [fpr],
                          'False Negative Rate': [fnr],
                          'Area Under ROC Curve': [auc]})

def results(model, test_features, test_target):
    """
    Evaluate predictions of a model with a test set.
    
    It makes predictions for the test set and returns those
    along with some evaluation metrics by using metrics().
    
    Parameters
    ----------
    model : sklearn classifier object
        The fitted model to be tested.
    test_features : pandas.DataFrame
        The features of the test set.
    test_target : pandas.Series
        The Series with the true class labels of the test set.
    
    Returns
    -------
    dict
    {'results': pandas.DataFrame,
     'predictions': pandas.Series}
        A dictionary containing the test result metrics and
        the predictions themselves.
        
    See Also
    --------
    metrics : Calculate evaluation metrics for a set of predictions.
    """
    predictions = model.predict(test_features)
                  
    results = metrics(test_target, predictions)

    return {'results': results,
            'predictions': predictions}
