"""
This module contains functions for a basic machine-learning
binary classification pipeline, including training models,
making predictions on test data and evaluating those
predictions.
Based on scikit-learn.
"""
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

alg_random_state = 1746

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

def train_logistic_regression(features, target, max_iter=500, penalty='l2', C=1e10, show_train_accuracy=False):
    """
    Train a Logistic Regression classifier.
    
    It is a simple wrapper that creates an
    sklearn.LogisticRegression model using the input
    parameters and then uses fit_model() to train it.
    
    Parameters
    ----------
    features : pandas.DataFrame
        The DataFrame containing the features that the
        classifier will be fitted on.
    target : pandas.Series
        The Series with the target class variable.
    max_iter : int, default 500
        Maximum number of iterations to converge.
        To be used by LogisticRegression.
    penalty : str, default 'l2'
        The norm of the penalty. To be used by
        LogisticRegression.
    C : float, default 1e10
        The Inverse of regularization strength.
        To be used by LogisticRegression.
    show_train_accuracy : bool, default False
        If True, it prints the accuracy of the model
        on the training data. To be used by fit_model().
        
    Returns
    -------
    sklearn.linear_model._logistic.LogisticRegression
        The fitted LogisticRegression classifier.
        
    See Also
    --------
    fit_model : Fit a classifier.
    """
    lr = LogisticRegression(max_iter=max_iter, penalty=penalty, C=C, random_state=alg_random_state)
    fitted_lr = fit_model(lr, features, target, show_train_accuracy)
    
    return fitted_lr

def train_decision_tree(features, target, max_depth=5, show_train_accuracy=False):
    """
    Train a Decision Tree classifier.
    
    It is a simple wrapper that creates an
    sklearn.DecisionTreeClassifier model using the input
    parameters and then uses fit_model() to train it.
    
    Parameters
    ----------
    features : pandas.DataFrame
        The DataFrame containing the features that the
        classifier will be fitted on.
    target : pandas.Series
        The Series with the target class variable.
    max_depth : int, default 5
        Maximum depth of the tree.
        To be used by DecisionTreeClassifier.
    show_train_accuracy : bool, default False
        If True, it prints the accuracy of the model
        on the training data. To be used by fit_model().
        
    Returns
    -------
    sklearn.tree._classes.DecisionTreeClassifier
        The fitted DecisionTreeClassifier classifier.
        
    See Also
    --------
    fit_model : Fit a classifier.
    """
    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=alg_random_state)
    fitted_dt = fit_model(dt, features, target, show_train_accuracy)
    
    return fitted_dt

def train_random_forest(features, target, max_depth=5, n_estimators=20, show_train_accuracy=False):
    """
    Train a Random Forest classifier.
    
    It is a simple wrapper that creates an
    sklearn.RandomForestClassifier model using the input
    parameters and then uses fit_model() to train it.
    
    Parameters
    ----------
    features : pandas.DataFrame
        The DataFrame containing the features that the
        classifier will be fitted on.
    target : pandas.Series
        The Series with the target class variable.
    max_depth : int, default 5
        Maximum depth of the tree.
        To be used by RandomForestClassifier.
    n_estimators : int, default 20
        The number of trees that will be used.
        To be used by RandomForestClassifier.
    show_train_accuracy : bool, default False
        If True, it prints the accuracy of the model
        on the training data. To be used by fit_model().
        
    Returns
    -------
    sklearn.ensemble._forest.RandomForestClassifier
        The fitted RandomForestClassifier classifier.
        
    See Also
    --------
    fit_model : Fit a classifier.
    """
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=alg_random_state)
    fitted_rf = fit_model(rf, features, target, show_train_accuracy)
    
    return fitted_rf

def train_gradient_boost(features, target, loss='log_loss', max_depth=3, learning_rate=0.1, show_train_accuracy=False):
    """
    Train a Gradient Boosting classifier.
    
    It is a simple wrapper that creates an
    sklearn.GradientBoostingClassifier model using the input
    parameters and then uses fit_model() to train it.
    
    Parameters
    ----------
    features : pandas.DataFrame
        The DataFrame containing the features that the
        classifier will be fitted on.
    target : pandas.Series
        The Series with the target class variable.
    loss : str, default 'log_loss'
        The loss function to be optimized. To be used
        by GradientBoostingClassifier.
    max_depth : int, default 3
        Maximum depth of the individual estimators. To be
        used by GradientBoostingClassifier.
    learning_rate : float, default 0.1
        Reduces the contribution of each tree by this amount.
        To be used by GradientBoostingClassifier.
    show_train_accuracy : bool, default False
        If True, it prints the accuracy of the model
        on the training data. To be used by fit_model().
        
    Returns
    -------
    sklearn.ensemble._gb.GradientBoostingClassifier
        The fitted GradientBoostingClassifier classifier.
        
    See Also
    --------
    fit_model : Fit a classifier.
    """
    gb = GradientBoostingClassifier(loss=loss, max_depth=max_depth, learning_rate=learning_rate, random_state=alg_random_state)
    fitted_gb = fit_model(gb, features, target, show_train_accuracy)
    
    return fitted_gb

def train_naive_bayes(features, target, alpha=1.0, show_train_accuracy=False):
    """
    Train a Multinomial Naive Bayes classifier.
    
    It is a simple wrapper that creates an
    sklearn.MultinomialNB model using the input
    parameters and then uses fit_model() to train it.
    
    Parameters
    ----------
    features : pandas.DataFrame
        The DataFrame containing the features that the
        classifier will be fitted on.
    target : pandas.Series
        The Series with the target class variable.
    alpha : float, default 1.0
        The parameter for additive smoothing. To be used
        by MultinomialNB.
    show_train_accuracy : bool, default False
        If True, it prints the accuracy of the model
        on the training data. To be used by fit_model().
        
    Returns
    -------
    sklearn.naive_bayes.MultinomialNB
        The fitted MultinomialNB classifier.
        
    See Also
    --------
    fit_model : Fit a classifier.
    """
    nb = MultinomialNB(alpha=alpha)
    fitted_nb = fit_model(nb, features, target, show_train_accuracy)
    
    return fitted_nb


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
