"""
This module contains functions for a basic machine-learning
binary classification pipeline, including training models,
making predictions on test data and evaluating those
predictions.
Based on scikit-learn.
"""
import pandas as pd

from preprocessing import separate_features_target

from sklearn.preprocessing import StandardScaler, MinMaxScaler

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

def train_logistic_regression(features, target, max_iter=1000, penalty='l2', C=1e10, standardization=True, show_train_accuracy=False):
    """
    Train a Logistic Regression classifier.
    
    It is a simple wrapper that creates an
    sklearn.LogisticRegression model using the input
    parameters and then uses fit_model() to train it.
    
    It also supports standardization, in order to be
    able to match Spark behavior.
    
    Parameters
    ----------
    features : pandas.DataFrame
        The DataFrame containing the features that the
        classifier will be fitted on.
    target : pandas.Series
        The Series with the target class variable.
    max_iter : int, default 1000
        Maximum number of iterations to converge.
        To be used by LogisticRegression.
    penalty : str, default 'l2'
        The norm of the penalty. To be used by
        LogisticRegression.
    C : float, default 1e10
        The Inverse of regularization strength.
        To be used by LogisticRegression.
    standardization : bool, default True
        Wether or not to apply standardization.
    show_train_accuracy : bool, default False
        If True, it prints the accuracy of the model
        on the training data. To be used by fit_model().
        
    Returns
    -------
    dict
        {'model': sklearn.linear_model._logistic.LogisticRegression, 
        'scaler': sklearn.preprocessing._data.StandardScaler or None}
        A dictionary containing the fitted LogisticRegression
        classifier and the scaler used for the standardization
        (if this option was selected).
        
    See Also
    --------
    fit_model : Fit a classifier.
    """
    lr = LogisticRegression(max_iter=max_iter, penalty=penalty, C=C, random_state=alg_random_state)
    
    if standardization:
        scaler = StandardScaler().fit(features)
        features = pd.DataFrame(scaler.transform(features), columns=features.columns)
    else:
        scaler = None
        
    models = dict()
    models['scaler'] = scaler
    
    fitted_lr = fit_model(lr, features, target, show_train_accuracy)
    models['model'] = fitted_lr
    
    return {'model': fitted_lr, 
            'scaler': scaler}

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

def train_naive_bayes(features, target, alpha=1.0, remove_negatives=False, show_train_accuracy=False):
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
    remove_negatives : bool, default False
        Scales the data to remove negative values (when
        with word2vec features for example).
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
    
    if remove_negatives:
        scaler = MinMaxScaler().fit(features)
        features = pd.DataFrame(scaler.transform(features), columns=features.columns)
    
    fitted_nb = fit_model(nb, features, target, show_train_accuracy)
    
    return fitted_nb


"""
Results Evaluation
"""
def confusion_matrix_values(true, predicted):
    """
    Calculate the confusion matrix.
    
    Parameters
    ----------
    true : pandas.Series
        The Series with the correct class labels.
    predicted : pandas.Series
        The Series with the predicted class labels.
        
    Returns
    -------
    tuple of float
        A tuple containing the values of the confusion
        matrix in order.
    """
    cm = confusion_matrix(true, predicted)
              
    tn = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tp = cm[1][1]
    
    return (tn, fp, fn, tp)

def metrics(true, predicted):
    """
    Calculate evaluation metrics for a set of predictions.
    
    The used metrics are Accuracy, Precision, Recall,
    F1 Score, False Positive and False Negative
    Rates, and Area Under ROC Curve.
    
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
    confusion_matrix_values : Calculate the confusion matrix.
    """
    (tn, fp, fn, tp) = confusion_matrix_values(true, predicted)
    
    acc = accuracy_score(true, predicted)
    pre = precision_score(true, predicted)
    rec = recall_score(true, predicted)
    f1  = f1_score(true, predicted)
    fpr = fp / (fp + tn)
    fnr = fn / (tp + fn)
    auc = roc_auc_score(true, predicted)
    
    return pd.DataFrame({'Accuracy': [acc],
                          'Precision': [pre],
                          'Recall': [rec],
                          'F1 Score': [f1],
                          'False Positive Rate': [fpr],
                          'False Negative Rate': [fnr],
                          'Area Under ROC Curve': [auc]})

def results(model, test_features, test_target, scaler=None):
    """
    Evaluate predictions of a model with a test set.
    
    It makes predictions for the test set and returns those
    along with some evaluation metrics by using metrics().
    
    It can accept a scaler as a parameter that will be used
    to scale the testing data.
    
    Parameters
    ----------
    model : sklearn classifier object
        The fitted model to be tested.
    test_features : pandas.DataFrame
        The features of the test set.
    test_target : pandas.Series
        The Series with the true class labels of the test set.
    scaler : sklearn scaler object or None, default None
        The scaler that can be used to standardize the testing
        data.
        
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
    if scaler:
        test_features = pd.DataFrame(scaler.transform(test_features), columns=test_features.columns)
        
    predictions = model.predict(test_features)
                  
    results = metrics(test_target, predictions)

    return {'results': results,
            'predictions': predictions}

def multi_model_results(models, names, test_features, test_target, lr_scaler=None):
    """
    Evaluate predictions of many models with a test set.
    
    It makes predictions for the test set and returns those
    along with some evaluation metrics by using metrics().
    
    If a scaler was used when training a Logistic Regression
    model, it can be passed as an argument and it will be
    used to standardize the data before making predictions.
    
    Parameters
    ----------
    models : list of sklearn classifier object
        The fitted models to be tested.
    names : list of str
        The names of the models, in the same order as the
        models parameter.
    test_features : pandas.DataFrame
        The features of the test set.
    test_target : pandas.Series
        The Series with the true class labels of the test set.
    lr_scaler : sklearn scaler object or None, default None
        If a scaler is provided and there is a Logistic
        Regression model in the model list, it will be used
        to standardize the test data.
        
    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the result metrics for all of the
        input models.
        
    See Also
    --------
    results : Evaluate predictions of a model with a test set.
    """
    final_df = pd.DataFrame()
    
    for model, name in zip(models, names):
        if 'LogisticRegression' in str(type(model)) and lr_scaler:
            result = results(model, test_features, test_target, lr_scaler)
        else:
            result = results(model, test_features, test_target)
        
        named_df = result['results'].rename(index={0: name})
        final_df = pd.concat([final_df, named_df])

    return final_df


def results_by_id(models, names, test_set, id_list, lr_scaler=None):
    """
    Present predictions of many models for specific rows.
    
    It works similarly to multi_model_results() but it reduces
    the test_set in the beginning using the provided list of
    ids and returns the raw predictions instead of the
    evaluation metrics.
    
    The id column is hardcoded to be 'email_id', since
    this function exists only for presentation purposes.
    
    Parameters
    ----------
    models : list of sklearn classifier object
        The fitted models to be tested.
    names : list of str
        The names of the models, in the same order as the
        models parameter.
    test_set : pandas.DataFrame
        The complete test dataset (that contains feature, class and
        id columns).
    id_list : list of int
        The list of ids to keep from the initial test dataset.
    lr_scaler : sklearn scaler object or None, default None
        If a scaler is provided and there is a Logistic
        Regression model in the model list, it will be used
        to standardize the test data.
        
    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the predictions about specific emails
        for all of the input models.
        
    See Also
    --------
    multi_model_results : Evaluate predictions of many models with a test set.
    """
    test_set = test_set[test_set['email_id'].isin(id_list)].reset_index(drop=True)

    test_features_target = separate_features_target(test_set)
    test_features = test_features_target['features']
    test_target = test_features_target['target']

    final_df = pd.DataFrame()

    final_df['Email ID'] = test_set['email_id']
    final_df['True Class'] = test_target.reset_index(drop=True)

    for model, name in zip(models, names):
        if 'LogisticRegression' in str(type(model)) and lr_scaler:
            test_features = pd.DataFrame(lr_scaler.transform(test_features), columns=test_features.columns)
            predictions = model.predict(test_features)
        else:
            predictions = model.predict(test_features)

        final_df[name] = predictions

    return final_df
