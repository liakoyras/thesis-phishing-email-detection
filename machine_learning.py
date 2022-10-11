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

from sklearn.model_selection import cross_val_predict

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
    
    fitted_lr = fit_model(lr, features, target, show_train_accuracy)
    
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
    
    It can also standardize the values to [0,1] in order to
    remove any negative values (that NB cannot work with).
    
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
        training with word2vec features for example).
    show_train_accuracy : bool, default False
        If True, it prints the accuracy of the model
        on the training data. To be used by fit_model().
        
    Returns
    -------
    dict
    {'model': sklearn.naive_bayes.MultinomialNB,
     'scaler': sklearn.preprocessing._data.MinMaxScaler or None}
        A dictionary containing the fitted MultinomialNB
        classifier and the scaler used for the standardization (if
        (if this option was selected).
        
    See Also
    --------
    fit_model : Fit a classifier.
    """
    nb = MultinomialNB(alpha=alpha)
    
    if remove_negatives:
        scaler = MinMaxScaler().fit(features)
        features = pd.DataFrame(scaler.transform(features), columns=features.columns)
    else:
        scaler = None
    
    fitted_nb = fit_model(nb, features, target, show_train_accuracy)
    
    return {'model': fitted_nb, 
            'scaler': scaler}


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
    predicted : pandas.Series or numpy.ndarray
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

def metrics(true, predicted, probabilities):
    """
    Calculate evaluation metrics for a set of predictions.
    
    The used metrics are Accuracy, Precision, Recall,
    F1 Score, False Positive and False Negative
    Rates, and Area Under ROC Curve.
    
    Parameters
    ----------
    true : pandas.Series
        The Series with the correct class labels.
    predicted : pandas.Series or numpy.ndarray
        The predicted class labels.
    probabilities : pandas.Series or numpy.ndarray
        The predicted probabilities.
    
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
    auc = roc_auc_score(true, probabilities)
    
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
     'predictions': numpy.ndarray}
        A dictionary containing the test result metrics and
        the predictions themselves.
        
    See Also
    --------
    metrics : Calculate evaluation metrics for a set of predictions.
    """
    if scaler:
        test_features = pd.DataFrame(scaler.transform(test_features), columns=test_features.columns)
        
    predictions = model.predict(test_features)
    probabilities = model.predict_proba(test_features)[:, 1]
                  
    results = metrics(test_target, predictions, probabilities)

    return {'results': results,
            'predictions': predictions}

def multi_model_results(models, names, test_features, test_target, lr_scaler=None, nb_scaler=None):
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
    nb_scaler : sklearn scaler object or None, default None
        If a scaler is provided and there is a Multinomial NB model
        in the model list, it will be used to standardize the test data
        (since Naive Bayes needs to be trained without negative values
        in the feature set).
        
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
        elif 'MultinomialNB' in str(type(model)) and nb_scaler:
            result = results(model, test_features, test_target, nb_scaler)
        else:
            result = results(model, test_features, test_target)
        
        named_df = result['results'].rename(index={0: name})
        final_df = pd.concat([final_df, named_df])

    return final_df


def results_by_id(models, names, test_set, id_list, lr_scaler=None, nb_scaler=None):
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
    nb_scaler : sklearn scaler object or None, default None
        If a scaler is provided and there is a Multinomial NB model
        in the model list, it will be used to standardize the test data
        (since Naive Bayes needs to be trained without negative values
        in the feature set).
        
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
            predictions = model.predict(pd.DataFrame(lr_scaler.transform(test_features), columns=test_features.columns))
        elif 'MultinomialNB' in str(type(model)) and nb_scaler:
            predictions = model.predict(pd.DataFrame(nb_scaler.transform(test_features), columns=test_features.columns))
        else:
            predictions = model.predict(test_features)

        final_df[name] = predictions

    return final_df


"""
Stacking
"""
def train_models(feature_sets, target):
    """
    Train multiple classifiers on different feature sets.
    
    The models that will be used are hardcoded.
    
    Parameters
    ----------
    feature_sets : list of dict
        {'name': str,
        'features': pandas.DataFrame}
        A list of dictionaries containing a name for the feature
        set and the DataFrame containing the features that the
        classifiers will be trained on.
    target : pandas.Series
        The Series with the target class variable.
        
    Returns
    -------
    list of dict
    {'name': str,
     'features': str,
     'model': sklearn classifier object or dict}
        A list of dictionaries containing the name of the model,
        the name of the feature set used to train it and the
        model itself (or a dictionary with the model and the scaler
        if one was used).
    """
    output_models = []
    for feature_set in feature_sets:
        set_name = feature_set['name']
        output_models.append({'name': 'lr', 'features': set_name, 'model': train_logistic_regression(feature_set['features'], target)})
        output_models.append({'name': 'dt', 'features': set_name, 'model': train_decision_tree(feature_set['features'], target)})
        output_models.append({'name': 'rf', 'features': set_name, 'model': train_random_forest(feature_set['features'], target)})
        output_models.append({'name': 'gb', 'features': set_name, 'model': train_gradient_boost(feature_set['features'], target)})
        output_models.append({'name': 'nb', 'features': set_name, 'model': train_naive_bayes(feature_set['features'], target, remove_negatives=True)})
        
    return output_models

def train_stacked_models(initial_models, train_feature_sets, train_target, final_classifier=None, exclude_models=[], append_features=False):
    """
    Train a Stacking classifier.
    
    It uses the same method as sklearn.StackingClassifier:
    the input level 0 classifiers (trained on the entirety
    of the training dataset) are used to make robust cross
    validated predictions on the training dataset and then
    those predictions are used to train another classifier
    (by default Logistical Regression).
    
    When LR is used for the final classifier, the data will also
    be scaled using sklearn.StandardScaler.
    
    When "predictions" are referenced, it means the probability
    of the phishing class that sklearn's predict_proba has
    returned.
    
    Parameters
    ----------
    initial_models : list of dict
        {'name': str,
         'features': str,
         'model': sklearn classifier object or dict}
        A list created by train_models containing the fitted
        classifier models that will be stacked.
    train_feature_sets : list of dict
        {'name': str,
        'features': pandas.DataFrame}
        A list of dictionaries containing a name for the feature
        set and the DataFrame containing the features that the
        classifiers will be trained on.
        
        Note that these dictionaries have to match the ones given
        as input to the instance of train_models used to train the
        initial_models.
    train_target : pandas.Series
        The Series with the target class variable.
    final_classifier : sklearn classifier model or None, default None
        A classifier to be used as the level 1 model. If None
        is given, the default is Logistic Regression.
    exclude_models : list of {'lr', 'dt', 'rf', 'gb', 'nb'}
        If a list of the above strings is passed, any model found
        with those names will be excluded from the stacking (useful
        in order to be able to pass the output of train_models
        directly in initial_models without pruning it).
    append_features : bool, default False
        If True, the training data for the level 0 classifiers will
        also be used for the training of the level 1 classifier.
        
    Returns
    -------
    dict
    {'model': sklearn classifier object, 
     'scaler': sklearn.preprocessing._data.StandardScaler or None}
        A dictionary containing the fitted final classifier and the
        scaler used for the standardization (if the classifier is
        Logistic Regression).
        
    See Also
    --------
    train_models : Train multiple classifiers on different feature sets.
    """
    initial_models = [model for model in initial_models if model['name'] not in exclude_models]
    
    predictions = pd.DataFrame()
    for model in initial_models:
        col_name = model['name'] + "_" + model['features']
        clf = model['model']
        if type(clf) is dict:
            # If the type is dictionary, it means that it was created
            # with a training function that outputs both the model and
            # a scaler.
            scaler = clf['scaler']
            clf = clf['model']
            
            for train_feature_set in train_feature_sets:
                # make predictions on the same set the model was trained on
                if train_feature_set['name'] == model['features']:
                    scaled_features = pd.DataFrame(scaler.transform(train_feature_set['features']),
                                                   columns=train_feature_set['features'].columns)
                    predictions[col_name] = pd.DataFrame(cross_val_predict(clf, scaled_features, train_target, method='predict_proba'))[1]
            
        else:
            for train_feature_set in train_feature_sets:
                # make predictions on the same set the model was trained on
                if train_feature_set['name'] == model['features']:
                    train_features = train_feature_set['features']
                    predictions[col_name] = pd.DataFrame(cross_val_predict(clf, train_features, train_target, method='predict_proba'))[1]
    
    if append_features:
        feature_sets = [feature_set['features'] for feature_set in train_feature_sets]
        feature_sets.append(predictions)
        final_features = pd.concat(feature_sets, axis=1)
    else:
        final_features = predictions
    
    if final_classifier is None:
        final_classifier = LogisticRegression(max_iter=1000, penalty='l2', C=1e10, random_state=alg_random_state)
    
    if 'LogisticRegression' in str(type(final_classifier)):
        scaler = StandardScaler().fit(final_features)
        final_features = pd.DataFrame(scaler.transform(final_features), columns=final_features.columns)
    else:
        scaler = None
    
    fitted_final_model = final_classifier.fit(final_features, train_target)
    
    return {'model': fitted_final_model, 
            'scaler': scaler}

def test_stacked_models(initial_models, test_feature_sets, test_target, final_model, exclude_models=[], append_features=False, result_row_name=None):
    """
    Evaluate a Stacking classifier with a test set.
    
    This function is similar to test_stacked_models with some
    key differences.
    
    It makes predictions about the test set with the initial
    models and uses these as the features for the fitted
    final_classifier.
    
    Note that there is no cross validation during the prediction
    making, since that would lead to data leakage from the test
    set to the level 1 classifier.
    
    Finally, both the predictions and a DataFrame with result
    metrics are being returned.
    
    Parameters
    ----------
    initial_models : list of dict
        {'name': str,
         'features': str,
         'model': sklearn classifier object or dict}
        A list created by train_models containing the fitted
        classifier models that will be stacked.
    test_feature_sets : list of dict
        {'name': str,
        'features': pandas.DataFrame}
        A list of dictionaries containing a name for the feature
        set and the DataFrame containing the features that the
        classifiers will be tested on.
        
        Note that these dictionaries have to match the ones given
        as input to the instance of train_models used to train the
        initial_models.
    test_target : pandas.Series
        The Series with the target class variable.
    final_model : dict
        {'model': sklearn classifier object, 
         'scaler': sklearn.preprocessing._data.StandardScaler or None}
        A dictionary containing the fitted classifier to be used as
        the level 1 model and the scaler that was used for training.
    exclude_models : list of {'lr', 'dt', 'rf', 'gb', 'nb'}
        If a list of the above strings is passed, any model found
        with those names will be excluded from the stacking (useful
        in order to be able to pass the output of train_models
        directly in initial_models without pruning it).
    append_features : bool, default False
        If True, the training data for the level 0 classifiers will
        also be used for the training of the level 1 classifier.
    result_row_name : str or None
        If a string is given, it will be used as the index name of the
        returned metrics DataFrame. The default behavior creates a name
        with the algorithms used for the stacking, the final classifier,
        and wether the initial features were appended.
        
    Returns
    -------
    dict
    {'results': pandas.DataFrame,
     'predictions': numpy.ndarray}
        The result metrics in a DataFrame row and the predictions array.
        
    See Also
    --------
    train_models : Train multiple classifiers on different feature sets.
    train_stacked_models : Train a Stacking classifier.
    """
    initial_models = [model for model in initial_models if model['name'] not in exclude_models]
    
    predictions = pd.DataFrame()
    for model in initial_models:
        col_name = model['name'] + "_" + model['features']
        clf = model['model']
        if type(clf) is dict:
            # If the type is dictionary, it means that it was created
            # with a training function that outputs both the model and
            # a scaler.
            scaler = clf['scaler']
            clf = clf['model']
            for test_feature_set in test_feature_sets:
                # make predictions on the same set the model was trained on
                if test_feature_set['name'] == model['features']:
                    scaled_features = pd.DataFrame(scaler.transform(test_feature_set['features']),
                                                   columns=test_feature_set['features'].columns)
                    predictions[col_name] = pd.DataFrame(clf.predict_proba(scaled_features))[1]
            
        else:
            for test_feature_set in test_feature_sets:
                # make predictions on the same set the model was trained on
                if test_feature_set['name'] == model['features']:
                    predictions[col_name] = pd.DataFrame(clf.predict_proba(test_feature_set['features']))[1]
    
    if append_features:
        feature_sets = [feature_set['features'] for feature_set in test_feature_sets]
        feature_sets.append(predictions)
        final_features = pd.concat(feature_sets, axis=1)
    else:
        final_features = predictions
    
    if final_model['scaler'] is not None:
        final_features = pd.DataFrame(final_model['scaler'].transform(final_features), columns=final_features.columns)
    
    final_predictions = final_model['model'].predict(final_features)
    final_probabilities = final_model['model'].predict_proba(final_features)[:, 1]
    
    # If no name is provided for the resulting metrics row, create one.
    if result_row_name is None:
        final_row_name = "Algorithms: " + ', '.join(set([model['name'] for model in initial_models]))
        final_row_name += ", with " + str(type(final_model['model'])).split("'")[1].split('.')[-1]
        if append_features:
            final_row_name += " (with appended features)"
    else:
        final_row_name = result_row_name
    
    results = metrics(test_target, final_predictions, final_probabilities).rename(index={0: final_row_name})
    
    return {'results': results,
            'predictions': final_predictions}
