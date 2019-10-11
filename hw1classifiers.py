from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
import random
import numpy as np


def train_and_evaluate_sgd(X_train, y_train, X_test, y_test):
    """
    Trains a SGDClassifier on the training data and computes two accuracy scores, the
    accuracy of the classifier on the training data and the accuracy of the decision
    tree on the testing data.

    Parameters
    ----------
    X_train: np.array
        The training features of shape (N_train, k)
    y_train: np.array
        The training labels of shape (N_train)
    X_test: np.array
        The testing features of shape (N_test, k)
    y_test: np.array
        The testing labels of shape (N_test)

    Returns
    -------
    The training and testing accuracies represented as a tuple of size 2.
    """
    model = SGDClassifier(loss='log', max_iter=10000).fit(X_train, y_train)
    return model.score(X_train, y_train), model.score(X_test, y_test)


def train_and_evaluate_decision_tree(X_train, y_train, X_test, y_test):
    """
    Trains an unbounded decision tree on the training data and computes two accuracy scores, the
    accuracy of the decision tree on the training data and the accuracy of the decision
    tree on the testing data.

    The decision tree should use the information gain criterion (set criterion='entropy')

    Parameters
    ----------
    X_train: np.array
        The training features of shape (N_train, k)
    y_train: np.array
        The training labels of shape (N_train)
    X_test: np.array
        The testing features of shape (N_test, k)
    y_test: np.array
        The testing labels of shape (N_test)

    Returns
    -------
    The training and testing accuracies represented as a tuple of size 2.
    """
    model = DecisionTreeClassifier(criterion = 'entropy').fit(X_train, y_train)
    return model.score(X_train, y_train), model.score(X_test, y_test)


def train_and_evaluate_decision_stump(X_train, y_train, X_test, y_test):
    """
    Trains a decision stump of maximum depth 4 on the training data and computes two accuracy scores, the
    accuracy of the decision stump on the training data and the accuracy of the decision
    tree on the testing data.

    The decision tree should use the information gain criterion (set criterion='entropy')

    Parameters
    ----------
    X_train: np.array
        The training features of shape (N_train, k)
    y_train: np.array
        The training labels of shape (N_train)
    X_test: np.array
        The testing features of shape (N_test, k)
    y_test: np.array
        The testing labels of shape (N_test)

    Returns
    -------
    The training and testing accuracies represented as a tuple of size 2.
    """
    model = DecisionTreeClassifier(criterion = 'entropy', max_depth = 4).fit(X_train, y_train)
    return model.score(X_train, y_train), model.score(X_test, y_test)


def train_and_evaluate_sgd_with_stumps(X_train, y_train, X_test, y_test):
    """
    Trains a SGDClassifier with stumps on the training data and computes two accuracy scores, the
    accuracy of the classifier on the training data and the accuracy of the decision
    tree on the testing data.

    Parameters
    ----------
    X_train: np.array
        The training features of shape (N_train, k)
    y_train: np.array
        The training labels of shape (N_train)
    X_test: np.array
        The testing features of shape (N_test, k)
    y_test: np.array
        The testing labels of shape (N_test)

    Returns
    -------
    The training and testing accuracies represented as a tuple of size 2.
    """
    # Initialize feature space with dimensionality 50; each of 50 features are output of a DT of depth 4
    X_new_train = np.zeros((np.shape(X_train, 0), 50))
    X_new_test = np.zeros((np.shape(X_test, 0), 50))
    # For each of 50 stumps, predict a binary label for each k-dimensional instance x
    for i in range(50):
        random_features = np.random.choice(range(len(X_train[0])), int(len(X_train[0]) / 2), replace=False)
        feature_new_train = X_train[:, random_features]
        feature_new_test = X_test[:, random_features]
        model = DecisionTreeClassifier(criterion = 'entropy', max_depth = 4).fit(feature_new_train, y_train)
        X_new_train[:, i] = model.predict(feature_new_train)
        X_new_test[:, i] = model.predict(feature_new_test)
    # Run SGD on the new dataset of dimensionality 50
    return train_and_evaluate_sgd(X_new_train, y_train, X_new_test, y_test)
