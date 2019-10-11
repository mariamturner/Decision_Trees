import math
import os
import matplotlib.pyplot as plt


def load_cv_split(fold):
    """
    Parameters
    ----------
    fold: int
        The integer index of the split to load, i.e. 0, 1, 2, 3, or 4

    Returnshw1
    -------
    A tuple of 4 numpy arrays that correspond to the following items:
        X_train, y_train, X_test, y_test
    """
    X_train = np.load('madelon/cv-train-X.'+ str(fold) + '.npy')
    y_train = np.load('madelon/cv-train-y.'+ str(fold) + '.npy')
    X_test = np.load('madelon/cv-heldout-X.'+ str(fold) + '.npy')
    y_test = np.load('madelon/cv-heldout-y.'+ str(fold) + '.npy')
    return X_train, y_train, X_test, y_test


def cross_val(model_type_method, X_train, y_train, X_test, y_test):
    k = 5
    acc_train_total = 0
    acc_held_out_total = 0
    arr_acc_train = []
    arr_acc_held_out = []
    for i in range(k):
        acc_train, acc_held_out = model_type_method(*load_cv_split(i))
        arr_acc_train.append([acc_train])
        arr_acc_held_out.append([acc_held_out])
        acc_train_total += acc_train
        acc_held_out_total += acc_held_out
    acc_train_avg = acc_train_total / k
    acc_held_out_avg = acc_held_out_total / k

    stddev_train = np.std(arr_acc_train)
    stddev_held_out = np.std(arr_acc_held_out)

    conf_interval_train = 2.015 * stddev_train / math.sqrt(k)
    conf_interval_test = 2.015 * stddev_held_out / math.sqrt(k)

    testing_acc = model_type_method(X_train, y_train, X_test, y_test)[1]
    return acc_train_avg, stddev_train, acc_held_out_avg, stddev_held_out, testing_acc


def plot_results(sgd_train_acc, sgd_train_std, sgd_heldout_acc, sgd_heldout_std, sgd_test_acc,
                 dt_train_acc, dt_train_std, dt_heldout_acc, dt_heldout_std, dt_test_acc,
                 dt4_train_acc, dt4_train_std, dt4_heldout_acc, dt4_heldout_std, dt4_test_acc,
                 stumps_train_acc, stumps_train_std, stumps_heldout_acc, stumps_heldout_std, stumps_test_acc):
    """
    Plots the final results from problem 2. For each of the 4 classifiers, pass
    the training accuracy, training standard deviation, held-out accuracy, held-out
    standard deviation, and testing accuracy.

    Although it should not be necessary, feel free to edit this method.
    """
    train_x_pos = [0, 4, 8, 12]
    cv_x_pos = [1, 5, 9, 13]
    test_x_pos = [2, 6, 10, 14]
    ticks = cv_x_pos

    labels = ['sgd', 'dt', 'dt4', 'stumps (4 x 50)']

    train_accs = [sgd_train_acc, dt_train_acc, dt4_train_acc, stumps_train_acc]
    train_errors = [sgd_train_std, dt_train_std, dt4_train_std, stumps_train_std]

    cv_accs = [sgd_heldout_acc, dt_heldout_acc, dt4_heldout_acc, stumps_heldout_acc]
    cv_errors = [sgd_heldout_std, dt_heldout_std, dt4_heldout_std, stumps_heldout_std]

    test_accs = [sgd_test_acc, dt_test_acc, dt4_test_acc, stumps_test_acc]

    fig, ax = plt.subplots()
    ax.bar(train_x_pos, train_accs, yerr=train_errors, align='center', alpha=0.5, ecolor='black', capsize=10, label='train')
    ax.bar(cv_x_pos, cv_accs, yerr=cv_errors, align='center', alpha=0.5, ecolor='black', capsize=10, label='held-out')
    ax.bar(test_x_pos, test_accs, align='center', alpha=0.5, capsize=10, label='test')
    ax.set_ylabel('Accuracy')
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_title('Models')
    ax.yaxis.grid(True)
    ax.legend()
    plt.tight_layout()


X_train = np.load('madelon/train-X.npy')
y_train = np.load('madelon/train-y.npy')
X_test = np.load('madelon/test-X.npy')
y_test = np.load('madelon/test-y.npy')


plot_results(*cross_val(train_and_evaluate_sgd, X_train, y_train, X_test, y_test),
             *cross_val(train_and_evaluate_decision_tree, X_train, y_train, X_test, y_test),
             *cross_val(train_and_evaluate_decision_stump, X_train, y_train, X_test, y_test),
             *cross_val(train_and_evaluate_sgd_with_stumps, X_train, y_train, X_test, y_test))
