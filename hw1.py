import numpy as np
import string

# When you turn this function in to Gradescope, it is easiest to copy and paste this cell to a new python file called hw1.py
# and upload that file instead of the full Jupyter Notebook code (which will cause problems for Gradescope)
def compute_features(names):
    """
    Given a list of names of length N, return a numpy matrix of shape (N, 260)
    with the features described in problem 2b of the homework assignment.

    Parameters
    ----------
    names: A list of strings
        The names to featurize, e.g. ["albert einstein", "marie curie"]

    Returns
    -------
    numpy.array:
        A numpy array of shape (N, 260)
    """
    feature_matrix = np.zeros((len(names), 260))
    alphabet_list = list(string.ascii_lowercase)
    badge_number = 0
    for name in names:
        first_name, last_name = name.split()
        first_name = first_name[:5]
        last_name = last_name[:5]
        char_pos = 0
        a_index = 0

        for l in first_name:
            a_index = 0
            for a in alphabet_list:
                if (l == a):
                    feature_matrix[badge_number, char_pos * 26 + a_index] = 1
                a_index += 1
            char_pos += 1

        char_pos = 5

        for l in last_name:
            a_index = 0
            for a in alphabet_list:
                if (l == a):
                    feature_matrix[badge_number, char_pos * 26 + a_index] = 1
                a_index += 1
            char_pos += 1

        badge_number += 1

    return feature_matrix
