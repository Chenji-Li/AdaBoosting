import numpy as np


def weakClassif(x, theta, p):

### Arguments:
### A simple classifier with a threshold and a condition
### Input:
###         x --- The input data(training data)
###         theta --- the threshold of the classifier
###         p --- the condition of classifier
### Output:
###         decision --- the decision that the classifier made

    decision = np.zeros(len(x), dtype= int)

    for i, val in enumerate(x):
        if p == 1:
            # when p = 1, positif|threshold|negatif ( +++|threshold|--- )
            if val < theta:
                decision[i] = 1
            else:
                decision[i] = -1

        elif p == -1:
            # when p = -1, negatif|threshold|positif ( ---|threshold|+++ )
            if val > theta:
                decision[i] = 1
            else:
                decision[i] = -1

    return decision



