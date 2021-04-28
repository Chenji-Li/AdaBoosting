import numpy as np
from returnWeakClassif import returnWeakClassif


def strongClassif(x, y, nb_iteration):
### Arguments:
### Input:
###         x --- Training set
###         y --- Label of Training set
###         nb_iteration --- Number of iteration, we can consider it as the number of weak classifier

### Output:
###         StrongClassifier --- Strong classifier combines several weak classifiers


    # Initialise the weight for the data x
    weight = (1/len(x))*np.ones(x.shape)

    # Initialise a strong classifier, who combine several weak classifiers
    StrongClassifier = []

    for i in range(nb_iteration):

        theta, p, decision, error = returnWeakClassif(x, y, weight)

        # Compute the weight of the weak classifier
        alpha = 0.5 * np.log((1-error)/error)

        # Normalisation term, there is other way to compute
        z = 2 * np.sqrt(error * (1-error))

        # Update the weight of data
        weight = np.multiply(weight, np.exp(- alpha * y * decision)) / z

        WeakClassifer = [theta, p, alpha]

        StrongClassifier.append(WeakClassifer)

    return StrongClassifier



