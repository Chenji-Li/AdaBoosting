from weakClassif import weakClassif
from evalError import evalError


def returnWeakClassif(x, y, weights):

### Arguments:
### According to the weight of each data, this function helps us choose and return the best weak classifier
### Input:
###         x --- training data ( list )
###         y --- label ( list )
###         weights --- weight of each data ( list )

### Output:
###         theta --- the threshold for the weak classifier
###         p --- the condition of the weak classifier(more details in weakClassif.py)
###         alpha --- the weight for the weak classifier
###         weights_new: the updated weighs of training data

    theta = 0
    p = 0
    err = 1

    # Select the best weak classifier
    for i, val in enumerate(x):
        theta_candidate = val
        p1 = 1
        p2 = -1
        decision1 = weakClassif(x, theta_candidate, p1)
        err1 = evalError(decision1, y, weights)

        decision2 = weakClassif(x, theta_candidate, p2)
        err2 = evalError(decision2, y, weights)

        if err1 < err2:
            err_candidate = err1
            p_candidate = p1
            decision_candidate = decision1
        else:
            err_candidate = err2
            p_candidate = p2
            decision_candidate = decision2

        if err_candidate < err:
            err = err_candidate
            theta = theta_candidate
            p = p_candidate
            decision = decision_candidate

    return theta, p, decision, err

