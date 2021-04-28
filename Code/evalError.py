

def evalError(decision, y, weight):

### Arguments:
### The function evalue the error of a weak classifier
### Input:
###         decision --- The decision which is made by the weak classifier
###         y --- label of training data
###         weight --- weight of training data
### Output:
###         err --- Error of a weak classifier

    err = 0.

    for i, val in enumerate(decision):
        if val != y[i]:
            err += weight[i]

    return err
