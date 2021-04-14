from weakClassif import weakClassif

def prediction(x, AdaBoostingModel):

### Arguments:
### Input:
###        x --- The number that we want to predict(in the form of list, i.e [6], [1, 7, 10])
###        AdaBoostingModel --- The AdaBoosting model which we have trained

### Output:
###        pred --- The prediction (in form of list), the prediction represention a probability which is not
###                 an entier like +1 or -1

    pred = 0

    for i, classifier in enumerate(AdaBoostingModel):
        decision = weakClassif(x, classifier[0], classifier[1])
        pred += classifier[2] * decision

    return pred