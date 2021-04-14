# AdaBoosting --- Python Code
Here is my python project for realising the algorithm AdaBoosting with one-dimension training data (1D array).<br>
<br>
## Functions<br>
- ### weakClassif
    This function allows us to train a weak classifier with a threshold and a parameter of condition p. It will return the decision of this weak classifier.<br>
    when p = +1, if x < threshold, return 1; if not, return -1 <br>
    when p = -1, if x > threshold, return 1; if not, return -1 <br><br>
- ### evalError
    According to the weight of each training data, the function helps us to compute the error between the decision made by the weak classifier and the label.<br>
    <br>
- ### returnWeakClassif
    The function allows us to choose and return the best weak classifier<br><br>
- ### strongClassif
    It will call the function **returnWeakClassif** to get the best classifier, then compute the weight of the classifier. Finally, the function will return all weak classifiers and their weights<br><br>
 - ### Prediction 
    Then we combine all weak classifiers to a strong classifier and we are able to predict. <br><br><br>
   
With these functions, you are allowed to realise the AdaBoosting with 1D training data.<br>
<br>
### This is my first code on GitHub, thanks for attention !