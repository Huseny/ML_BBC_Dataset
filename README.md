# The BBC NEWS DATASET

## Naive Bayes ##
The Naive Bayes works based on Bayes’s theorem. We used the Decimal library to prevent python from rounding off very small values into zero. It iterates over each label and calculate P(label | article). It does so by converting P(label | article) into last term ∏ P(term∨label) * P(label) for each term. If the term is present in that article we take its term=0 probability, otherwise we take laplace_smoothing/total_terms + laplace_smoothing.
We have been able to achieve an accuracy of 99% for the BBC dataset. But that value is achieved when a carefully selected smoothing value is used. But generally, we have observed that, as the smoothing value increases the accuracy of the prediction decreases dramatically. This is illustrated in the graph (run the function “try_naive” in the results.py file to see the graph).

## Logistic Regression ##
we initialize our weights as zero first and then update them on each loop of the training based on the learning rate and gradient descent value. Then after finishing the training phase, it is assumed that we have good enough weights to predict labels of the test set.
We have achieved similar results with Logistic Regression too. We have tried the logistic regression with different learning rates and the same trend as above is observed. We got an accuracy of 97.8% when running with learning rate of 0.01 and the accuracy decreases as we move forward. But, it is noticeable that the decrease isn’t as sharp the Naive Bayes one. To see the graph run the “try_logistic” inside the results.py page.
