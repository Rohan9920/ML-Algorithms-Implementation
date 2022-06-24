# Naive Bayes algorithm #

This is a classification algorithm is based on the Naive Bayes theorem. It uses probabilities of features to determine the dependent class.

## Important points to remember ##
- Works best when all the variables are categorical.
- For continuous features, normality is assumed.
- The algorithm assumes that all the predictor features are independent. 
- Can be used as a baseline algorithm for classification.
- Laplace smoothing must be used to avoid zero probability issue.

## Algorithm steps: ##
The example used in the code is 'classification of text messages into spam/not spam'.
- Split the data into training and testing.
- Calculate prior probabilities of the output class. (Eg: No of samples that are spam messages)
- Record the count of occurences of each word and store it in appropriate class. (Spam/ Not spam)
- For test data, calculate probability of each word and multiply it. Result will be the class with highest probability.
