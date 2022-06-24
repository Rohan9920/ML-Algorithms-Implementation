from collections import defaultdict
import re
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

class NaiveBayes:                 

    def __init__(self):
        self.ham = defaultdict(int)
        self.spam = defaultdict(int)
    

    def fit(self, X_train, y_train):
        self.count_spam = y_train[y_train==0].shape[0]
        self.count_ham = y_train[y_train==1].shape[0]
        for i in range(len(X_train)):
            all_words = self.tokenize(X_train.Message.iloc[i])
            self.word_count(all_words, y_train.iloc[i])

    def predict(self, X_test):
        """
            This function applies the Naive Bayes formula to the input bag of words and outputs a 'spam'
            or 'ham'.
        """
        y_pred = np.array([])
        for i in range(len(X_test)):
            print(i)
            y_pred = np.append(y_pred,self._predict(self.tokenize(X_test.Message.iloc[i])))
        return y_pred
        
    
    def tokenize(self, text):
        """
            This function tokenizes (breaks the sentence into a bag of words) and returns unique words
        """
        text = text.lower()
        all_words = re.findall('[a-z0-9]+', text)
        return set(all_words)

    
    def word_count(self, words, is_spam):
        """
            This function takes in a bag of words and spam indicator and adds the word to the appropriate
            dictionary.
        """
        for word in words:
            if is_spam == 1:
                self.spam[word]+=1
            else:
                self.ham[word]+=1
            
    def _predict(self, words):
        self.prob_spam = self.prob_ham = 1
        for word in words:
            self.prob_spam = self.prob_spam*((self.spam[word])/self.count_spam)
            self.prob_ham = self.prob_ham*((self.spam[word])/self.count_ham)
        if self.prob_spam > self.prob_ham:
            return 1
        else: 
            return 0     
    
