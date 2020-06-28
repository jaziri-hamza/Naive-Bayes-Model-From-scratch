import numpy as np
from Probability import Probability
import pandas as pd

"""
    Naive bayes is a machine learning model for classification
"""
class NaiveBayes:

    def __init__(self, X,Y):
        if (isinstance(X, pd.core.series.Series) or isinstance(X, pd.DataFrame)) and (isinstance(Y, pd.core.series.Series) or isinstance(Y, pd.DataFrame)):
            self._X = X
            self._Y = Y
        else:
            raise Exception("Invalide Type")

    def _setup(self):
        self._X_info = []
        self._Y_info = Probability.getTargetInfoProb(Probability.getColumnInfo(self._Y) , self._Y)
        self._target_unique_values = Probability.getUniqueValues(self._Y)
    
    def fit(self):
        self._setup()
        for i in range(self._X.shape[1]): # loop on columns
            self._X_info.append( Probability.getColumnInfoConstraintTarget(self._X.iloc[:, i], self._Y ) )
        
    def _calcLikehood(self, mean, variance, label):
        return 1 / np.sqrt( 2 * np.pi * np.square(variance) )  +  np.exp( - (np.square((label - mean)) / 2 * variance) )
    
    def _getProbXContinous(self, index, label, target):
        # print("Likehoooooood", Likehood)
        prob = 1
        for _target in self._target_unique_values: 
            mean = self._X_info[index][target]["__mean__"]
            variance = self._X_info[index][target]["__variance__"]
            if(_target == target):
                prob *= self._calcLikehood(mean, variance, label)
            else:
                prob *= 1 - self._calcLikehood(mean, variance, label)
        return prob

    """
        index : the column
        label: column attribute
        target: target albel (class)
    """
    def _getProbX(self, index, label, target): # return double
        # print(self._X_info[index]["is_continous"])
        if self._X_info[index]["is_continous"] == True:
            return self._getProbXContinous(index, label, target)
        prob = 1
        for _target in self._X_info[index]["__classes__"][label]: 
            # print(_target)
            if(_target == "__count__"):
                continue
            if(_target == target):
                prob *= self._X_info[index]["__classes__"][label][_target]["__prob__"]
            else:
                # print(self._X_info[index]["__classes__"][label][_target]["__prob__"])
                prob *= 1 - self._X_info[index]["__classes__"][label][_target]["__prob__"]
        return prob #self._X_info[index]["__classes__"][label][target]["__prob__"]


    """
        get the probablity of the label constraint targets
        label: is the target class
    """
    def _getProbY(self, label): # return double
        return self._Y_info["__classes__"][label]["__prob__"]
    
    
    def predict(self, X_test):
        result = []
        for row in range(X_test.shape[0]):
            _res = {}
            for targetLabel in self._target_unique_values:
                _res[targetLabel] = 1
                for col in range(X_test.shape[1]):
                    _res[targetLabel] *= self._getProbX(col, X_test.iloc[row, col], targetLabel)
                    # print(col, "=>", _res[targetLabel])
                _res[targetLabel] *= self._getProbY(targetLabel)
            for i in _res:
                # print(i, " => res ", _res[i])
                _res[i] = _res[i] / self._sum(_res)
            nb = _res[list(_res.keys())[0]]
            class_name = list(_res.keys())[0]
            for i in _res:
                if(_res[i] >= nb):
                    nb = _res[i]
                    class_name = i
            result.append(class_name)
        return result

    def _sum(self, res):
        sum = 0
        for i in res:
            sum += res[i]
        return sum


    def _printLabel(self):
        for i in self._X_info:
            print("----")
            print(i)