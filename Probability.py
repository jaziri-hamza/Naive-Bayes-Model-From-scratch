import numpy as np
import pandas as pd

class Probability:

    def __init__(self):
        pass


    """
        return unique values of column
        ....

        Attributes
        ----------
        column: Type ->  pandas.core.series.Series
    """
    @staticmethod
    def getUniqueValues(column):
        if isinstance(column, pd.core.series.Series): # pandas
            return column.unique()
        raise Exception("Invalide Type")

    """
        return statistique information about column
            total number, and total number of each class
            {
                __total: int
                __classes__: {
                    class1: int
                    class2: int
                    ...
                }
            }
    """
    @staticmethod
    def getColumnInfo(column):
        info = {}
        if isinstance(column, pd.core.series.Series): # pandas
            info["__total_count__"] = column.shape[0]
            info["__classes__"] = {}
            values_counts = column.value_counts()
            for label in Probability.getUniqueValues(column):
                info["__classes__"][label] = {} 
                info["__classes__"][label]["__count__"] = values_counts.at[label]
            return info
        raise Exception("Invalide Type")
    

    @staticmethod
    def _isContinous(column):
        return column.dtypes == 'float64'

    @staticmethod
    def getColumnInfoConstraintTarget(column, target):
        if Probability._isContinous(column) == True:
            return Probability.getGaussianInfo(column, target)
        info = Probability.getColumnInfo(column)
        info["is_continous"] = False
        targetLabels = Probability.getUniqueValues(target)
        if isinstance(column, pd.core.series.Series): # pandas
            df = pd.DataFrame({'column': column, 'target': target})
            for targetLabel in targetLabels:
                for inputClass in info["__classes__"]:
                    info["__classes__"][inputClass][targetLabel] = {}
                    info["__classes__"][inputClass][targetLabel]["__count__"]  = df[ (df["column"] == inputClass) & (df["target"] == targetLabel) ].count().at["column"]
                    info["__classes__"][inputClass][targetLabel]["__prob__"]  = info["__classes__"][inputClass][targetLabel]["__count__"] / info["__total_count__"]  
            return info
        raise Exception("Invalide Type")
        
    @staticmethod
    def getTargetInfoProb(infoTarget, target):
        for label in Probability.getUniqueValues(target):
            infoTarget["__classes__"][label]["__prob__"] = infoTarget["__classes__"][label]["__count__"] / infoTarget["__total_count__"]
        return infoTarget
        
    """
        calcul the mean, the variance
        Attribute
        ---------
        column: it's a panda column (Series type)
            column of <Continous value> like age, height, width, etc..
    """
    @staticmethod
    def getGaussianInfo(column, target):
        # print(column)
        if not isinstance(column, pd.core.series.Series): # pandas
            raise Exception("Invalide type")
        df = pd.DataFrame({'column': column, 'target': target})
        targetLabels = Probability.getUniqueValues(target)
        info = {}
        info["__total_count__"] = column.shape[0]
        info["is_continous"] = True
        for targetLabel in targetLabels:
            info[targetLabel] = {} 
            info[targetLabel]["__count__"] = df[ df["target"] == targetLabel ].count().at["column"]
            info[targetLabel]["__mean__"] = df["column"][df["target"] == targetLabel].sum() / info["__total_count__"]
            newColumn = df["column"][df["target"] == targetLabel].transform( lambda x: np.square(x - info[targetLabel]["__mean__"]) ) / info[targetLabel]["__count__"]
            info[targetLabel]["__variance__"] = newColumn.sum()/info[targetLabel]["__count__"]
        return info

    