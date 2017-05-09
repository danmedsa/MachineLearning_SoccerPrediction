import sqlite3
import pandas as pd
from time import time
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn import svm
from itertools import product
import warnings


warnings.simplefilter("ignore")


def get_accuracy(name, clf, test_size, pred, compare):
    pred = clf.predict(pred.tail(test_size))
    x = 0
    for i in range(0,test_size):
        # print(str(inputs['label'][len(inputs)-test_size+i]), pred[i])
        if pred[i] == str(compare['label'][len(compare)-test_size+i]):
            x = x+1
    accuracy = x/test_size
    print(name + ":", accuracy)

def main():

    #All Matches
    TRAIN_SIZE = 18916
    TEST_SIZE = 2448

    # #La Liga
    # TRAIN_SIZE = 2396
    # TEST_SIZE = 311

    # #Premier League
    # TRAIN_SIZE = 2621
    # TEST_SIZE = 341

    #Bundesliga
    # TRAIN_SIZE = 2102
    # TEST_SIZE = 274

    # #Serie A
    # TRAIN_SIZE = 2431
    # TEST_SIZE = 316

    open_file = open("Pickle/All_Matches_NL.pickle", "rb")
    # open_file = open("Pickle/LaLiga_Input.pickle", "rb")
    # open_file = open("Pickle/BPL_Input.pickle", "rb")
    # open_file = open("Pickle/Bundesliga_Input.pickle", "rb")
    # open_file = open("Pickle/SerieA_Input.pickle", "rb")

    inputs = pickle.load(open_file)
    open_file.close()



    print(inputs.columns.values)
    # Get labels from features
    labels = inputs.loc[:,'label']

    PCAinput = inputs.drop('label', axis = 1)

    input_headers = PCAinput.columns.values

    X = PCAinput.head(TRAIN_SIZE)
    y = inputs['label'][:TRAIN_SIZE]


    open_file = open("Pickle/SVM_clf_All_Matches_NL.pickle","rb")
    # open_file = open("Pickle/SVM_clf_LaLiga.pickle","rb")
    # open_file = open("Pickle/SVM_clf_BPL.pickle","rb")
    # open_file = open("Pickle/SVM_clf_Bundesliga.pickle","rb")
    # open_file = open("Pickle/SVM_clf_SerieA.pickle","rb")
    clf = pickle.load(open_file)
    open_file.close()
    get_accuracy("SVC",clf,TEST_SIZE, PCAinput, inputs)


    open_file = open("Pickle/GNB_clf_All_Matches_NL.pickle","rb")
    # open_file = open("Pickle/GNB_clf_LaLiga.pickle","rb")
    # open_file = open("Pickle/GNB_clf_BPL.pickle","rb")
    # open_file = open("Pickle/GNB_clf_Bundesliga.pickle","rb")
    # open_file = open("Pickle/GNB_clf_SerieA.pickle","rb")
    GNB_clf = pickle.load(open_file)
    open_file.close()
    get_accuracy("GaussianNB",GNB_clf,TEST_SIZE, PCAinput, inputs)


    open_file = open("Pickle/BNB_clf_All_Matches_NL.pickle","rb")
    # open_file = open("Pickle/BNB_clf_LaLiga.pickle","rb")
    # open_file = open("Pickle/BNB_clf_BPL.pickle","rb")
    # open_file = open("Pickle/BNB_clf_Bundesliga.pickle","rb")
    # open_file = open("Pickle/BNB_clf_SerieA.pickle","rb")
    BNB_clf = pickle.load(open_file)
    open_file.close()
    get_accuracy("BultinomialNB",BNB_clf,TEST_SIZE, PCAinput, inputs)


    open_file = open("Pickle/KNN23_clf_All_Matches_NL.pickle","rb")
    # open_file = open("Pickle/KNN23_clf_LaLiga.pickle","rb")
    # open_file = open("Pickle/KNN23_clf_BPL.pickle","rb")
    # open_file = open("Pickle/KNN23_clf_Bundesliga.pickle","rb")
    # open_file = open("Pickle/KNN23_clf_SerieA.pickle","rb")
    KNN_clf23 = pickle.load(open_file)
    open_file.close()
    get_accuracy("kNN23",KNN_clf23,TEST_SIZE, PCAinput, inputs)



    open_file = open("Pickle/LR_clf_All_Matches_NL.pickle","rb")
    # open_file = open("Pickle/LR_clf_LaLiga.pickle","rb")
    # open_file = open("Pickle/LR_clf_BPL.pickle","rb")
    # open_file = open("Pickle/LR_clf_Bundesliga.pickle","rb")
    # open_file = open("Pickle/LR_clf_SerieA.pickle","rb")
    LR_clf = pickle.load(open_file)
    open_file.close()
    get_accuracy("LogReg",LR_clf,TEST_SIZE, PCAinput, inputs)

    print(inputs.shape)

main()