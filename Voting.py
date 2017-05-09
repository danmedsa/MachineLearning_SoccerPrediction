import pickle
import pandas as pd

def Voting(clfs, inp):
    preds = []
    decisions = []
    vote = []

    for clf in clfs:
        pred = clf.predict(inp)
        # print(pred)
        preds.append(pred)

    for i in range(len(preds[0])):
        new = []
        for clf in preds:
            new.append(clf[i])
        decisions.append(new)

    for d in decisions:
        if(d.count('Win') > d.count('Draw') and d.count('Win') > d.count('Defeat')):
            vote.append('Win')

        elif (d.count('Draw') > d.count('Win') and d.count('Draw') > d.count('Defeat')):
            vote.append('Draw')

        elif (d.count('Defeat') > d.count('Draw') and d.count('Defeat') > d.count('Win')):
            vote.append('Defeat')

        else:
            vote.append(d[2])

        # print("Votes:",decisions)
        
    return vote


def predict(inp):
    clfs = []
    inpt = pd.DataFrame(inp)
    open_file = open("Pickle/SVM_clf_All_Matches_NL.pickle", "rb")


    SVM_clf = pickle.load(open_file)
    open_file.close()
    clfs.append(SVM_clf)

    open_file = open("Pickle/GNB_clf_All_Matches_NL.pickle", "rb")


    GNB_clf = pickle.load(open_file)
    open_file.close()
    clfs.append(GNB_clf)

    open_file = open("Pickle/BNB_clf_All_Matches_NL.pickle", "rb")

    
    BNB_clf = pickle.load(open_file)
    open_file.close()
    clfs.append(BNB_clf)

    open_file = open("Pickle/KNN_clf_All_Matches_NL.pickle", "rb")


    KNN_clf = pickle.load(open_file)
    open_file.close()
    clfs.append(KNN_clf)

    open_file = open("Pickle/LR_clf_All_Matches_NL.pickle", "rb")


    LR_CLF = pickle.load(open_file)
    open_file.close()
    clfs.append(LR_CLF)

    result = Voting(clfs,inpt)
    # print("Outcome",result)

    return result


open_file = open("Pickle/All_Matches_NL.pickle", "rb")


matches = pickle.load(open_file)
open_file.close()

TEST_SIZE = 2448

inpt = matches.drop('label', axis = 1)
inpt.tail(TEST_SIZE)
pred = predict(inpt)

x = 0
for i in range(0,TEST_SIZE):
    try:
        lbl = str(matches['label'][len(matches)-(TEST_SIZE)+i])
    except TypeError as err:
        print(i)

    if pred[i] == lbl:
        x = x+1
accuracy = x/TEST_SIZE
print(accuracy)




