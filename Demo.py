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

        print("kNN Outcome:", d[3])
        print("       [[ SVM ,  NB ,   BNB ,   kNN   ,   LR ]]")
        print("Votes:",decisions)


        
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

print("Home: Santos vs. Away: Toluca")


inpt = {'home_team_goals_difference': [5], 'away_team_goals_difference' : [1] ,\
 'games_won_home_team': [5], 'games_won_away_team': [8], 'games_against_won':[18],\
 'games_against_lost':[20],\

 'home_player_1_overall_rating': [77], 'home_player_2_overall_rating': [72],\
 'home_player_3_overall_rating': [74], 'home_player_4_overall_rating': [78],\
 'home_player_5_overall_rating': [71], 'home_player_6_overall_rating': [73],\
 'home_player_7_overall_rating': [72], 'home_player_8_overall_rating': [61],\
 'home_player_9_overall_rating': [72], 'home_player_10_overall_rating': [74],\
 'home_player_11_overall_rating': [73],\

 'away_player_1_overall_rating' : [77],\
 'away_player_2_overall_rating':[69] , 'away_player_3_overall_rating': [70],\
 'away_player_4_overall_rating': [74] , 'away_player_5_overall_rating': [70],\
 'away_player_6_overall_rating': [74], 'away_player_7_overall_rating': [73],\
 'away_player_8_overall_rating': [69], 'away_player_9_overall_rating': [74],\
 'away_player_10_overall_rating': [73], 'away_player_11_overall_rating': [74]}


predict(inpt)

