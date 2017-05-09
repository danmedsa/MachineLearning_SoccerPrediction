import sqlite3
import pandas as pd
from time import time
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.metrics import make_scorer
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn import svm
import seaborn as sns
from itertools import product
import warnings


warnings.simplefilter("ignore")

def get_fifa_data(matches, player_stats, path = None, data_exists = False):
    ''' Gets fifa data for all matches. '''  

    #Check if fifa data already exists
    if data_exists == True:
        
        fifa_data = pd.read_pickle(path)
        
    else:
        
        print("Collecting fifa data for each match...")       
        start = time()
        
        #Apply get_fifa_stats for each match
        fifa_data = matches.apply(lambda x :get_fifa_stats(x, player_stats), axis = 1)
        
        end = time()    
        print("Fifa data collected in {:.1f} minutes".format((end - start)/60))

    #Return fifa_data
    return fifa_data

def get_fifa_stats(match, player_stats):
    ''' Aggregates fifa stats for a given match. '''    
    

    #Define variables
    match_id =  match.match_api_id
    date = match['date']
    players = ['home_player_1', 'home_player_2', 'home_player_3', "home_player_4", "home_player_5",
               "home_player_6", "home_player_7", "home_player_8", "home_player_9", "home_player_10",
               "home_player_11", "away_player_1", "away_player_2", "away_player_3", "away_player_4",
               "away_player_5", "away_player_6", "away_player_7", "away_player_8", "away_player_9",
               "away_player_10", "away_player_11"]
    player_stats_new = pd.DataFrame()

    names = []
    
    #Loop through all players
    for player in players:   
            
        #Get player ID
        player_id = match[player]
        
        #Get player stats 
        stats = player_stats[player_stats.player_api_id == player_id]
            
        #Identify current stats       
        current_stats = stats[stats.date < date].sort_values(by = 'date', ascending = False)[:1]
        
        if np.isnan(player_id) == True:
            overall_rating = pd.Series(0)
        else:
            current_stats.reset_index(inplace = True, drop = True)
            overall_rating = pd.Series(current_stats.loc[0, "overall_rating"])

        #Rename stat
        name = "{}_overall_rating".format(player)
        names.append(name)
        #Aggregate stats
        player_stats_new = pd.concat([player_stats_new, overall_rating], axis = 1)

    player_stats_new.columns = names        
    player_stats_new['match_api_id'] = match_id
    # print(player_stats_new)

    player_stats_new.reset_index(inplace = True, drop = True)
    
    #Return player stats   
    return player_stats_new.ix[0] 


def create_feables(matches, fifa, get_overall = False, horizontal = True, x = 10, verbose = True):
    ''' Create and aggregate features and labels for all matches. '''

    #Get fifa stats features
    fifa_stats = get_overall_fifa_rankings(fifa, get_overall)
    # print(fifa_stats)
    
    if verbose == True:
        print("Generating match features...")
    start = time()
    

    #Get match features for all matches
    match_stats = matches.apply(lambda x: get_match_features(x, matches, x = 10), axis = 1)
    # print("Match Stats:",match_stats)

    end = time()
    if verbose == True:
        print("Match features generated in {:.1f} minutes".format((end - start)/60))
    
    if verbose == True:    
        print("Generating match labels...")
    start = time()
    
    #Create match labels
    labels = matches.apply(get_match_label, axis = 1)
    # print("Labels:", labels)
    end = time()
    if verbose == True:
        print("Match labels generated in {:.1f} minutes".format((end - start)/60))
    
    start = time()

    #Merges features and labels into one frame
    features = pd.merge(match_stats, fifa_stats, on = 'match_api_id', how = 'left')
    feables = pd.merge(features, labels, on = 'match_api_id', how = 'left')
    
    #Drop NA values
    feables.dropna(inplace = True)
    
    #Return preprocessed data
    return feables

def get_last_matches(matches, date, team, x = 10):
    ''' Get the last x matches of a given team. '''
    
    #Filter team matches from matches
    team_matches = matches[(matches['home_team_api_id'] == team) | (matches['away_team_api_id'] == team)]
    # print("Team Matches:", team_matches)    

    last_matches = team_matches[team_matches.date < date].sort_values(by = 'date', ascending = False).iloc[0:x,:]
    # print("Last Matches:", last_matches)                       


    #Return last matches
    return last_matches


def get_match_features(match, matches, x = 10):
    ''' Create match specific features for a given match. '''

    #Define variables
    date = match.date
    home_team = match.home_team_api_id
    away_team = match.away_team_api_id
    
    #Get last x matches of home and away team
    matches_home_team = get_last_matches(matches, date, home_team, x = 10)
    matches_away_team = get_last_matches(matches, date, away_team, x = 10)
    
    # TODO: Try without last matches against eachother... Too much time for change
    #Get last x matches of both teams against each other
    last_matches_against = get_last_matches_against_eachother(matches, date, home_team, away_team, x = 3)
    
    #Create goal variables
    home_goals = get_goals(matches_home_team, home_team)
    away_goals = get_goals(matches_away_team, away_team)
    home_goals_conceided = get_goals_conceided(matches_home_team, home_team)
    away_goals_conceided = get_goals_conceided(matches_away_team, away_team)
    
    #Define result data frame
    result = pd.DataFrame()
    

    #Define ID features
    result.loc[0, 'match_api_id'] = match.match_api_id
    # result.loc[0, 'league_id'] = match.league_id

    #Create match features
    result.loc[0, 'home_team_goals_difference'] = home_goals - home_goals_conceided
    result.loc[0, 'away_team_goals_difference'] = away_goals - away_goals_conceided
    result.loc[0, 'games_won_home_team'] = get_wins(matches_home_team, home_team) 
    result.loc[0, 'games_won_away_team'] = get_wins(matches_away_team, away_team)
    result.loc[0, 'games_against_won'] = get_wins(last_matches_against, home_team)
    result.loc[0, 'games_against_lost'] = get_wins(last_matches_against, away_team)

    #Return match features
    # print("Match Features Result [0]:", result.loc[0])
    return result.loc[0]

def get_match_label(match):
    ''' Derives a label for a given match. '''
    
    #Define variables
    home_goals = match['home_team_goal']
    away_goals = match['away_team_goal']
     
    label = pd.DataFrame()
    label.loc[0,'match_api_id'] = match['match_api_id'] 

    #Identify match label  
    if home_goals > away_goals:
        label.loc[0,'label'] = "Win"
    if home_goals == away_goals:
        label.loc[0,'label'] = "Draw"
    if home_goals < away_goals:
        label.loc[0,'label'] = "Defeat"

    #Return label        
    return label.loc[0]

def get_wins(matches, team):
    ''' Get the number of wins of a specfic team from a set of matches. '''
    
    #Find home and away wins
    home_wins = int(matches.home_team_goal[(matches.home_team_api_id == team) & (matches.home_team_goal > matches.away_team_goal)].count())
    away_wins = int(matches.away_team_goal[(matches.away_team_api_id == team) & (matches.away_team_goal > matches.home_team_goal)].count())

    total_wins = home_wins + away_wins

    #Return total wins
    return total_wins     
def get_goals(matches, team):
    ''' Get the goals of a specfic team from a set of matches. '''
    
    #Find home and away goals
    home_goals = int(matches.home_team_goal[matches.home_team_api_id == team].sum())
    away_goals = int(matches.away_team_goal[matches.away_team_api_id == team].sum())

    total_goals = home_goals + away_goals
    
    #Return total goals
    return total_goals

def get_goals_conceided(matches, team):
    ''' Get the goals conceided of a specfic team from a set of matches. '''

    #Find home and away goals
    home_goals = int(matches.home_team_goal[matches.away_team_api_id == team].sum())
    away_goals = int(matches.away_team_goal[matches.home_team_api_id == team].sum())

    total_goals = home_goals + away_goals

    #Return total goals
    return total_goals

def get_last_matches_against_eachother(matches, date, home_team, away_team, x = 10):
    ''' Get the last x matches of two given teams. '''


    #Find matches of both teams
    home_matches = matches[(matches['home_team_api_id'] == home_team) & (matches['away_team_api_id'] == away_team)]    
    away_matches = matches[(matches['home_team_api_id'] == away_team) & (matches['away_team_api_id'] == home_team)]  
    total_matches = pd.concat([home_matches, away_matches])
    
    #Get last x matches
    try:    
        last_matches = total_matches[total_matches.date < date].sort_values(by = 'date', ascending = False).iloc[0:x,:]
    except:
        last_matches = total_matches[total_matches.date < date].sort_values(by = 'date', ascending = False).iloc[0:total_matches.shape[0],:]
        
        #Check for error in data
        if(last_matches.shape[0] > x):
            print("Error in obtaining matches")

    #Return data
    return last_matches


def get_overall_fifa_rankings(fifa, get_overall = False):
    ''' Get overall fifa rankings from fifa data. '''
    

    temp_data = fifa
    
    #Check if only overall player stats are desired
    if get_overall == True:
        
        #Get overall stats
        data = temp_data.loc[:,(fifa.columns.str.contains('overall_rating'))]
        data.loc[:,'match_api_id'] = temp_data.loc[:,'match_api_id']
    else:
        
        #Get all stats except for stat date
        cols = fifa.loc[:,(fifa.columns.str.contains('date_stat'))]
        temp_data = fifa.drop(cols.columns, axis = 1)        
        data = temp_data

    #Return data
    return data


def explore_data(features, inputs, path):
    ''' Explore data by plotting KDE graphs. '''
    #Define figure subplots
    fig = plt.figure(1)
    fig.subplots_adjust(bottom= -1, left=0.025, top = 2, right=0.975)
    
    #Loop through features    
    i = 1
    for col in features.columns:
        
        #Set subplot and plot format        
        sns.set_style("whitegrid")
        sns.set_context("paper", font_scale = 0.5, rc={"lines.linewidth": 1})
        plt.subplot(7,7,0 + i)
        j = i - 1
        
        #Plot KDE for all labels
        print("KDE Matrix:")
        print(inputs[inputs['label'] == 'Win'].iloc[:,j])
        #TODO: Check the inputs

        sns.distplot(inputs[inputs['label'] == 'Win'].iloc[:,j], hist = False, label = 'Win')
        sns.distplot(inputs[inputs['label'] == 'Draw'].iloc[:,j], hist = False, label = 'Draw')
        sns.distplot(inputs[inputs['label'] == 'Defeat'].iloc[:,j], hist = False, label = 'Defeat')
        plt.legend()
        i = i + 1
    
    #Define plot format    
    DefaultSize = fig.get_size_inches()
    fig.set_size_inches((DefaultSize[0]*1.2, DefaultSize[1]*1.2))

    plt.show()
    
    #Compute and print label weights
    labels = inputs.loc[:,'label']
    class_weights = labels.value_counts() / len(labels)
    print(class_weights)
    
    #Store description of all features
    feature_details = features.describe().transpose()

    #Return feature details
    return feature_details

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
    start = time()
    sqlite_file = 'database.sqlite'

    conn = sqlite3.connect(sqlite_file)

    c = conn.cursor()


    # #  #Fetching required data tables
    player_data = pd.read_sql("SELECT * FROM Player;", conn)
    player_stats_data = pd.read_sql("SELECT * FROM Player_Attributes;", conn)
    team_data = pd.read_sql("SELECT * FROM Team;", conn)
    match_data = pd.read_sql("SELECT * FROM Match", conn)  
    # match_data = pd.read_sql("SELECT * FROM Match WHERE league_id = 7809", conn)  
    ### use to get match data from leagues 
    #WHERE league_id =  LaLiga = 21518 BPL = 1729 BUndesliga = 7809   Serie A = 10257

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


    #Reduce match data to fulfill run time requirements
    rows = ["country_id", "league_id", "season", "stage", "date", "match_api_id", "home_team_api_id", 
            "away_team_api_id", "home_team_goal", "away_team_goal", "home_player_1", "home_player_2",
            "home_player_3", "home_player_4", "home_player_5", "home_player_6", "home_player_7", 
            "home_player_8", "home_player_9", "home_player_10", "home_player_11", "away_player_1",
            "away_player_2", "away_player_3", "away_player_4", "away_player_5", "away_player_6",
            "away_player_7", "away_player_8", "away_player_9", "away_player_10", "away_player_11"]
    match_data.dropna(subset = rows, inplace = True)
    # match_data = match_data.tail(500)  #1500)
        


    ## Generating features, exploring the data, and preparing data for model training
    #Generating or retrieving already existant FIFA data
    fifa_data = get_fifa_data(match_data, player_stats_data, data_exists = False)

    feables = create_feables(match_data, fifa_data, get_overall = True)

    inputs = feables.drop('match_api_id', axis = 1)

    print(inputs.columns.values)
    # Get labels from features
    labels = inputs.loc[:,'label']


    #PCA Attempt
    PCAinput = inputs.drop('label', axis = 1)
    # # print(PCAinput)
    # pca = PCA(n_components=PCAinput.shape[1])
    # X_std = StandardScaler().fit_transform(PCAinput)
    # pca.fit(X_std)
    # print(pca.explained_variance_ratio_)
  

    # print("{} : {}".format(cmps,var))
    # print(PCAinput.get(1))
    input_headers = PCAinput.columns.values

    # mean_vec = np.mean(X_std, axis=0)
    # cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
    # print('Covariance matrix \n%s' %cov_mat)

    # print(PCAinput)

    # print('NumPy covariance matrix: \n%s' %np.cov(X_std.T))


    X = PCAinput.head(TRAIN_SIZE)
    y = inputs['label'][:TRAIN_SIZE]


    clf = svm.SVC()
    clf.fit(X,y)
    get_accuracy("SVC",clf,TEST_SIZE, PCAinput, inputs)


    GNB_clf = GaussianNB()
    GNB_clf.fit(X,y)
    get_accuracy("GaussianNB",GNB_clf,TEST_SIZE, PCAinput, inputs)


    BNB_clf = BernoulliNB()
    BNB_clf.fit(X,y)
    get_accuracy("BultinomialNB",BNB_clf,TEST_SIZE, PCAinput, inputs)


    # KNN_clf7 =  KNeighborsClassifier(n_neighbors=7)
    # KNN_clf7.fit(X,y)
    # get_accuracy("kNN7",KNN_clf7,TEST_SIZE, PCAinput, inputs)



    # KNN_clf13 =  KNeighborsClassifier(n_neighbors=13)
    # KNN_clf13.fit(X,y)
    # get_accuracy("kNN13",KNN_clf13,TEST_SIZE, PCAinput, inputs)


    KNN_clf23 =  KNeighborsClassifier(n_neighbors=23)
    KNN_clf23.fit(X,y)
    get_accuracy("kNN23",KNN_clf23,TEST_SIZE, PCAinput, inputs)


    # KNN_clf27 =  KNeighborsClassifier(n_neighbors=27)
    # KNN_clf27.fit(X,y)
    # get_accuracy("kNN27",KNN_clf27,TEST_SIZE, PCAinput, inputs)


    LR_clf = LogisticRegression(multi_class = "ovr", solver = "sag", class_weight = 'balanced')
    LR_clf.fit(X,y)
    get_accuracy("LogReg",LR_clf,TEST_SIZE, PCAinput, inputs)



    print(inputs.shape)

    conn.close()

main()