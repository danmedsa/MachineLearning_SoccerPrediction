# MachineLearning_SoccerPrediction

ZIP of the whole project, due to upload size limitations by github:
https://drive.google.com/file/d/0B5dOWg0uoH8Kb3RmNkNMaDJ0Y00/view?usp=sharing
  + includes 'database.sqlite'
  + includes Pickle folder with all the pickled data





Dataset was obtained from: https://www.kaggle.com/hugomathien/soccer

## How to run the program:

For easy and fast check, All_Matches_Pickled.py can be used. All the data needed has been pickled for fast access and usability. Uncomment the lines according to what you want to run (All leagues, La Liga, etc.) and run the program.

(less than 10 sec.)


For using the program that extracts all the data from the database ‘database.sqlite’ and trains the classifiers, run All_Matches.py. if you want to run for a certain league, go to line 413 and append the ‘WHERE league_id = ###’, the leagues that were used are in line 416 commented. Also, uncomment the correct Train Size and Test Size according to what you will test and train.

(~45 min.)


To try the Voting Approach, run Voting.py. it uses already pickled classifiers to make it fast.


Demo.py was the program used to present the demo. The values for the match that you want to predict can be edited and you can make custom predictions with it.
