def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings("ignore", category=DeprecationWarning)

import json
import statsmodels.api as stats_api 
from sklearn.svm import LinearSVR, SVR
import numpy as np
from math import sqrt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from utils import fileLocation, save_obj, load_obj, tsDiffHour, extractFirstTsAndLastTs, \
get_feature, createData, make_plot, createQ2Data, plot_confusion_matrix, cross_val, \
metrics, plot_ROC, FIRST_TS, LAST_TS, cross_val2
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt 
import datetime
# from textblob import TextBlob



PERIOD1 = 1422806400    # PST: 2015 Feb. 1, 8:00 a.m.
PERIOD2 = 1422849600    # PST: 2015 Feb. 1, 8:00 p.m.

def Q1_1(category):
    with open(fileLocation(category), encoding="utf8") as f:
        tweets = f.readlines()
        firstTs = FIRST_TS[category]
        firstTs = firstTs // 3600 * 3600
        lastTs = LAST_TS[category]
        totalHours = tsDiffHour(firstTs, lastTs) + 1

        hourCount = [0] * totalHours
        followerCount = 0
        retweetCount = 0

        for tweet in tweets:
            t = json.loads(tweet)
            ts = t['citation_date']
            # count hour
            hourDiff = tsDiffHour(firstTs, ts)
            hourCount[hourDiff] += 1
            # count follower
            followerCount += t['author']['followers']
            # count retweets
            retweetCount += t['metrics']['citations']['total']

        save_obj(category + '_numTweetsInHour', hourCount)
        # report average number of tweets per hour
        print(category + ': ' + 'Average number of tweets per hour: ' + str(np.mean(hourCount)))
        print(category + ': ' + 'Average number of followers of users posting the tweets: ' +
              str(followerCount / len(tweets)))
        print(category + ': ' + 'Average number of retweets: ' + str(retweetCount / len(tweets)))


def Q1_1_plot(category):
    hourCount = load_obj(category + '_numTweetsInHour')
    hours = [x for x in range(0, len(hourCount))]
    ys = [[hourCount, 'Number of tweets per hour']]
    make_plot(hours, ys, bar=True, xlabel='hours', ylabel='Number of tweets')


def Q1_2():
    hashtags = ['#gohawks', '#nfl', '#sb49', '#gopatriots', '#patriots', '#superbowl']
    for category in hashtags: 
        with open(fileLocation(category), encoding="utf8") as f:
            tweets = f.readlines()
            firstTs = FIRST_TS[category]
            firstTs = firstTs // 3600 * 3600
            lastTs = LAST_TS[category]
            totalHours = tsDiffHour(firstTs, lastTs) + 1

            hourCount = [0] * totalHours
            followerCount = [0] * totalHours
            retweetCount = [0] * totalHours
            tweetCount = [0] * totalHours
            max_followers = [0] * totalHours

            users = set()
            for tweet in tweets:
                t = json.loads(tweet)
                ts = t['citation_date']
                # count hour
                hourDiff = tsDiffHour(firstTs, ts)
                hourCount[hourDiff] += 1
                # count follower
                if t['tweet']['user']['id'] not in users:
                    users.add(t['tweet']['user']['id'])
                    followerCount[hourDiff] += t['author']['followers']
                # count retweets
                retweetCount[hourDiff] += t['metrics']['citations']['total']
                tweetCount [hourDiff] += 1
                max_followers [hourDiff] = max(max_followers [hourDiff], t['author']['followers'])

            time_of_day = [0] * totalHours
            i = 0
            while i < totalHours -1:
                time_of_day [i] = i %24
                i += 1

            dataset = np.array( [tweetCount, retweetCount, followerCount, max_followers, time_of_day])
            dataset = dataset.transpose()
            y = dataset [1:,0]
            X = dataset [:-1, 0:5]
            
            result = stats_api.OLS(y,X).fit()
            y_predict = result.predict(X)
            rmse = stats_api.tools.eval_measures.rmse(y,y_predict)
            print ('category: ', category)
            print ('rmse: ' ,rmse)

            print (result.summary())
            print ('=============================')


def Q1_3():
    hashtags = ['#gohawks', '#nfl', '#sb49', '#gopatriots', '#patriots', '#superbowl']
    for tag in hashtags:
        X = load_obj(tag+'_Q13')[:-1,:]
        y = load_obj(tag+'_numTweetsInHour')[1:]
        model = stats_api.OLS(y,X)
        res = model.fit()
        y_pred = res.predict(X)
        y_resid = y-y_pred
        sum_err = pow(y_resid,2)
        sum_err = np.sum(sum_err)
        print(res.summary())
    #     print(sum_err)
        rmse = sqrt(sum_err/len(y_resid))
        print('%s has RMSE of %.3f' % (tag, rmse))

        features = ['mentionCount','rankScore', 'passitivity', 
                'co-occurrence_of_tags', 'unique_author']
        for i in [0,2,3]:
            x_plt = X[:, i]
            ys = [[y, 'Predictant']]
            x_label = features[i]
            y_label = 'number of tweets for next hour'
            title = tag+', '+x_label
            make_plot(x_plt, ys, scatter=True, xlabel=x_label,
                     ylabel=y_label, title=title)
        print ('=============================')

    # after reading the p value of 5 features and majority votes,
    # we find that x1 (mentionCount), x3 (passitivity) and x4 
    # (tags) are most significant features


def Q1_4():
    hashtags = ['#gohawks', '#nfl', '#sb49', '#gopatriots', '#patriots', '#superbowl']

    rfr = RandomForestRegressor(criterion='mae')
    rbf = SVR(kernel='rbf', gamma=0.1, C=1)
    lsvr = LinearSVR()

    P1 = np.array([0.0, 0.0, 0.0])
    P2 = np.array([0.0, 0.0, 0.0])
    P3 = np.array([0.0, 0.0, 0.0])

    for tag in hashtags:
        with open(fileLocation(tag), encoding="utf8") as f:
            firstTs = FIRST_TS[tag] // 3600 * 3600
            idx1 = tsDiffHour(firstTs, PERIOD1)
            idx2 = tsDiffHour(firstTs, PERIOD2) + 1
            X = load_obj(tag + '_Q13')[:-1, :]
            y = load_obj(tag + '_numTweetsInHour')[1:]

            X1, X2, X3 = X[0:idx1, :], X[idx1:idx2, :], X[idx2:, :]
            y1, y2, y3 = y[0:idx1], y[idx1:idx2], y[idx2:]

            score1_1 = cross_val2(rbf, X=X1, y=y1)
            score1_2 = cross_val2(rbf, X=X2, y=y2)
            score1_3 = cross_val2(rbf, X=X3, y=y3)
            print('\\'+tag, '&', 'Random Forest Regressor', '&', '%.3f' % np.mean(score1_1), '&', '%.3f' % np.mean(score1_2), '&', '%.3f' % np.mean(score1_3), '\\\\')
            print('\\hline')
            score2_1 = cross_val2(rfr, X=X1, y=y1)
            score2_2 = cross_val2(rfr, X=X2, y=y2)
            score2_3 = cross_val2(rfr, X=X3, y=y3)
            print('\\'+tag, '&', 'Support Vector Regressor', '&', '%.3f' % np.mean(score2_1), '&', '%.3f' % np.mean(score2_2), '&', '%.3f' % np.mean(score2_3), '\\\\')
            print('\\hline')
            score3_1 = cross_val2(lsvr, X=X1, y=y1)
            score3_2 = cross_val2(lsvr, X=X2, y=y2)
            score3_3 = cross_val2(lsvr, X=X3, y=y3)
            print('\\'+tag,'&','Linear SVR', '&','%.3f' % np.mean(score3_1),'&', '%.3f' % np.mean(score3_2),'&', '%.3f' % np.mean(score3_3), '\\\\')
            print('\\hline')

            P1 += np.array([np.mean(score1_1), np.mean(score1_2), np.mean(score1_3)])
            P2 += np.array([np.mean(score2_1), np.mean(score2_2), np.mean(score2_3)])
            P3 += np.array([np.mean(score3_1), np.mean(score3_2), np.mean(score3_3)])

    print(P1, P2, P3)

def Q1_4_1():
    hashtags = ['#gohawks', '#nfl', '#sb49', '#gopatriots', '#patriots', '#superbowl']

    rfr = RandomForestRegressor(criterion='mae')
    rbf = SVR(kernel='rbf', gamma=0.1, C=1)
    lsvr = LinearSVR()

    tag = '#gohawks'
    with open(fileLocation(tag), encoding="utf8") as f:
        firstTs = FIRST_TS[tag] // 3600 * 3600
        idx1 = tsDiffHour(firstTs, PERIOD1)
        idx2 = tsDiffHour(firstTs, PERIOD2) + 1
        X = load_obj(tag + '_Q13')[:-1, :]
        y = load_obj(tag + '_numTweetsInHour')[1:]

        X1, X2, X3 = X[0:idx1, :], X[idx1:idx2, :], X[idx2:570, :]
        y1, y2, y3 = y[0:idx1], y[idx1:idx2], y[idx2:570]


    X1.fill(0.0)
    X2.fill(0.0)
    X3.fill(0.0)
    y1 = np.array([0.0] * len(y1))
    y2 = np.array([0.0] * len(y2))
    y3 = np.array([0.0] * len(y3))

    for tag in hashtags:
        with open(fileLocation('#gohawks'), encoding="utf8") as f:
            firstTs = FIRST_TS['#gohawks'] // 3600 * 3600
            idx1 = tsDiffHour(firstTs, PERIOD1)
            idx2 = tsDiffHour(firstTs, PERIOD2) + 1
            X = load_obj(tag + '_Q13')[:-1, :]
            y = load_obj(tag + '_numTweetsInHour')[1:]

            X1 += X[0:idx1, :]
            X2 += X[idx1:idx2, :]
            X3 += X[idx2:570, :]
            y1 += y[0:idx1]
            y2 += y[idx1:idx2]
            y3 += y[idx2:570]

    rfr = RandomForestRegressor(criterion='mae')
    rbf = SVR(kernel='rbf', gamma=0.1, C=1)
    lsvr = LinearSVR()

    score1_1 = cross_val2(rbf, X=X1, y=y1)
    score1_2 = cross_val2(rbf, X=X2, y=y2)
    score1_3 = cross_val2(rbf, X=X3, y=y3)
    print('\\' + tag, '&', 'Random Forest Regressor', '&', '%.3f' % np.mean(score1_1), '&', '%.3f' % np.mean(score1_2),
          '&', '%.3f' % np.mean(score1_3), '\\\\')
    print('\\hline')
    score2_1 = cross_val2(rfr, X=X1, y=y1)
    score2_2 = cross_val2(rfr, X=X2, y=y2)
    score2_3 = cross_val2(rfr, X=X3, y=y3)
    print('\\' + tag, '&', 'Support Vector Regressor', '&', '%.3f' % np.mean(score2_1), '&', '%.3f' % np.mean(score2_2),
          '&', '%.3f' % np.mean(score2_3), '\\\\')
    print('\\hline')
    score3_1 = cross_val2(lsvr, X=X1, y=y1)
    score3_2 = cross_val2(lsvr, X=X2, y=y2)
    score3_3 = cross_val2(lsvr, X=X3, y=y3)
    print('\\' + tag, '&', 'Linear SVR', '&', '%.3f' % np.mean(score3_1), '&', '%.3f' % np.mean(score3_2), '&',
          '%.3f' % np.mean(score3_3), '\\\\')
    print('\\hline')

def Q2():

    X = load_obj('X_Q2')
    y = load_obj('label_Q2')

    rf = RandomForestClassifier(max_features=50,random_state=20)
    svc = svm.LinearSVC(C=10)
    lr = LogisticRegression(random_state=20)
    knn = KNeighborsClassifier(n_neighbors=3)
    mlp = MLPClassifier(solver='lbfgs', activation="relu", alpha=1e-4,hidden_layer_sizes=(200,400), random_state=1)
    dt = DecisionTreeClassifier(random_state=20)
    clfs = [rf, svc, lr, knn, mlp, dt]
    clf_names = ['rf','svc','lr','knn','mlp','dt']

    for clf, clf_name in zip(clfs, clf_names):
        print(clf_name)
        score = True
        if clf_name == 'svc':
            score = False
        y_t_train, y_p_train, y_t_test, y_p_test, y_score_train, y_score_test \
            = cross_val(clf, X, y, shuffle=True, score=score, verbose=True)

        acc_test, rec_test, prec_test = metrics(y_t_test, y_p_test)
        acc_train, rec_train, prec_train = metrics(y_t_train, y_p_train)
        print('Test accuracy %0.4f, recall score %0.4f and precision score %.4f'
              % (acc_test, rec_test, prec_test))
        print('Train accuracy %0.4f, recall score %0.4f and precision score %.4f'
              % (acc_train, rec_train, prec_train))

        classnames = ['Washington', 'Massachusetts']

        plot_confusion_matrix(y_t_test, y_p_test, classnames)

        plot_ROC(y_t_test, y_score_test, no_score=(not score))

def Q3():
    import nltk
    nltk.download('vader_lexicon')
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sid = SentimentIntensityAnalyzer()

    hashtags = ['#gohawks', '#gopatriots', '#patriots']
    for tag in hashtags: 
        with open(fileLocation(tag), encoding="utf8") as f:
            print ('tag: ',tag)

            firstTs = FIRST_TS[tag]
            firstTs = firstTs // 3600 * 3600
            lastTs = LAST_TS[tag]
            totalHours = tsDiffHour(firstTs, lastTs) + 1

            neg_sentiments = [0] * totalHours
            pos_sentiments = [0] * totalHours
            neu_sentiments = [0] * totalHours
            hourCount = [0] * totalHours

            tweets = f.readlines()
            for tweet in tweets:
                t = json.loads(tweet)
                ts = t['citation_date']
                date = datetime.datetime.fromtimestamp(t['citation_date'])
                title = t ['title']
                hourDiff = tsDiffHour(firstTs, ts)
                hourCount[hourDiff] += 1
                ss = sid.polarity_scores(title)
                h_r = hourCount[hourDiff]
                neg_sentiments[hourDiff] = (neg_sentiments[hourDiff] * (h_r-1) + ss['neg'])/h_r
                pos_sentiments[hourDiff] = (pos_sentiments[hourDiff] * (h_r-1) + ss['pos'])/h_r
                neu_sentiments[hourDiff] = (neu_sentiments[hourDiff] * (h_r-1) + ss['neu'])/h_r

            # plot sentiments vs time (hour from beginning)
            ys = [[neg_sentiments, 'negative sentiment'],
                [pos_sentiments, 'positive sentiment']]
            x_plt = range(0, totalHours)
            x_label = 'time'
            y_label = 'sentiment'
            title = 'sentiment vs time for '+tag
            make_plot(x_plt, ys, scatter=False, xlabel=x_label,
                     ylabel=y_label, title=title)

def Q3_jack():
    hashtags = ['#gohawks','#patriots']
    num_of_positive_hawks = [0] 
    num_of_negative_hawks = [0] 
    num_of_positive_patriots = [0]
    num_of_negative_patriots = [0] 
    totalHoursMin = 999999;

    for category in hashtags: 
        with open(fileLocation(category), encoding="utf8") as f:
            tweets = f.readlines() #read all lines
            firstTs = FIRST_TS[category]
            firstTs = firstTs // 3600 * 3600
            lastTs = LAST_TS[category]
            totalHours = tsDiffHour(firstTs, lastTs) + 1
            if totalHours < totalHoursMin:
                totalHoursMin = totalHours
            hourCount = [0] * totalHours 
            if category == '#gohawks':
                num_of_positive_hawks = [0] * totalHours
                num_of_negative_hawks = [0] * totalHours
            else:
                num_of_positive_patriots = [0] * totalHours
                num_of_negative_patriots = [0] * totalHours               

            #for tweet in tweets:
            for i in range(0,len(tweets)):
                tweet = tweets[i]
                t = json.loads(tweet)
                ts = t['citation_date']
                date = datetime.datetime.fromtimestamp(t['citation_date'])
                title = t ['title']
                hourDiff = tsDiffHour(firstTs, ts)
                hourCount[hourDiff] += 1 # number
                if category=='#gohawks':
                    if sentiment(tweet) is 'positive':
                        num_of_positive_hawks[hourDiff] += 1
                    elif sentiment(tweet) is 'negative':
                        num_of_negative_hawks[hourDiff] += 1
                else:
                    if sentiment(tweet) is 'positive':
                        num_of_positive_patriots[hourDiff] += 1
                    elif sentiment(tweet) is 'negative':
                        num_of_negative_patriots[hourDiff] += 1                 
    result_positive = [0] * totalHours
    result_negative = [0] * totalHours

    #the perspective of hawks
    for i in range (0, totalHoursMin): #totalHours
        result_positive[i] = num_of_positive_hawks[i] + num_of_negative_patriots[i]
        result_negative[i] = num_of_positive_patriots[i] + num_of_negative_hawks[i]




    ys = [[result_positive, 'positive sentiment'],
        [result_negative, 'negative sentiment']]
    x_plt = range(0, totalHours)
    x_label = 'time'
    y_label = 'sentiment'
    title = 'sentiment vs time for hawks'
    make_plot(x_plt, ys, scatter=False, xlabel=x_label,
             ylabel=y_label, title=title)



    s = """for tag in hashtags:

        num_of_positive = 0
        num_of_negative = 0
        num_of_nuetral = 0
        for tweet in somedata[hashtag] #? time scope
            if sentiment(tweet) is 'positive':
                num_of_positive += 1
            elif sentiment(tweet) is 'negative':
                num_of_negative += 1

        result[tag + "_positive"] = num_of_positive
        result[tag + "_negative"] = num_of_negative

    #from gohawks perspective
    total_negatives = result["#gohawks_negative"] + result["#gopatriots_positive"]
    total_positives = result["#gohawks_positive"] + result["#gopatriots_negative"] """


def sentiment(tweet):
    testimonial = TextBlob(tweet)
    if TextBlob(tweet).sentiment.polarity > 0:
        return 'positive'
    elif TextBlob(tweet).sentiment.polarity == 0:
        return 'neutural'
    else:
        return 'negative'








if __name__ == '__main__':
    Q1_4()

