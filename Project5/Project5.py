def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings("ignore", category=DeprecationWarning)

import json
import statsmodels.api as stats_api 
from sklearn.svm import SVR
import numpy as np
from math import sqrt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestClassifier
from utils import fileLocation, save_obj, load_obj, tsDiffHour, extractFirstTsAndLastTs, \
get_feature, createData, make_plot, createQ2Data, plot_confusion_matrix, cross_val, \
metrics, plot_ROC, FIRST_TS, LAST_TS
from sklearn.ensemble import RandomForestRegressor


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
    hashtags = ['#gohawks']

    def score_func(y_pred, y):
        return np.mean(abs(y_pred - y))

    scorer = make_scorer(score_func, greater_is_better=False)

    for tag in hashtags:
        with open(fileLocation(tag), encoding="utf8") as f:
            firstTs = FIRST_TS[tag] // 3600 * 3600
            idx1 = tsDiffHour(firstTs, PERIOD1)
            idx2 = tsDiffHour(firstTs, PERIOD2) + 1
            X = load_obj(tag + '_Q13')[:-1, :]
            y = load_obj(tag + '_numTweetsInHour')[1:]

            X1, X2, X3 = X[0:idx1, :], X[idx1:idx2, :], X[idx2:, :]
            y1, y2, y3 = y[0:idx1], y[idx1:idx2], y[idx2:]

            rfr = RandomForestRegressor(criterion='mae')
            score1_1 = cross_val_score(rfr, X=X1, y=y1, scoring=scorer, cv=10)
            score1_2 = cross_val_score(rfr, X=X2, y=y2, scoring=scorer, cv=10)
            score1_3 = cross_val_score(rfr, X=X3, y=y3, scoring=scorer, cv=10)
            print(np.mean(score1_1), np.mean(score1_2), np.mean(score1_3), '\n')

            # svr_rbf = SVR(kernel='rbf', gamma=0.1, C=0.1)
            # svr_lin = SVR(kernel='linear', C=0.1)
            # svr_poly = SVR(kernel='poly', degree=2, C=1, cache_size=7000)

            # score1_1 = cross_val_score(svr_rbf, X=X1, y=y1, scoring=scorer, cv=10)
            # score1_2 = cross_val_score(svr_rbf, X=X2, y=y2, scoring=scorer, cv=10)
            # score1_3 = cross_val_score(svr_rbf, X=X3, y=y3, scoring=scorer, cv=10)

            # score2_1 = cross_val_score(svr_lin, X=X1, y=y1, scoring=scorer, cv=10)
            # score2_2 = cross_val_score(svr_lin, X=X2, y=y2, scoring=scorer, cv=10)
            # score2_3 = cross_val_score(svr_lin, X=X3, y=y3, scoring=scorer, cv=10)


            # print(np.mean(score1_1), np.mean(score1_2), np.mean(score1_3), '\n')
            # print(np.mean(score2_1), np.mean(score2_2), np.mean(score2_3), '\n')
            # print(np.mean(score3_1), np.mean(score3_2), np.mean(score3_3), '\n')
            #
            # print(score1_1, np.mean(score1_1))
            # print(score1_2, np.mean(score1_2))
            # print(score1_3, np.mean(score1_3))
            #
            # print(score2_1, np.mean(score2_1))
            # print(score2_2, np.mean(score2_2))
            # print(score2_3, np.mean(score2_3))
            #
            # print(score3_1, np.mean(score3_1))
            # print(score3_2, np.mean(score3_2))
            # print(score3_3, np.mean(score3_3))


def Q2():
    X = load_obj('X_Q2')
    y = load_obj('label_Q2')

    clf = RandomForestClassifier(max_features=50,random_state=20)

    y_t_train, y_p_train, y_t_test, y_p_test, y_score_train, y_score_test \
        = cross_val(clf, X, y, shuffle=True, score=True, verbose=True)

    acc_test, rec_test, prec_test = metrics(y_t_test, y_p_test)
    acc_train, rec_train, prec_train = metrics(y_t_train, y_p_train)
    print('Test accuracy %0.4f, recall score %0.4f and precision score %.4f'
          % (acc_test, rec_test, prec_test))
    print('Train accuracy %0.4f, recall score %0.4f and precision score %.4f'
          % (acc_train, rec_train, prec_train))

    classnames = ['Washington', 'Massachusetts']

    plot_confusion_matrix(y_t_test, y_p_test, classnames)

    auc = plot_ROC(y_t_test, y_score_test)

if __name__ == '__main__':
    Q1_2()

