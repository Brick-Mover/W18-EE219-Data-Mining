def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings("ignore", category=DeprecationWarning)

import json
import pickle
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as stats_api 
from sklearn.svm import SVR
from datetime import date
from math import sqrt

FIRST_TS = {
    "#gohawks": 1421222681,
    "#nfl": 1421222404,
    "#sb49": 1421238675,
    "#gopatriots": 1421229011,
    "#patriots": 1421222838,
    "#superbowl": 1421223187
}

LAST_TS = {
    "#gohawks": 1423304269,
    "#nfl": 1423335336,
    "#sb49": 1423335336,
    "#gopatriots": 1423295675,
    "#patriots": 1423335300,
    "#superbowl": 1423332008
}


def fileLocation(category):
    return 'tweet_data/tweets_%s.txt' % category


def save_obj(name, obj):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def tsDiffHour(startTs:int, endTs:int) -> int:
    """
    startTs: the rounded(down) timestamp of the first tweet
    endTs: the raw timestamp of a tweet
    :return: the hour difference between startTs and endTs(0 -> ...)
    """
    return (endTs // 3600 * 3600 - startTs) // 3600


def extractFirstTsAndLastTs():
    """
    This function is only called once to extract the first timestamp of each category.
    Included for the completeness of code submission.
    """
    hashtags = ['#gohawks', '#nfl', '#sb49', '#gopatriots', '#patriots', '#superbowl']
    for category in hashtags:
        with open(fileLocation(category), encoding="utf8") as f:
            tweets = f.readlines()
            firstTs = json.loads(tweets[0])['citation_date']
            lastTs = firstTs
            for t in tweets:
                t = json.loads(t)
                if t['citation_date'] < firstTs:
                    firstTs = t['citation_date']
                if t['citation_date'] > lastTs:
                    lastTs = t['citation_date']
            print(category, firstTs, lastTs)


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
    plt.bar(hours, hourCount, label="Number of tweets per hour", color='g')
    plt.xlabel('hours')
    plt.ylabel('Number of tweets')
    plt.show()


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

# 
# for function days_of_account
# 
month_num = {
    'Jan': 1,
    'Feb': 2,
    'Mar': 3,
    'Apr': 4,
    'May': 5,
    'Jun': 6,
    'Jul': 7,
    'Aug': 8,
    'Sep': 9,
    'Oct': 10,
    'Nov': 11,
    'Dec': 12
}

def days_of_account(t):
    account_date = t['tweet']['user']['created_at'].split(' ')
    post_date = t['tweet']['created_at'].split(' ')
    d_account = date(int(account_date[5]), month_num[account_date[1]],
                     int(account_date[2]))
    d_post = date(int(post_date[5]), month_num[post_date[1]],
                     int(post_date[2]))
    return (d_post-d_account).days

# 
# feat = 'retweet', 'follower', 'mention' (for mention count sum), 
# 'rank_score', 'passitivity' (rareness, as defined in report, for sum),
# and 'tags' (for sum), 'author' (for unique author count)
# 
def get_feature(tweet, feat):
    if feat == 'retweet':
        return tweet['metrics']['citations']['total']
    elif feat == 'follower':
        return tweet['author']['followers']
    elif feat == 'mention':
        return len(tweet['tweet']['entities']['user_mentions'])
    elif feat == 'rank_score':
        return tweet['metrics']['ranking_score']
    elif feat == 'passitivity':
        days_account = days_of_account(tweet)
        followers = tweet['tweet']['user']['followers_count']
        res = days_account/(1.0+followers)
        return res
    elif feat == 'tags':
        res = len(tweet['tweet']['entities']['hashtags'])
        return res
    elif feat == 'author':
        return tweet['author']['name']

# 
# create X (new features) for Q1_3
# 
def createData():
    hashtags = ['#gohawks', '#nfl', '#sb49', '#gopatriots', '#patriots', '#superbowl']
    for tag in hashtags:
        with open(fileLocation(tag), encoding="utf8") as f:
            tweets = f.readlines()
            firstTs = FIRST_TS[tag]
            firstTs = firstTs // 3600 * 3600
            lastTs = LAST_TS[tag]
            totalHours = tsDiffHour(firstTs, lastTs) + 1

            mentionCount = [0] * totalHours
            rankScore = [0] * totalHours
            passitivity = [0] * totalHours
            tags = [0] * totalHours
            author = [0] * totalHours
            uniq_author = {}

            for tweet in tweets:
                t = json.loads(tweet)
                ts = t['citation_date']
                hourDiff = tsDiffHour(firstTs, ts)
                
                mentionCount[hourDiff] += get_feature(t, 'mention')
                rankScore[hourDiff] += get_feature(t, 'rank_score')
                passitivity[hourDiff] += get_feature(t, 'passitivity')
                tags[hourDiff] += get_feature(t, 'tags')
                aut = get_feature(t, 'author')
                if aut not in uniq_author:
                    uniq_author[aut] = len(uniq_author)
                    author[hourDiff] += 1
            
            X = np.array([mentionCount, rankScore, passitivity, tags, author])
            X = X.transpose()
            save_obj(tag + '_Q13', X)

def make_plot(x, ys, scatter=False, xlabel=None, ylabel=None, 
              xticks=None, grid=False, title=None, 
              size_marker = 20, marker = '.'):
    for y, label in ys:
        if scatter:
            plt.scatter(x, y, s=size_marker, marker=marker, label=label)
        else:
            plt.plot(x, y, label=label)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if xticks is not None:
        plt.xticks(x)
    plt.legend()
    if grid == True:
        plt.grid()
    if title is not None:
        plt.title(title)
    plt.show() 

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
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr_lin = SVR(kernel='linear', C=1e3)
    svr_poly = SVR(kernel='poly', C=1e3, degree=2)
    y_rbf = svr_rbf.fit(X, y).predict(X)
    y_lin = svr_lin.fit(X, y).predict(X)
    y_poly = svr_poly.fit(X, y).predict(X)




if __name__ == '__main__':
    # hashtags = ['#gohawks', '#sb49', '#gopatriots', '#patriots', "#superbowl", "#nfl"]
    # for cate in hashtags:
    #     Q1_1(cate)
    # Q1_1("#superbowl")
    # Q1_1_plot("#superbowl")
    # Q1_1("#nfl")
    # Q1_1_plot("#nfl")
    # Q1_2()
    Q1_3()
