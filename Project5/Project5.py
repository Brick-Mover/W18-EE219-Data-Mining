import json
import pickle
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as stats_api 
from sklearn.svm import SVR
from datetime import date


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
        users = set()
        for tweet in tweets:
            t = json.loads(tweet)
            ts = t['firstpost_date']
            # count hour
            hourDiff = tsDiffHour(firstTs, ts)
            hourCount[hourDiff] += 1
            # count follower
            if t['tweet']['user']['id'] not in users:
                users.add(t['tweet']['user']['id'])
                followerCount += t['author']['followers']
            # count retweets
            retweetCount += t['metrics']['citations']['total']


        save_obj(category + '_numTweetsInHour', hourCount)
        # report average number of tweets per hour
        print(category + ': ' + 'Average number of tweets per hour: ' + str(np.mean(hourCount)))
        print(category + ': ' + 'Average number of followers of users posting the tweets: ' +
              str(followerCount / len(users)))
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
            firstTs = json.loads(tweets[0])['firstpost_date']
            firstTs = firstTs // 3600 * 3600
            lastTs = json.loads(tweets[-1])['firstpost_date']
            # total hour 
            totalHours = tsDiffHour(firstTs, lastTs) + 1

            hourCount = [0] * totalHours
            followerCount = [0] * totalHours
            retweetCount = [0] * totalHours
            tweetCount = [0] * totalHours
            max_followers = [0] * totalHours

            users = set()
            for tweet in tweets:
                t = json.loads(tweet)
                ts = t['firstpost_date']
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
# and 'tags' (for sum)
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

def Q1_3():
    return 0

def Q1_4():
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr_lin = SVR(kernel='linear', C=1e3)
    svr_poly = SVR(kernel='poly', C=1e3, degree=2)
    y_rbf = svr_rbf.fit(X, y).predict(X)
    y_lin = svr_lin.fit(X, y).predict(X)
    y_poly = svr_poly.fit(X, y).predict(X)




if __name__ == '__main__':
    # extractFirstTsAndLastTs()
    Q1_3()
