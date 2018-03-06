import json
import pickle
import matplotlib.pyplot as plt
import numpy as np



def fileLocation(category):
    return 'tweet_data/tweets_%s.txt' % category


def save_obj(name, obj):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


"""
startTs: the rounded(down) timestamp of the first tweet
endTs: the raw timestamp of a tweet
return: the hour difference between startTs and endTs(0 -> ...)
"""
def tsDiffHour(startTs:int, endTs:int) -> int:
    return (endTs // 3600 * 3600 - startTs) // 3600


def Q1_1(category):
    with open(fileLocation(category), encoding="utf8") as f:
        tweets = f.readlines()
        firstTs = json.loads(tweets[0])['firstpost_date']
        firstTs = firstTs // 3600 * 3600
        lastTs = json.loads(tweets[-1])['firstpost_date']
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


if __name__ == '__main__':
    Q1_1('#patriots')
    #Q1_1_plot('#gohawks')
