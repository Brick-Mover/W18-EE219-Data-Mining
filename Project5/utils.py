import json
import pickle
import re
import numpy as np
from datetime import date
import matplotlib.pyplot as plt

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


def make_plot(x, ys, scatter=False, bar=False, xlabel=None, ylabel=None, 
              xticks=None, grid=False, title=None, 
              size_marker = 20, marker = '.'):
    for y, label in ys:
        if scatter:
            plt.scatter(x, y, s=size_marker, marker=marker, label=label)
        elif bar:
            plt.bar(x, y, label=label, color='g', width=1)
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

def match(l):
    if (re.match('.*WA.*', l) or re.match('.*Wash.*', l)):
        return 0
    if (re.match('.*MA.*', l) or re.match('.*Mass.*', l)):
        return 1
    return -1

def createQ2Data():
    loc = np.array([]) #y
    text_data = np.array([]) #X
    with open(fileLocation(tag), encoding="utf8") as f:
        tweets = f.readlines()

        count = 0 
        count_loc = 0
        for tweet in tweets:
    #         if count%10000 == 0:
    #             print('count ',count)
            count+=1
            t = json.loads(tweet)
            location = t['tweet']['user']['location']
            mat_res = match(location)
            if mat_res != -1:
    #             if count_loc%1000 == 0:
    #                 print('count_loc ',count_loc)
                count_loc += 1
                text = t['tweet']['text']
                loc = np.append(loc, mat_res)
                text_data = np.append(text_data, text)

    save_obj('text_data_Q2', text_data)
    save_obj('label_Q2', loc)