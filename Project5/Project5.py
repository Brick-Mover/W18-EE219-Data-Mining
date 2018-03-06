import json


def fileLocation(category):
    return 'tweet_data/tweets_%s.txt' % category

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

        hourCount = {}
        for tweet in tweets:
            ts = json.loads(tweet)['firstpost_date']
            hrDiff = tsDiffHour(firstTs, ts)
            if hrDiff in hourCount:
                hourCount[hrDiff] += 1
            else:
                hourCount[hrDiff] = 1

        print(hourCount)

if __name__ == '__main__':
    Q1_1('#gohawks')

