import json
import pickle
import re
import numpy as np
from datetime import date
import matplotlib.pyplot as plt
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
import nltk
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import accuracy_score, recall_score, precision_score
import itertools
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import TruncatedSVD

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

class mytokenizer(object):
    def __init__(self):
        self.stemmer = SnowballStemmer("english", ignore_stopwords=True)
        self.tokenizer = RegexpTokenizer(r'\w+')

    def __call__(self, text):
        tokens = re.sub(r'[^A-Za-z]', " ", text)
        tokens = re.sub("[,.-:/()?{}*$#&]"," ",tokens)
        tokens =[word for tk in nltk.sent_tokenize(tokens) for word in nltk.word_tokenize(tk)]
        new_tokens = []
        for token in tokens:
            if re.search('[a-zA-Z]{2,}', token):
                new_tokens.append(token)     
        stems = [self.stemmer.stem(t) for t in new_tokens]
        return stems

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

    TFidf = TfidfVectorizer(analyzer='word',tokenizer=mytokenizer(), 
                            stop_words=ENGLISH_STOP_WORDS, 
                            norm = 'l2', max_df=0.9, min_df=2)
    svd = TruncatedSVD(n_components=50)
    X = svd.fit_transform(TFidf.fit_transform(text_data))

    # save_obj('text_data_Q2', text_data)
    save_obj('label_Q2', loc)
    save_obj('X_Q2', X)



def plot_confusion_matrix(label_true, label_pred, classname, normalize=False, title='Confusion Matrix'):
    plt.figure()
    cmat = confusion_matrix(label_true, label_pred)
    cmap = plt.cm.Blues
    plt.imshow(cmat, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(classname))
    plt.xticks(tick_marks, classname, rotation=45)
    plt.yticks(tick_marks, classname)

    if normalize:
        cmat = cmat.astype('float') / cmat.sum(axis=1)[:, np.newaxis]

    # print(cmat)

    thresh = cmat.max() / 2.
    for i, j in itertools.product(range(cmat.shape[0]), range(cmat.shape[1])):
        if normalize == False:
            plt.text(j, i, cmat[i, j], horizontalalignment="center", color="white" if cmat[i, j] > thresh else "black")
        else:
            plt.text(j, i, "%.2f"%cmat[i, j], horizontalalignment="center", color="white" if cmat[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# 
# if score is true, return y_score for usage in plot of ROC
# ! Only set score to True if the classifier has function predict_proba!!!
# 
def cross_val(clf, X, y, shuffle=False, score=False, verbose=False):
    kf = KFold(n_splits=10, shuffle = shuffle)

    y_true_train = np.array([])
    y_pred_train = np.array([])
    y_true_test = np.array([])
    y_pred_test = np.array([])
    y_score_train = np.array([])
    y_score_test = np.array([])
    X = np.array(X)
    y = np.array(y)
    epoch = 1
    for train_index, test_index in kf.split(X):
        if verbose:
            print('epoch', epoch)
        epoch += 1
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)
        sub_y_pred_test = clf.predict(X_test)
        sub_y_pred_train = clf.predict(X_train)

        y_true_test = np.append(y_true_test, y_test)
        y_true_train = np.append(y_true_train, y_train)
        y_pred_test = np.append(y_pred_test, sub_y_pred_test)
        y_pred_train = np.append(y_pred_train, sub_y_pred_train)

        if score == True:
            sub_y_score_train = clf.predict_proba(X_train)
            sub_y_score_test = clf.predict_proba(X_test)
        else:
            sub_y_score_train = clf.decision_function(X_train)
            sub_y_score_test = clf.decision_function(X_test)
        y_score_train = np.append(y_score_train, sub_y_score_train)
        y_score_test = np.append(y_score_test, sub_y_score_test)

    return y_true_train, y_pred_train, y_true_test, y_pred_test, y_score_train, y_score_test


def cross_val2(clf, X, y, shuffle=False, score=False, verbose=False):
    kf = KFold(n_splits=10, shuffle = shuffle)

    y_true_test = np.array([])
    y_pred_test = np.array([])

    X = np.array(X)
    y = np.array(y)
    epoch = 1
    for train_index, test_index in kf.split(X):
        if verbose:
            print('epoch', epoch)
        epoch += 1
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)
        sub_y_pred_test = clf.predict(X_test)

        y_true_test = np.append(y_true_test, y_test)
        y_pred_test = np.append(y_pred_test, sub_y_pred_test)

    return np.mean(np.abs(y_pred_test - y_true_test))


# 
# calculate accuracy score and f1 score
# 
def metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    return acc, rec, prec

# 
# return area under curve
# 
def plot_ROC(yTrue, yScore, title='ROC Curve', rang=4, no_score=False):
    # pay attention here
    if not no_score:
        yScore = yScore.reshape(len(yTrue), 2)
        yScore = yScore[:,1]
    fpr, tpr, _ = roc_curve(yTrue, yScore)
    roc_auc = auc(fpr, tpr)
    lw=2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()
    plt.close()

    return roc_auc

TS_test = np.array([
[1422554405, 1422575945],
[1422817200, 1422838799],
[1422874802, 1422896399],
[1422223204, 1422244781],
[1422406820, 1422428389],
[1422810001, 1422831599],
[1422943203, 1422964632],
[1422489605, 1422507351],
[1422813600, 1422835199],
[1423166443, 1423187958],
])

TS_train =np.array([
    [1419804875,1423304269],
    [1419999683,1423335336],
    [1421238675,1423335336],
    [1420835445,1423295675],
    [1419805279,1423335300],
    [1419866833,1423332008]
])

def createTrainDataQ1_5():
    hashtags = ['#gohawks', '#nfl', '#sb49', '#gopatriots', '#patriots', '#superbowl']
    dim = 575
    X_tr = np.zeros([dim,5])
    y_tr = np.zeros(dim)
    for tag in hashtags:
        X = load_obj(tag+'_Q13')[:dim,:]
        y = load_obj(tag+'_numTweetsInHour')[:dim]
        X_tr = X_tr + X
        y_tr = y_tr + y
        print(np.amax(y))
        print(X.shape, len(y))
    X_tr = np.insert(X_tr, 0, y_tr, axis = 1)
    X_train = X_tr[:570]+X_tr[1:571]+X_tr[2:572]+X_tr[3:573]+X_tr[4:574]
    y_train = y_tr[5:]
    save_obj('Q1_5XTrain',X_train)
    save_obj('Q1_5yTrain',y_train)

def hour(ft, lt):
    return math.ceil((lt-ft)/3600)

def createTestDataQ1_5():
    files = ['sample1_period1.txt', 'sample2_period2.txt', 'sample3_period3.txt',
            'sample4_period1.txt', 'sample5_period1.txt', 'sample6_period2.txt',
            'sample7_period3.txt', 'sample8_period1.txt', 'sample9_period2.txt',
            'sample10_period3.txt']
    for file,ind in zip(files, range(10)):
        file_name = 'test_data/'+file
        with open(file_name, encoding="utf8") as f:

            tweets = f.readlines()
            firstTs = TS_test[ind][0]
            lastTs = TS_test[ind][1]
            totalHours = hour(firstTs, lastTs)
            print(file[:7], totalHours)

            tweetCount = [0] * totalHours
            mentionCount = [0] * totalHours
            rankScore = [0] * totalHours
            passitivity = [0] * totalHours
            tags = [0] * totalHours
            author = [0] * totalHours
            uniq_author = {}

            cnt = 0
            for tweet in tweets:
                cnt += 1
                t = json.loads(tweet)
                ts = t['firstpost_date']
                hourDiff = hour(firstTs, ts)-1

                tweetCount[hourDiff] += 1
                mentionCount[hourDiff] += get_feature(t, 'mention')
                rankScore[hourDiff] += get_feature(t, 'rank_score')
                passitivity[hourDiff] += get_feature(t, 'passitivity')
                tags[hourDiff] += get_feature(t, 'tags')
                aut = get_feature(t, 'author')
                if aut not in uniq_author:
                    uniq_author[aut] = len(uniq_author)
                    author[hourDiff] += 1

            X = np.array([tweetCount, mentionCount, rankScore, passitivity, tags, author])
            X = X.transpose()
            y = np.array(tweetCount)
            X_test = np.sum(X[:5,:],axis=0)
            print(y)
            if file[:7] == 'sample8':
                y_test=y[4:]
            else:
                y_test = y[5:]
            save_obj(file[:6]+str(ind+1)+'X',X_test)
            save_obj(file[:6]+str(ind+1)+'y',y_test)
            print(X.shape, y.shape)
            print(X_test.shape, y_test.shape)

X_prevHour = np.array([
    [171.,109.,680.3551283,821.60117786,0.,0.],
    [81973.,20644.,370865.6633202,950335.82445006,0.,0.],
    [616.,312.,2876.7586997,5826.8334464,0.,0.],
    [267.,103.,1224.7706613,2848.39517932,0.,0.],
    [282.,142.,1253.6308908,3130.1769946,0.,0.,],
    [41014.,72836.,170236.48472889,335943.93341304,0.,0.],
    [55.,14.,240.6450064,352.58880725,0.,0.],
    [41.,7.,164.630682,1194.36608456,0.,0.],
    [1857.,288.,7993.1228076,22272.8003312,0.,0.],
    [58.,4.,229.6415427,1054.4130926,0.,0.]
])