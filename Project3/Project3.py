import numpy as np
import pandas as pd
import time, datetime
import matplotlib.pyplot as plt
from surprise.prediction_algorithms.knns import KNNWithMeans
from surprise.model_selection import cross_validate
from surprise import Dataset, Reader
from surprise.model_selection import KFold


#
# !!!!!!!!!!!!!!!!!!!!!!!!!
#
# Jupyter Notebook for a better view only, you can edit/run code
# on your local Jupyter Notebook, but please push/sync code through
# this Project3.py only!!!
#
# (You can upload your own Jupyter Notebook if you want, just name
#  as Project3_yourname.ipynb)
#
# This is because we have not found a tool (like ShareLatex) to
# share/edit/run upyter Notebook simultaneously. Although
# Jupyter Notebook is really intuitive and convenient to use
#
# !!!!!!!!!!!!!!!!!!!!!!!!!
#

#
# Create R matrix
#
data = np.loadtxt('ml-latest-small/ratings.csv',
                  delimiter=',', skiprows=1, usecols=(0,1,2))

# tranform data type (from float to int for first 2 rows)
# 'userId', 'movieId', 'rating'
row_userId = data[:,:1].astype(int)
row_movieId = data[:,1:2].astype(int)
row_rating = data[:,2:3]
R_row = np.amax(row_userId)
R_col = np.amax(row_movieId)
print('Matrix has row size (users) %s, and col size (movies) %s'
      % (R_row, R_col))
R = np.zeros([R_row, R_col])
for i in range(row_userId.size):
    r = row_userId[i] - 1
    c = row_movieId[i] - 1
    rating = row_rating[i]
    R[r, c] = rating

assert(R[1,109]==4.0)


# map movie ids to remove nonexistent movieId
sortedId = np.sort(row_movieId.transpose()[0])
m = {}
idx = 0
last = None
for i in sortedId.tolist():
    if i != last:
        m[i] = idx
        idx += 1
    last = i
mapped_row_movieId = np.copy(row_movieId)
for r in mapped_row_movieId:
    r[0] = m[r[0]]


#
# Question 1
#

# Sparsity = Total number of available ratings
#           / Total number of possible ratings

# Currentyly, row of R is 671 which corresponds to the 671 users listed
# in the dataset README file.
# However, column of R is 163949 which does not correspond to the 9125
# movies listed in dataset README file. As a result, the sparsity if very
# low.
# This is because the max movieId is 163949 from the rating.csv file. We
# will see if this is the correct choice later. If not, we need to find
# a way to map 9000+ movies (with movieIDs max to 163949) from 163949
# columns to 9000+ columns.

rating_avl = np.count_nonzero(R)
rating_psb = np.prod(R.shape)
sparsity = rating_avl/rating_psb
print(sparsity)

# Question 2

# plot a historgram showing frequency of rating values
# ratings_arr = []
# for r in range(R_row):
#     for c in range(R_col):
#         if R[r,c]!=0.0:
#             ratings_arr.append(R[r,c])
# binwidth = 0.5
# print (min(ratings_arr))
# print (max(ratings_arr))
#
# plt.hist(ratings_arr, bins=np.arange(min(ratings_arr), max(ratings_arr) + binwidth, binwidth))
# plt.show()

def make_plot(x, ys, xlabel, ylabel, xticks=None, grid=False, title=None):
    for y, label in ys:
        plt.plot(x, y, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if xticks is not None:
        plt.xticks(x)
    plt.legend()
    if grid == True:
        plt.grid()
    if title is not None:
        plt.title(title)
    plt.show()

def load_data():
    ratings_dict = {
        'movieID': mapped_row_movieId.transpose().tolist()[0],
        'userID': row_userId.transpose().tolist()[0],
        'rating': (row_rating.transpose()*2).tolist()[0]    # map (0.5, 1, ..., 5) to (1, 2, ..., 10)
    }
    df = pd.DataFrame(ratings_dict)
    reader = Reader(rating_scale=(1, 10))
    data = Dataset.load_from_df(df[['userID', 'movieID', 'rating']], reader)
    return data

def Q10():
    data = load_data()

    sim_options = {'name': 'pearson_baseline',
                   'shrinkage': 0  # no shrinkage
                 }

    meanRMSE, meanMAE = [], []
    start = time.time()
    for k in range(2, 102, 2):
        knnWithMeans = KNNWithMeans(k, sim_options=sim_options)
        out = cross_validate(knnWithMeans, data, measures=['RMSE', 'MAE'], cv=10)
        meanRMSE.append(np.mean(out['test_rmse']))
        meanMAE.append(np.mean(out['test_mae']))
    cv_time = str(datetime.timedelta(seconds=int(time.time() - start)))
    print("Total time used for cross validation: " + cv_time)

    k = list(range(2, 102, 2))
    ys = [[meanRMSE, 'mean RMSE'], [meanMAE, 'mean MAE']]
    make_plot(k, ys, 'Number of Neighbors', 'Error')
    return meanRMSE, meanMAE

def popularTrim(testSet):
    colCnt = {}
    for (_, c, _) in testSet:
        if c in colCnt.keys():
            colCnt[c] += 1
        else:
            colCnt[c] = 1
    result = []
    for (r, c, rating) in testSet:
        if colCnt[c] > 2:
            result.append((r, c, rating))
    return result

def unpopularTrim(testSet):
    colCnt = {}
    for (_, c, _) in testSet:
        if c in colCnt.keys():
            colCnt[c] += 1
        else:
            colCnt[c] = 1
    result = []
    for (r, c, rating) in testSet:
        if colCnt[c] <= 2:
            result.append((r, c, rating))
    return result

def highVarTrim(testSet):
    colCnt = {}
    for (_, c, rating) in testSet:
        if c in colCnt.keys():
            colCnt[c].append(rating)
        else:
            colCnt[c] = []
    result = []
    for (r, c, rating) in testSet:
        if len(colCnt[c]) > 5 and np.var(np.array(colCnt[c])) > 2:
            result.append((r, c, rating))
    return result


def Q12To14(qNum, n_splits=10):
    data = load_data()
    kf = KFold(n_splits=10)

    sim_options = {'name': 'pearson_baseline',
                   'shrinkage': 0  # no shrinkage
                 }
    trimFun = {12: popularTrim,
               13: unpopularTrim,
               14: highVarTrim}
    RMSE = []
    for k in range(2, 10, 2):
        knnWithMeans = KNNWithMeans(k, sim_options=sim_options)
        subRMSE = []
        for trainSet, testSet in kf.split(data):
            subsubRMSE = 0
            knnWithMeans.fit(trainSet)
            testSet = trimFun[qNum](testSet)
            nTest = len(testSet)
            print("test set size after trimming: %d", nTest)
            for (r, c, rating) in testSet:
                predictedRating = knnWithMeans.predict(str(r), str(c))
                subsubRMSE += (pow(rating - predictedRating.est, 2))
            # calculate RMSE of this train-test split
            subRMSE.append(np.sqrt(subsubRMSE / nTest))
        # average of all train-test splits of k-NN
        RMSE.append(np.mean(subRMSE))
    return RMSE

if __name__ == '__main__':
    RMSE12 = Q12To14(12)
    # Q12To14(13)
    # Q12To14(14)