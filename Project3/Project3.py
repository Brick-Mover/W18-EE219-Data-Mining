import numpy as np
import pandas as pd
import time, datetime
import matplotlib.pyplot as plt
from surprise.prediction_algorithms.knns import KNNWithMeans
from surprise.prediction_algorithms.matrix_factorization import SVD, NMF
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


"""
This function should not be called during multi-threading!!
"""
def saveDfToPickle():
    data = np.loadtxt('ml-latest-small/ratings.csv',
                      delimiter=',', skiprows=1, usecols=(0, 1, 2))

    # tranform data type (from float to int for first 2 rows)
    # 'userId', 'movieId', 'rating'
    row_userId = data[:, :1].astype(int)
    row_movieId = data[:, 1:2].astype(int)
    row_rating = data[:, 2:3]
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

    ratings_dict = {
        'movieID': mapped_row_movieId.transpose().tolist()[0],
        'userID': row_userId.transpose().tolist()[0],
        'rating': (row_rating.transpose()*2).tolist()[0]    # map (0.5, 1, ..., 5) to (1, 2, ..., 10)
    }
    df = pd.DataFrame(ratings_dict)
    df.to_pickle('df.pkl')

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

def Q1():
    data = np.loadtxt('ml-latest-small/ratings.csv',
                      delimiter=',', skiprows=1, usecols=(0, 1, 2))

    # tranform data type (from float to int for first 2 rows)
    # 'userId', 'movieId', 'rating'
    row_userId = data[:, :1].astype(int)
    row_movieId = data[:, 1:2].astype(int)
    row_rating = data[:, 2:3]
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
    df = pd.read_pickle('df.pkl')
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

def popularTrim(testSet, pop):
    return list(filter(lambda x: x[1] in pop, testSet))

def unpopularTrim(testSet, unpop):
    return list(filter(lambda x: x[1] in unpop, testSet))

def highVarTrim(testSet, highVar):
    return list(filter(lambda x: x[1] in highVar, testSet))

"""
Return 3 sets: popular, unpopular and highVar
"""
def classifyMovies():
    df = pd.read_pickle('df.pkl')
    pop, unpop, highVar = set(), set(), set()
    colCnt = {}
    for m in df.itertuples():
        if m.movieID in colCnt.keys():
            colCnt[m.movieID].append(m.rating)
        else:
            colCnt[m.movieID] = []
    for m in df.itertuples():
        if len(colCnt[m.movieID]) <= 2:
            unpop.add(m.movieID)
        if len(colCnt[m.movieID]) > 2:
            pop.add(m.movieID)
        if len(colCnt[m.movieID]) > 5 and np.var(colCnt[m.movieID]) > 2.0:
            highVar.add(m.movieID)
    return pop, unpop, highVar

def Q12To14And19To21And26To28(qNum, maxk=None):
    data = load_data()
    kf = KFold(n_splits=10)
    if maxk is None:
        if 12 <= qNum <= 14:
            maxk = 100
        elif 19 <= qNum <= 21:
            maxk = 50
        elif 26 <= qNum <= 28:
            maxk = 50

    pop, unpop, highVar = classifyMovies()

    sim_options = {
        'name': 'pearson_baseline',
        'shrinkage': 0  # no shrinkage
    }
    trimAndModel = {
        12: (pop, 'KNNWithMeans'),
        13: (unpop, 'KNNWithMeans'),
        14: (highVar, 'KNNWithMeans'),
        19: (pop, 'NMF'),
        20: (unpop, 'NMF'),
        21: (highVar, 'NMF'),
        26: (pop, 'SVD'),
        27: (unpop, 'SVD'),
        28: (highVar, 'SVD')
    }

    RMSE = []   #  RMSE for each k
    for k in range(2, maxk + 1, 2): # inclusive
        print('-' * 20 + ' k = ' + str(k) + ' ' + '-' * 20)
        trimSet, modelName = trimAndModel[qNum]
        if modelName == 'KNNWithMeans':
            model = KNNWithMeans(k, sim_options=sim_options)
        elif modelName == 'NMF':
            model = NMF()
        else:
            model = SVD(n_factors = k)
        subRMSE = []    # RMSE for each k for each train-test split
        iter = 1
        for trainSet, testSet in kf.split(data):
            subsubRMSE = 0
            model.fit(trainSet)
            testSet = list(filter(lambda x: x[1] in trimSet, testSet))
            nTest = len(testSet)
            print("Split " + str(iter) + ": test set size after trimming: %d", nTest)
            iter += 1
            predictions = model.test(testSet)
            for p in predictions:
                subsubRMSE += pow(p.est - p.r_ui, 2)
            # calculate RMSE of this train-test split
            subRMSE.append(np.sqrt(subsubRMSE / nTest))
        # average of all train-test splits of k-NN for this k
        RMSE.append(np.mean(subRMSE))

    # plotting
    k = list(range(2, maxk+1, 2))
    ys = [[RMSE, 'RMSE']]
    make_plot(k, ys, 'Number of Neighbors', 'Error')
    return RMSE


def Q17():
    data = load_data()

    meanRMSE, meanMAE = [], []
    start = time.time()
    for k in range(2, 52, 2):
        print(k)
        nmf = NMF()
        out = cross_validate(nmf, data, measures=['RMSE', 'MAE'], cv=10)
        meanRMSE.append(np.mean(out['test_rmse']))
        meanMAE.append(np.mean(out['test_mae']))
    cv_time = str(datetime.timedelta(seconds=int(time.time() - start)))
    print("Total time used for cross validation: " + cv_time)

    k = list(range(2, 52, 2))
    ys = [[meanRMSE, 'mean RMSE'], [meanMAE, 'mean MAE']]
    make_plot(k, ys, 'Number of Neighbors', 'Error')
    return meanRMSE, meanMAE


def Q19to21(qNum):
    data = load_data()
    kf = KFold(n_splits=10)

    trimFun = {12: popularTrim,
               13: unpopularTrim,
               14: highVarTrim}
    RMSE = []
    for k in range(2, 20, 2):
        nmf = NMF()
        subRMSE = []
        for trainSet, testSet in kf.split(data):
            subsubRMSE = 0
            nmf.fit(trainSet)
            testSet = trimFun[qNum](testSet)
            nTest = len(testSet)
            print("test set size after trimming: %d", nTest)
            predictions = nmf.test(testSet)
            for p in predictions:
                subsubRMSE += pow(p.est - p.r_ui, 2)
        # average of all train-test splits of k-NN
        RMSE.append(np.mean(subRMSE))
    return RMSE


def Q24():

# so far using same code as Q10, Q12-14 for Q24, Q26-28, can combine code later
# only using SVD for Q24 for now, but the RMSE and MAE don't change much with latent factor
    data = load_data()

    meanRMSE, meanMAE = [], []
    start = time.time()
    for k in range(2, 52, 2):
        MF_svd = SVD(n_factors = k)
        out = cross_validate(MF_svd, data, measures=['RMSE', 'MAE'], cv=10)
        meanRMSE.append(np.mean(out['test_rmse']))
        meanMAE.append(np.mean(out['test_mae']))
    cv_time = str(datetime.timedelta(seconds=int(time.time() - start)))
    print("Total time used for cross validation: " + cv_time)

    k = list(range(2, 52, 2))
    ys = [[meanRMSE, 'mean RMSE'], [meanMAE, 'mean MAE']]
    #currently plot meanRMSE and meanMAE separately because it's hard to see the trend when they are plotted in same graph 
    make_plot(k, [[meanRMSE, 'mean RMSE']], 'Number of Neighbors', 'Error')
    make_plot(k, [[meanMAE, 'mean MAE']], 'Number of Neighbors', 'Error')
    return meanRMSE, meanMAE

def Q26To28(qNum, n_splits=10):
    data = load_data()
    kf = KFold(n_splits=10)

    trimFun = {26: popularTrim,
               27: unpopularTrim,
               28: highVarTrim}
    RMSE = []
    for k in range(2, 52, 2):
        MF_svd = SVD(n_factors = k)
        subRMSE = []
        for trainSet, testSet in kf.split(data):
            subsubRMSE = 0
            MF_svd.fit(trainSet)
            testSet = trimFun[qNum](testSet)
            nTest = len(testSet)
            print("test set size after trimming: %d", nTest)
            for (r, c, rating) in testSet:
                predictedRating = MF_svd.predict(str(r), str(c))
                subsubRMSE += (pow(rating - predictedRating.est, 2))
            # calculate RMSE of this train-test split
            subRMSE.append(np.sqrt(subsubRMSE / nTest))
        # average of all train-test splits of k-NN
        RMSE.append(np.mean(subRMSE))

    return RMSE

if __name__ == '__main__':
    #pop, unpop, highVar = classifyMovies()
    RMSE = Q12To14And19To21And26To28(12, 10)




