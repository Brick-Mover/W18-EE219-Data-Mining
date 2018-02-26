
import numpy as np
import pandas as pd
import time, datetime
import matplotlib.pyplot as plt
from surprise.prediction_algorithms.knns import KNNWithMeans
from surprise.prediction_algorithms.matrix_factorization import SVD, NMF
from surprise.model_selection import cross_validate
from surprise import Dataset, Reader
from surprise.model_selection import KFold
import math
from surprise.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from collections import defaultdict

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
        'rating': row_rating.transpose().tolist()[0]    # map (0.5, 1, ..., 5) to (1, 2, ..., 10)
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

def Q1to6():
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

    R_row = np.amax(row_userId)
    R_col = np.amax(mapped_row_movieId)
    print('Matrix has row size (users) %s, and col size (movies) %s'
          % (R_row, R_col))
    R = np.zeros([R_row, R_col])
    for i in range(row_userId.size):
        r = row_userId[i] - 1
        c = mapped_row_movieId[i] - 1
        rating = row_rating[i]
        R[r, c] = rating

    rating_avl = np.count_nonzero(R)
    rating_psb = np.prod(R.shape)
    sparsity = rating_avl/rating_psb
    print(sparsity)

    # Question 2
    # plot a historgram showing frequency of rating values
    ratings_arr = []
    for r in range(R_row):
        for c in range(R_col):
            if R[r,c]!=0.0:
                ratings_arr.append(R[r,c])
    binwidth = 0.5
    print (min(ratings_arr))
    print (max(ratings_arr))

    plt.hist(ratings_arr, bins=np.arange(min(ratings_arr), max(ratings_arr) + binwidth, binwidth))
    plt.show()
    plt.close()

    # Question 3
    l = [0 for x in range(0, R_col)] #R_row

    for r in range(R_row):
        for c in range(R_col): 
            if R[r,c]!=0.0:
                l[c] = l[c] + 1
    l_no_zero = [val for val in l if val!=0]
    l_no_zero.sort(reverse = True)

    plt.plot([i+1 for i in range(0, len(l_no_zero))], l_no_zero)
    plt.show()
    plt.close()

    # Question 4
    l = np.zeros(R_row)
    for r in row_userId:
        l[r[0]-1] += 1
    l[::-1].sort()
    plt.plot([i for i in range(1, len(l)+1)], l)
    plt.show()
    plt.close()

    # Q6
    var = np.array([])
    for c in range(R_col):
        var = np.append(var, np.var(R[:,c]))
    var_bin = np.zeros(math.ceil((np.amax(var)-np.amin(var))/0.5))
    for v in var:
        var_bin[math.floor(v/0.5)] += 1
    plt.hist(var, bins=np.arange(min(var),max(var),0.5))
    plt.show()
    plt.close()


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
    reader = Reader(rating_scale=(0.5, 5))
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

def plot_ROC(yTrue, yScore, title='ROC Curve'):
    fpr, tpr, thresholds = roc_curve(yTrue, yScore)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    lw = 2
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

def Q15and22and29(qNum, bestK, thres=[2.5,3,3.5,4]):
    range = 5.0
    sim_options = {
        'name': 'pearson_baseline',
        'shrinkage': 0  # no shrinkage
    }
    data = load_data()
    trainset, testset = train_test_split(data, test_size=0.1)
    if qNum == 15:
        model = KNNWithMeans(bestK, sim_options=sim_options)
    elif qNum == 22:
        model = NMF(n_factors=bestK)
    else:
        model = SVD(n_factors=bestK)

    model.fit(trainset)
    pred = model.test(testset)
    for thrs in thres:
        np_true = np.array([])
        np_score = np.array([])
        for u, i, t, p, d in pred:
            if t >= thrs:
                t = 1
            else:
                t = 0
            np_true = np.append(np_true, t)
            np_score = np.append(np_score, p/range)
        title = 'Threshold '+str(thrs)
        plot_ROC(np_true, np_score, title=title)

def Q23(col=0):
    print('Chosen column is '+str(col))
    data = np.loadtxt('ml-latest-small/ratings.csv',
                  delimiter=',', skiprows=1, usecols=(0,1,2))

    row_userId = data[:,:1].astype(int)
    row_movieId = data[:,1:2].astype(int)
    row_rating = data[:,2:3]

    sortedId = np.sort(row_movieId.transpose()[0])
    m = {}
    idx = 0
    last = None
    for i in sortedId.tolist():
        if i != last:
            m[i] = idx
            idx += 1
        last = i
    
    data = load_data()
    model = NMF(n_factors = 20)
    trainset, testset = train_test_split(data, test_size=0.0001)
    model.fit(trainset)
    U = model.pu
    V = model.qi

    import csv
    dict_ID_to_genre = {}
    with open('ml-latest-small/movies.csv', 'rt') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        cnt = 0
        for row in reader:
            if cnt != 0:
                dict_ID_to_genre[row[0]] = row[1:]
            cnt += 1

    dict_col_to_ID = {}
    for key in m:
        dict_col_to_ID[m[key]] = key

    V_col = V[:, col]
    V_col_sort_top10 = np.sort(V_col)[::-1][:10]
    V_col_list = V_col.tolist()
    for val in V_col_sort_top10:
        ind = V_col_list.index(val)
        m_id = dict_col_to_ID[ind]
        genre = dict_ID_to_genre[str(m_id)]
        print(genre[-1])

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
            colCnt[m.movieID] = [m.rating]
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
            model = NMF(n_factors=k)
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
    xTitle = 'Number of Neighbors' if qNum <= 14 else 'Number of latent factors'
    make_plot(k, ys, xTitle, 'Error')
    return RMSE


def Q17():
    data = load_data()

    meanRMSE, meanMAE = [], []
    start = time.time()
    for k in range(16, 24, 2):
        nmf = NMF(n_factors=k)
        out = cross_validate(nmf, data, measures=['RMSE', 'MAE'], cv=10)
        meanRMSE.append(np.mean(out['test_rmse']))
        meanMAE.append(np.mean(out['test_mae']))
    cv_time = str(datetime.timedelta(seconds=int(time.time() - start)))
    print("Total time used for cross validation: " + cv_time)

    k = list(range(16, 24, 2))
    ys = [[meanRMSE, 'mean RMSE'], [meanMAE, 'mean MAE']]
    make_plot(k, ys, 'Number of Latent Factors', 'ratings')
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

# 
# ---------------------------------------------------------
# Begin Q30-34
# ---------------------------------------------------------
# 
class NaiveCF(object):
    
    def __init__(self):
        self.est=None
    
    def fit(self, trainset):
        ratings = trainset.all_ratings()
        self.est = np.array([0.0 for x in trainset.all_users()])
        d = {}
        for r in ratings:
            key = r[0]
            val = r[2]
            if key in d.keys():
                d[key] = np.append(d[key], val)
            else:
                d[key] = np.array([val])
#         print(d[1])
#         print(np.mean(d[1]))
#         print(d)
        for k in d.keys():
            self.est[k] = np.mean(d[k])
#             print(k, self.est[k], d[k])
#         print(self.est)
        
        
    # simplified test function, only return tr (true rating)
    # and est (estimated rating)
    def test(self, testset):
        uid = np.array([])
        iid = np.array([])
        tr = np.array([])
        est = np.array([])
        for t in testset:
            uid = np.append(uid, t[0])
            iid = np.append(iid, t[1])
            tr = np.append(tr, t[2])
            est = np.append(est, self.est[t[0]-1])
        return uid, iid, tr, est

def Q30to33(qNum):
    data = load_data()
    data_full = data.build_full_trainset()
    pop, unpop, highVar = classifyMovies()
    kf = KFold(n_splits=10)
    ncf = NaiveCF()
    ncf.fit(data_full)
    subRMSE = np.array([])
    iter = 1
    for trainSet, testSet in kf.split(data):
        if qNum == 31:
            testSet = list(filter(lambda x: x[1] in pop, testSet))
        if qNum == 32:
            testSet = list(filter(lambda x: x[1] in unpop, testSet))
        if qNum == 33:
            testSet = list(filter(lambda x: x[1] in highVar, testSet))
        nTest = len(testSet)
        print("Split " + str(iter) + ": test set size after trimming: %d", nTest)
        iter += 1
        uid, iid, tr, est = ncf.test(testSet)
        subsubRMSE = pow(est-tr, 2)
        subsubRMSE = np.sum(subsubRMSE)
        subRMSE = np.append(subRMSE, np.sqrt(subsubRMSE/nTest))
    RMSE = np.mean(subRMSE)
    print("Q"+str(qNum)+" has RMSE "+str(RMSE))

def Q34():
    rang = 5.0
    sim_options = {
        'name': 'pearson_baseline',
        'shrinkage': 0  # no shrinkage
    }
    data = load_data()
    trainset, testset = train_test_split(data, test_size=0.1)
    knn = KNNWithMeans(22, sim_options=sim_options)
    nmf = NMF(n_factors=18)
    svd = SVD(n_factors=8)
    fp = {}
    tp = {}
    area = np.array([])
    for model, key in zip([knn, nmf, svd], ['KNN','NNMF','SVD']):
        model.fit(trainset)
        pred = model.test(testset)
        np_true = np.array([])
        np_score = np.array([])
        for _, _, t, p, _ in pred:
            if t >= 3:
                t = 1
            else:
                t = 0
            np_true = np.append(np_true, t)
            np_score = np.append(np_score, p/rang)
        fpr, tpr, thresholds = roc_curve(np_true, np_score)
        print(fpr.shape, tpr.shape)
        roc_auc = auc(fpr, tpr)
        fp[key] = fpr
        tp[key] = tpr
        area = np.append(area, roc_auc)
    plt.figure()
    lw = 2
    for mod, f, t, roc_auc in zip(['KNN','NNMF','SVD'], fp, tp, area):
        fpr = fp[f]
        tpr = tp[t]
    #     label = mod+'ROC curve (area = '+str(roc_auc)+'0.2f)'
        plt.plot(fpr, tpr, lw=lw, label='%s ROC curve (area = %0.2f)' % (mod,roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    plt.show()
    plt.close()
# 
# ---------------------------------------------------------
# End Q30-34
# ---------------------------------------------------------
# 

# Note that this function, precision_recall, is referenced from: 
# http://surprise.readthedocs.io/en/stable/FAQ.html 
def precision_recall (predictions, t):
    threshold = 3
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        n_rec_t = sum((est >= threshold) for (est, _) in user_ratings[:t])
        n_rel_and_rec_t = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:t])
        precisions[uid] = n_rel_and_rec_t / n_rec_t if n_rec_t != 0 else 1
        recalls[uid] = n_rel_and_rec_t / n_rel if n_rel != 0 else 1

    return precisions, recalls

def Q36To38(qNum):

    print ("problem ",qNum)

    data = load_data()
    sim_options = {
        'name': 'pearson_baseline',
        'shrinkage': 0  # no shrinkage
    }
    filter = {
        36: 'KNNWithMeans',
        37: 'NMF',
        38: 'SVD',
    }
    k_KNNWithMeans = 30 # from Q11
    k_NMF = 18 # from Q18
    k_SVD = 8 # from Q25 

    modelName = filter[qNum]

    if modelName == 'KNNWithMeans':
        model = KNNWithMeans(k_KNNWithMeans, sim_options=sim_options)
    elif modelName == 'NMF':
        model = NMF(n_factors=k_NMF)
    else:
        model = SVD(n_factors = k_SVD)

    # sweep t from 1 to 25
    precision_arr = []
    recall_arr = []
    for t in range (1,26):
        kf = KFold(n_splits=10)
        for trainSet, testSet in kf.split(data):
            sub_precisions = 0.0
            sub_recalls = 0.0
            model.fit(trainSet)
            predictions = model.test(testSet)
            precisions, recalls = precision_recall (predictions, t)
            print(sum(prec for prec in precisions.values()) / len(precisions))
            sub_precisions += (sum(prec for prec in precisions.values()) / len(precisions))
            print(sum(rec for rec in recalls.values()) / len(recalls))
            sub_recalls += (sum(rec for rec in recalls.values()) / len(recalls))
        precision_arr.append(np.mean(sub_precisions))
        recall_arr.append(np.mean(sub_recalls))


    t_list = list(range (1,26))
    ys = [[precision_arr, 'mean precisions'], [recall_arr, 'mean recalls']]

    print ("model name: ",modelName)

    # make_plot(t_list, ys, 'recommended item size t','Precision')
    # precision vs t
    title_ = "precision vs t for: " + modelName
    make_plot(t_list, [[precision_arr, 'mean precisions']], 'recommended item size t','Precision', title=title_)
    # recall vs t
    title_ = "recall vs t for: " + modelName
    make_plot(t_list, [[recall_arr, 'mean recalls']], 'recommended item size t','Recall', title=title_)
    # precision vs recall 
    title_ = "precision vs recall for: " + modelName
    #make_plot([recall_arr, 'mean recalls'], [[precision_arr, 'mean precisions']], 'Recall','Precision', title = title_)

    plt.plot(recall_arr, precision_arr, label = modelName)
    xlabel = "recall"
    ylabel = "precision"
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()
    plt.title(title_)
    plt.show()


    return precision_arr, recall_arr 

def Q39():
    # KNN
    precision_knn, recall_knn = Q36To38(36)
    # NMF
    precision_nmf, recall_nmf = Q36To38(37)
    # SVD 
    precision_svd, recall_svd = Q36To38(38)

    # precision vs recall 
    plt.plot(recall_knn, precision_knn, label = "KNNWithMeans")
    plt.plot(recall_nmf, precision_nmf, label = "NMF")
    plt.plot(recall_svd, precision_svd, label = "SVD")
    xlabel = "recall"
    ylabel = "precision"
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()
    plt.title("precision-recall curve for KNN, NMF, SVD")
    plt.show()



if __name__ == '__main__':

    # RMSE12 = Q12To14And19To21And26To28(12)
    # RMSE13 = Q12To14And19To21And26To28(13)
    # RMSE14 = Q12To14And19To21And26To28(14)
    # RMSE19 = Q12To14And19To21And26To28(19, 30)
    # RMSE20 = Q12To14And19To21And26To28(20)
    # RMSE21 = Q12To14And19To21And26To28(21)
    # RMSE26 = Q12To14And19To21And26To28(26, 20)
    # RMSE27 = Q12To14And19To21And26To28(27)
    # RMSE28 = Q12To14And19To21And26To28(28)
    # meanRMSE, meanMAE = Q17()
    # Q15and22and29(22, bestK=18)
    #Q23(col=0)
    # for q in [30,31,32,33]:
    #     Q30to33(q)
    Q17()