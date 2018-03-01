import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict, KFold
from sklearn import linear_model, cross_validation
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn import feature_selection
from sklearn.feature_selection import VarianceThreshold, f_regression, mutual_info_regression, SelectKBest
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPClassifier
import collections

"""
CONSTANTS HERE
"""
N_WORKFLOW = 5
N_FILE = 30
N_DAY = 7
N_WEEK = 15
N_HOUR = 24


data_frame = pd.read_csv("./data/network_backup_dataset.csv")
column_names = data_frame.columns
counter = 0
print("column index --- name:")
for i in column_names:
    print("           " + str(counter) + " --- "+ i)
    counter = counter + 1

def get_date(week_num, day_num):
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    return ((int(week_num) - 1) * 7 + weekdays.index(day_num))


prepare_data = dict() # {"work_flow_id": {"file_name": [0,0,....,0]}}

for label_list, row_info in data_frame.groupby([column_names[3],column_names[4]]):
    work_flow_id = label_list[0]
    file_name = label_list[1]

    if work_flow_id not in prepare_data:
        prepare_data[work_flow_id] = dict()
    if file_name not in prepare_data[work_flow_id]:
        prepare_data[work_flow_id][file_name] = [0] * 106

    for week_and_day_list, row in row_info.groupby([column_names[0], column_names[1]]):
        date = get_date(week_and_day_list[0], week_and_day_list[1])
        prepare_data[work_flow_id][file_name][date] = sum(row[column_names[5]])


def problem1_plot(period):
    x = [a for a in range(0, period+1)]
    for work_flow_id in prepare_data:
        yaxis = [0] * (period+1)
        for day in range(0, period+1, 1):
            totalsize = 0
            for file_name in prepare_data[work_flow_id]:
                totalsize = totalsize + prepare_data[work_flow_id][file_name][day]
            yaxis[day] = totalsize
        #print(yaxis)
        #print("------")
        plt.plot(x, yaxis, label = "work_flow_ " + str(work_flow_id))
        plt.title(str(period) + " days period")
        plt.xlabel('Days')
        plt.ylabel('Total Size of Backup in GB')
    plt.legend()
    plt.show()

def Q1(option):
    if option == 'a':
        problem1_plot(20)
    elif option == 'b':
        problem1_plot(105)


def one_hot(total: int, one: int):
    result = [0] * total
    result[one] = 1
    return result


def encode_workflow(workflow):
    for i in range(len(workflow)):
        workflow[i] = int(workflow[i].split('_')[-1])
    return workflow


def encode_files(files):
    for i in range(len(files)):
        files[i] = int (files[i].split('_')[-1])
    return files


def encode_day(days):
    week_days = { 'Monday' : 1,
                  'Tuesday' : 2,
                  'Wednesday' : 3,
                  'Thursday' : 4,
                  'Friday' : 5,
                  'Saturday' : 6,
                  'Sunday' : 7 }
    for i in range(len(days)):
        days[i] = week_days[days[i]]
    return days


def flatten(l):
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el


def getXy(useOnehot=False):
    X = data_frame.iloc[:,[0,1,2,3,4]].values
    X[:,1] = encode_day(X[:,1])
    X[:,3] = encode_workflow(X[:,3])
    X[:,4] = encode_files(X[:,4])
    if useOnehot:
        for r in range(X.shape[0]):
            X[r][0] = one_hot(N_WEEK, X[r][0]-1)
            X[r][1] = one_hot(N_DAY, X[r][1]-1)
            X[r][2] = one_hot(N_HOUR, X[r][2]-1)
            X[r][3] = one_hot(N_WORKFLOW, X[r][3])
            X[r][4] = one_hot(N_FILE, X[r][4])
        X = X.tolist()
        for i in range(len(X)):
            X[i] = [item for sublist in X[i] for item in sublist]
        X = np.array(X)
    y = data_frame.iloc[:,5].values
    return X,y


def cross_val(clf, X, y):
    kf = KFold(n_splits=10)

    # squre root errors sr_test and sr_train
    sr_test = np.array([])
    sr_train = np.array([])
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)
        y_pred_test = clf.predict(X_test)
        y_pred_train = clf.predict(X_train)
        sub_sr_test = pow(np.array(y_test)-np.array(y_pred_test), 2)
        sr_test = np.append(sr_test, sub_sr_test)
        sub_sr_train = pow(np.array(y_train)-np.array(y_pred_train), 2)
        sr_train = np.append(sr_train, sub_sr_train)

    rmse_test = sqrt(np.sum(sr_test)/sr_test.size)
    rmse_train = sqrt(np.sum(sr_train)/sr_train.size)
    print(sr_test.size, sr_train.size)
    # TODO: need to double check how to report test and train rmse. refer to Piazza
    print ('test rmse: ', np.mean(rmse_test))
    print ('train rmse: ', np.mean(rmse_train))
    return rmse_test, rmse_train

# 
# ys is [[y, 'label'],...]
# 
def make_plot(x, ys, scatter=False, xlabel=None, ylabel=None, xticks=None, grid=False, title=None):
    for y, label in ys:
        if scatter:
            plt.scatter(x, y, s=1, marker='.', label=label)
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


def Q2a(option):
    X,y = getXy()

    if (option == 'i'):
        lr = linear_model.LinearRegression()
        cross_val(lr, X, y)

        # plot scatter
        lr.fit(X, y)
        y_predicted = lr.predict(X)
        x_plt = [x for x in range(len(y))]
        title = 'Fitted against true values'
        ys = [[y, 'True'], [y_predicted, 'Fitted']]
        fig, ax = plt.subplots()
        make_plot(x_plt, ys, scatter=True, title=title)

        # plot residual
        y_residual = y - y_predicted
        title = 'Residual against fitted values'
        ys = [[y_residual, 'Residual'],[y_predicted, 'Fitted']]
        make_plot(x_plt, ys, scatter=True, title=title)
    elif (option == 'ii'):
        # standardize
        x_plt = [x for x in range(len(y))]
        x_scaler = StandardScaler()
        X_stan = x_scaler.fit_transform(X)
        lr_stan = linear_model.LinearRegression()
        cross_val(lr_stan, X_stan, y)

        lr_stan.fit(X_stan, y)
        y_predicted = lr_stan.predict(X_stan)
        title = 'Fitted against true values'
        ys = [[y, 'True'],[y_predicted, 'Fitted']]
        make_plot(x_plt, ys, scatter=True, title=title)
    elif (option == 'iii'):
        # f_regression and mutual info regression
        F, p = f_regression(X,y)
        mi = mutual_info_regression(X,y)
        print (F)
        print (mi)

        X_reg = SelectKBest(f_regression, k=3).fit_transform(X,y)
        X_mi = SelectKBest(mutual_info_regression, k=3).fit_transform(X,y)

        lr_reg = linear_model.LinearRegression()
        cross_val(lr_reg, X_reg, y)

        lr_mi = linear_model.LinearRegression()
        cross_val(lr_mi, X_mi, y)

def Q2b(option):
    X,y = getXy()

    if(option == 'i'):
        regr = RandomForestRegressor(n_estimators=20, max_depth=4, bootstrap=True,
            max_features=5, oob_score=True)
        cross_val(regr, X, y)
        regr.fit(X,y)
        print('OOB Score is ', 1-regr.oob_score_)
    elif(option == 'ii'):
        oob = np.zeros([5,200])
        rmse_test = np.zeros([5,200])
        for feat in range(0,5,3):
            for tree in range(0,200,50):
                regr = RandomForestRegressor(n_estimators=tree+1, max_depth=4, bootstrap=True,
                    max_features=feat+1, oob_score=True)
                sub_rmse_test, _ = cross_val(regr, X, y)
                regr.fit(X,y)
                sub_oob = 1-regr.oob_score_
                rmse_test[feat][tree] = sub_rmse_test
                oob[feat][tree] = sub_oob
                print('feature %s and tree num %s',(feat, tree))

        x_plt = [x for x in range(200)]
        ys_rmse = []
        for i in range(5):
            ys_rmse.append([rmse_test[i], 'max feature '+str(i)])
        title = 'Test-RMSE against number of trees'
        make_plot(x_plt, ys_rmse, title=title)

        ys_oob = []
        for i in range(5):
            ys_oob.append([oob[i], 'max feature '+str(i)])
        title = 'Out of bag error against number of trees'
        make_plot(x_plt, ys_oob, title=title)
    elif(option == 'iii'):
        oob = np.zeros([5,200])
        rmse_test = np.zeros([5,200])
        for feat in range(0,5,3):
            for depth in range(0,200,50):
                regr = RandomForestRegressor(n_estimators=20, max_depth=depth+1, bootstrap=True,
                    max_features=feat+1, oob_score=True)
                sub_rmse_test, _ = cross_val(regr, X, y)
                regr.fit(X,y)
                sub_oob = 1-regr.oob_score_
                rmse_test[feat][depth] = sub_rmse_test
                oob[feat][depth] = sub_oob
                print('max feature %s and max tree depth %s',(feat, depth))

        x_plt = [x for x in range(200)]
        ys_rmse = []
        for i in range(5):
            ys_rmse.append([rmse_test[i], 'max feature '+str(i)])
        title = 'Test-RMSE against max depth of trees'
        make_plot(x_plt, ys_rmse, title=title)

        ys_oob = []
        for i in range(5):
            ys_oob.append([oob[i], 'max feature '+str(i)])
        title = 'Out of bag error against max depth of trees'
        make_plot(x_plt, ys_oob, title=title)




def Q2c():
    X, y = getXy(useOnehot=True)
    activity = ['relu', 'logistic', 'tanh']
    nHiddenUnits = range(50, 200, 5)
    for a in activity:
        for n in nHiddenUnits:
            nn = MLPClassifier((n,), activation=a)
            cross_val(nn, X, y)

if __name__ == '__main__':
    #Q1('a')
    #Q1('b')
    # Q2a('i')
    # Q2a('ii')
    Q2b('ii')
