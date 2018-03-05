def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict, KFold
from sklearn import linear_model, cross_validation
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn import feature_selection
from sklearn.feature_selection import VarianceThreshold, f_regression, mutual_info_regression, SelectKBest
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import collections
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

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




def cross_val(clf, X, y, neighbor=False, shuffle=False):
    kf = KFold(n_splits=10, shuffle = shuffle)

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
    print('RMSE test %s and RMSE train %s' % (rmse_test, rmse_train))
    if neighbor == True:
        return rmse_test, rmse_train, clf.coef_
    else:
        return rmse_test, rmse_train

# 
# ys is [[y, 'label'],...]
# 
def make_plot(x, ys, scatter=False, xlabel=None, ylabel=None, xticks=None, grid=False, title=None, size_marker = 1, marker = '.'):
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

def find_best_combo(clf, hyper, ratio=[-1]):
    min_test = 999
    min_train = 999 
    min_combo = []
    min_ar = 999 
    min_ra = 999 
    min_coef = [] 
    if ratio == -1:
        ratio_list = [1]
    else:
        ratio_list = ratio
    for ra in ratio_list:
        for ar in hyper:
            if (clf == 'Ridge'):
                clf_ = Ridge(alpha = ar)
            elif (clf == 'Lasso'):
                clf_ = Lasso(alpha = ar)
            elif (clf == 'Net'):
                clf_ = ElasticNet(alpha = ar, l1_ratio = ra)
            for i in range(1,32):
                n = ([int(d) for d in str(bin(i))[2:]])
                m = [ 0 for i in range(5-len(n))]
                m.extend(n)
                mask = [] 
                for k in m:
                    if k==1:
                        mask.append(True)
                    elif k==0:
                        mask.append(False)
                X,y = getXy()
                enc = OneHotEncoder(categorical_features=mask)
                enc.fit(X)
                onehotlabels = enc.transform(X).toarray()
                clf_used = clf_
                rmse_test, rmse_train, coef = cross_val(clf_used, onehotlabels, y)

                if rmse_test < min_test:
                    min_test = rmse_test 
                    min_train = rmse_train
                    min_combo = mask 
                    min_ar = ar 
                    min_ra = ra 
                    min_coef = coef 
    print ('Regularizer used: ', clf)
    print ('Test RMSE: ',min_test)
    print ('Train RMSE: ',min_train)
    print ('Mask: ',min_combo)
    print ('Best alpha: ', min_ar)
    print ('Ratio for elastic net regularizer: ',min_ra)
    print ('Coefficient for regression: ', min_coef)
    return min_test, min_combo, min_ar

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

        # plot residual
        y_residual = y - y_predicted
        title = 'Residual against fitted values'
        ys = [[y_residual, 'Residual'],[y_predicted, 'Fitted']]
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
        lr_reg.fit(X,y)
        y_predicted = lr_reg.predict(X)
        title = 'Fitted against true values'
        ys = [[y, 'True'],[y_predicted, 'Fitted']]
        make_plot(x_plt, ys, scatter=True, title=title)

        # plot residual
        y_residual = y - y_predicted
        title = 'Residual against fitted values'
        ys = [[y_residual, 'Residual'],[y_predicted, 'Fitted']]
        make_plot(x_plt, ys, scatter=True, title=title)



        lr_mi = linear_model.LinearRegression()


        lr_reg = linear_model.LinearRegression()
        cross_val(lr_reg, X_reg, y)

        lr_mi = linear_model.LinearRegression()
        cross_val(lr_mi, X_mi, y)
    elif (option == 'iv'):
        y_test = []
        y_train = []

        print ([False, False, False, False, False])
        X,y = getXy()
        lr = linear_model.LinearRegression()
        rmse_test, rmse_train, coef = cross_val(lr, X, y)
        y_test.append(rmse_test)
        y_train.append(rmse_train)

        min_test = 999
        min_combo = []
        min_coef = []

        for i in range(1,32):
            n = ([int(d) for d in str(bin(i))[2:]])
            m = [ 0 for i in range(5-len(n))]
            m.extend(n)
            mask = [] 

            for k in m:
                if k==1:
                    mask.append(True)
                elif k==0:
                    mask.append(False)
            X,y = getXy()
            enc = OneHotEncoder(categorical_features=mask)
            enc.fit(X)
            onehotlabels = enc.transform(X).toarray()
            lr = linear_model.LinearRegression()
            rmse_test, rmse_train, coef = cross_val(lr, onehotlabels, y)

            y_test.append(rmse_test)
            y_train.append(rmse_train)
            if rmse_test < min_test:
                min_test = rmse_test 
                min_combo = mask 
                min_coef = coef 

        xs = range(1,33)
        print (xs)
        print (y_test)
        print (y_train)
        print (min_combo)
        print (min_coef)
        ys = []
        ys = [[y_test, 'RMSE_test'], [y_train, 'RMSE_train']]

        xlabel = 'Combination #'
        ylabel = 'RMSE'
        title = 'Test and Train RMSE for 32 Combinations'
        make_plot(xs, ys, scatter = True, xlabel=xlabel, ylabel=ylabel, grid=True, title=title, size_marker=40, marker='.')
    elif (option == 'v'):
        alpha_list = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        clf = 'Ridge'
        hyper = alpha_list
        find_best_combo(clf, hyper)
        clf = 'Lasso'
        find_best_combo(clf, hyper)
        clf = 'Net'
        ratio = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
        find_best_combo(clf, hyper, ratio)

def Q2b(option=None):
    X,y = getXy()

    if(option == 'i'):
        regr = RandomForestRegressor(n_estimators=20, max_depth=4, bootstrap=True,
            max_features=5, oob_score=True)
        test,train=cross_val(regr, X, y)
        print(test,train)
        regr.fit(X,y)
        print('OOB Score is ', 1-regr.oob_score_)
        print('Feature importance ', regr.feature_importances_)
    elif(option == 'ii'):
        oob = np.zeros([5,200])
        rmse_test = np.zeros([5,200])
        for feat in range(0,5):
            for tree in range(0,200):
                regr = RandomForestRegressor(n_estimators=tree+1, max_depth=4, bootstrap=True,
                    max_features=feat+1, oob_score=True)
                sub_rmse_test, _ = cross_val(regr, X, y)
                regr.fit(X,y)
                sub_oob = 1-regr.oob_score_
                rmse_test[feat][tree] = sub_rmse_test
                oob[feat][tree] = sub_oob
                if tree % 20 == 0:
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
        return rmse_test, oob
    elif(option == 'iii'):
        oob = np.zeros([5,200])
        rmse_test = np.zeros([5,200])
        for feat in range(0,5):
            for depth in range(0,200):
                regr = RandomForestRegressor(n_estimators=20, max_depth=depth+1, bootstrap=True,
                    max_features=feat+1, oob_score=True)
                sub_rmse_test, _ = cross_val(regr, X, y)
                regr.fit(X,y)
                sub_oob = 1-regr.oob_score_
                rmse_test[feat][depth] = sub_rmse_test
                oob[feat][depth] = sub_oob
                if depth%20 == 0:
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
        return rmse_test, oob
    elif(option == 'iv'):
        bestTree = 23
        bestFeature = 3
        bestDepth = 10
        regr = RandomForestRegressor(n_estimators=bestTree, max_depth=bestDepth, bootstrap=True,
            max_features=bestFeature, oob_score=True)
        rmse_test,rmse_train=cross_val(regr, X, y)
        regr.fit(X,y)
        importances = regr.feature_importances_
        indices = np.argsort(importances)[::-1]
        # Print the feature ranking
        print("Feature ranking:")

        for f in range(X.shape[1]): 
            print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

        # Plot the feature importances of the forest
        plt.figure()
        plt.title("Feature importances")
        plt.bar(range(X.shape[1]), importances[indices], align="center")
        plt.xticks(range(X.shape[1]), indices)
        plt.show()
    elif(option == 'v'):
        regr = RandomForestRegressor(n_estimators=23, max_depth=4, bootstrap=True,
            max_features=3, oob_score=True)
        rmse_test,rmse_train=cross_val(regr, X, y)
        regr.fit(X,y)
        print('OOB Score is ', 1-regr.oob_score_)
        print('Feature importance ', regr.feature_importances_)
        # save the classifier -- requires GraphViz and pydot
        import io, pydot
        from sklearn import tree
        clf = regr.estimators_[0]
        clf.fit(X,y)
        dot_data = io.StringIO()
        tree.export_graphviz(clf, out_file=dot_data)
        graph = pydot.graph_from_dot_data(dot_data.getvalue())
        graph[0].write_pdf("dtree.pdf")
    else:
        regr = RandomForestRegressor(n_estimators=23, max_depth=10, bootstrap=True,
            max_features=3, oob_score=True)
        test, train = cross_val(regr, X,y)
        print(test, train)
        regr.fit(X,y)
        
        y_pred = regr.predict(X)
        x_plt = [x for x in range(len(y))]
        title = 'Fitted against true values (Random Forest)'
        ys = [[y, 'True'], [y_pred, 'Fitted']]
        make_plot(x_plt, ys, scatter=True, title=title)

        y_resd = y - y_pred
        title = 'Residual against fitted values (Random Forest)'
        ys = [[y_resd, 'Residual'], [y_pred, 'Fitted']]
        make_plot(x_plt, ys, scatter=True, title=title)




def Q2c():
    X, y = getXy(useOnehot=True)
    y = y.astype('float16')
    activity = ['relu', 'logistic', 'tanh']
    nHiddenUnits = range(10, 250, 5)
    RMSE_test = { 'relu': [],
                  'logistic': [],
                  'tanh': []
                }
    RMSE_train = { 'relu': [],
                   'logistic': [],
                   'tanh': []
                 }
    # for a in activity:
    for a in activity:
        for n in nHiddenUnits:
            print(a, n)
            nn = MLPRegressor((n,), activation=a)
            rmse_test, rmse_train = cross_val(nn, X, y)
            RMSE_test[a].append(rmse_test)
            RMSE_train[a].append(rmse_train)
    make_plot(nHiddenUnits, [[RMSE_test['relu'], 'test'], [RMSE_train['relu'], 'train']], 'relu RMSE vs number of hidden units', 'RMSE')
    make_plot(nHiddenUnits, [[RMSE_test['logistic'], 'test'], [RMSE_train['logistic'], 'train']], 'logistic RMSE vs number of hidden units', 'RMSE')
    make_plot(nHiddenUnits, [[RMSE_test['tanh'], 'test'], [RMSE_train['tanh'], 'train']], 'tanh RMSE vs number of hidden units', 'RMSE')
    return RMSE_test, RMSE_train

def Q2cPlot():
    # plot scatter
    X, y = getXy(useOnehot=True)
    # nn = MLPRegressor((240,), activation='relu')
    # nn = MLPRegressor((35,), activation='logistic')
    nn = MLPRegressor((95,), activation='tanh')
    nn.fit(X, y)
    y_predicted = nn.predict(X)
    x_plt = [x for x in range(len(y))]
    title = 'Fitted against true values'
    ys = [[y, 'True'], [y_predicted, 'Fitted']]
    fig, ax = plt.subplots()
    make_plot(x_plt, ys, scatter=True, title=title)

    # plot residual
    y_residual = y - y_predicted
    title = 'Residual against fitted values'
    ys = [[y_residual, 'Residual'], [y_predicted, 'Fitted']]
    make_plot(x_plt, ys, scatter=True, title=title)


def Q2d(option):
    # X[week #, day of week, hour of day, work flow, file id]
    # group data by workflow
    X,y=getXy()
    Xy = np.concatenate((X,np.array([y]).T),axis=1)
    group_by_workflow = {}
    for x in Xy:
        wf = x[3]
        if wf not in group_by_workflow:
            group_by_workflow[wf] = np.array([x])
        else:
            group_by_workflow[wf] = np.concatenate(
            (group_by_workflow[wf], [x]),axis=0)
    for key in group_by_workflow:
#         print(key)
        print(len(group_by_workflow[key]))  
        Xy_wf = group_by_workflow[key]
        X_wf = Xy_wf[:,:5]
#         print(X_wf)
        y_wf = Xy_wf[:,5]
        
        if option == 'i':
            # RMSE (cross val) and plot
            lr = linear_model.LinearRegression()
            rmse_test, rmse_train = cross_val(lr, X_wf, y_wf)
            print('For work flow %s, train rmse is %f and test rmse is %f' 
                  % (key, rmse_train, rmse_test))
            
            lr_all = linear_model.LinearRegression()
            lr_all.fit(X_wf, y_wf)
            y_pred = lr_all.predict(X_wf)
            x_plt = [x for x in range(len(y_wf))]
            title = 'Fitted against true values (LR wf '+str(key)+')'
            ys = [[y_wf, 'True'], [y_pred, 'Fiited']]
            make_plot(x_plt, ys, scatter=True, title=title)

            y_resd = y_wf - y_pred
            title = 'Residual against fitted values (LR wf '+str(key)+')'
            ys = [[y_resd, 'Residual'], [y_pred, 'Fitted']]
            make_plot(x_plt, ys, scatter=True, title=title)

        if option == 'ii':
            rmse_test = np.array([])
            rmse_train = np.array([])
            deg_range = 10
            for degree in range(deg_range):
                if degree % 2 == 0:
                    print(degree)
                model = make_pipeline(PolynomialFeatures(degree), Ridge())
                sub_rmse_test, sub_rmse_train = cross_val(model, X_wf, y_wf)
                rmse_test = np.append(rmse_test, sub_rmse_test)
                rmse_train = np.append(rmse_train, sub_rmse_train)
            x_plt = [x for x in range(deg_range)]
            title = 'RMSE againse degree of the polynomial (wf '+str(key)+')'
            ys = [[rmse_test, 'RMSE Test'],[rmse_train, 'RMSE Train']]
            make_plot(x_plt, ys, title=title)

def Q2e():
    X, y = getXy()
    num_n = range(1,101)

    min_test = 999
    min_train = 999
    min_n = -1

    test_list = []
    train_list = []

    for n in num_n:
        clf = KNeighborsRegressor(n_neighbors = n)
        rmse_test, rmse_train = cross_val (clf, X, y, neighbor=True, shuffle=True)
        if rmse_test < min_test:
            min_test = rmse_test 
            min_train = rmse_train
            min_n = n 
        test_list.append(rmse_test)
        train_list.append(rmse_train)
    print ('test RMSE',min_test)
    print ('train_RMSE',min_train)
    print ('best n_neighbor',min_n)

    # plot test and train against num neighbor 
    x_plt = num_n
    title = 'RMSE over number of neighbors'
    ys = [[test_list, 'test_RMSE'], [train_list, 'train_RMSE']]
    fig, ax = plt.subplots()
    make_plot(x_plt, ys, xlabel = 'n_neighbors', ylabel = 'RMSE', scatter=False, title=title)

    # plot scatter
    clf = KNeighborsRegressor(n_neighbors = min_n)
    clf.fit(X, y)
    y_predicted = clf.predict(X)
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


if __name__ == '__main__':
    #Q1('a')
    #Q1('b')
    # Q2a('i')
    # Q2a('v')
    Q2b()
    # Q2b()
    # Q2cPlot()
    # Q2e()
