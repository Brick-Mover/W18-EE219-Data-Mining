import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict, KFold 
from sklearn import linear_model, cross_validation
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn import feature_selection
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestRegressor

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


def encode_workflow(workflow, useOnehot=False):
	for i in range(len(workflow)):
		workflow[i] = int(workflow[i].split('_')[-1])
		if useOnehot:
			workflow[i] = one_hot(N_WORKFLOW, workflow[i])
	return workflow 


def encode_files(files, useOnehot=False):
	for i in range(len(files)):
		files[i] = int (files[i].split('_')[-1])
		if useOnehot:
			files[i] = one_hot(N_FILE, files[i])
	return files


def encode_day(days, useOnehot=False):
	week_days = { 'Monday' : 1,
				  'Tuesday' : 2,
				  'Wednesday' : 3,
				  'Thursday' : 4,
				  'Friday' : 5,
				  'Saturday' : 6,
				  'Sunday' : 7 }
	for i in range(len(days)):
		days[i] = week_days[days[i]]
		if useOnehot:
			days[i] = one_hot(N_DAY, days[i]-1)
	return days


def getXy(useOnehot=False):
	X = data_frame.iloc[:,[0,1,2,3,4]].values
	X[:,1] = encode_day(X[:,1], useOnehot)
	X[:,3] = encode_workflow(X[:,3], useOnehot)
	X[:,4] = encode_files(X[:,4], useOnehot)
	if useOnehot:
		for i in range(len(X[:0])):
			X[:0][i] = one_hot(N_WEEK, X[:0][i]-1)
			X[:2][i] = one_hot(N_HOUR, X[:2][i]-1)
	y = data_frame.iloc[:,5].values
	return X,y


def cross_val(clf, X, y):
	kf = KFold(n_splits=10)

	rmse_test = []
	rmse_train = []
	for train_index, test_index in kf.split(X):
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
		clf.fit(X_train, y_train)
		y_predicted = clf.predict(X_test)
		y_predicted_train = clf.predict(X_train)
		rmse_test.append(sqrt(mean_squared_error(y_test, y_predicted)))
		rmse_train.append(sqrt(mean_squared_error(y_train, y_predicted_train)))

	# TODO: need to double check how to report test and train rmse. refer to Piazza 
	print ('test rmse: ', np.mean(rmse_test))
	print ('train rmse: ', np.mean(rmse_train))

# 
# ys is [[y, 'label'],...]
# 
def scatter(x, ys, xlabel=None, ylabel=None, xticks=None, grid=False, title=None):
    for y, label in ys:
        plt.scatter(x, y, s=1, marker='.', label=label)
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

def Q2a():
	X,y = getXy()

	lr = linear_model.LinearRegression()
	cross_val(lr, X, y)

	# TODO: do we use all data to plot scatterplot? 
	# plot scatter 
	lr.fit(X, y)
	y_predicted = lr.predict(X)
	x_plt = [x for x in range(len(y))]
	title = 'Fitted against ture values'
	ys = [[y, 'Ture'], [y_predicted, 'Fitted']]
	fig, ax = plt.subplots()
	scatter(x_plt, ys, title=title)


	# plot residual 
	y_residual = y - y_predicted 
	title = 'Residual against fitted values'
	ys = [[y_residual, 'Residual'],[y_predicted, 'Fitted']]
	scatter(x_plt, ys, title=title)

	# TODO: 
	# standardize 
	# standardization shows no change right now. need to fix 

	x_scaler = StandardScaler()
	X_stan = x_scaler.fit_transform(X)
	lr_stan = linear_model.LinearRegression()
	cross_val(lr_stan, X_stan, y)

	lr_stan.fit(X_stan, y)
	y_predicted = lr_stan.predict(X_stan)
	title = 'Fitted against ture values'
	ys = [[y, 'True'],[y_predicted, 'Fitted']]
	scatter(x_plt, ys, title=title)


if __name__ == '__main__':
	#Q1('a')
	# Q1('b')
	Q2a()