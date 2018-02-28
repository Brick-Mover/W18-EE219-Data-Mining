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


data_frame = pd.read_csv("./data/network_backup_dataset.csv")
column_names = data_frame.columns
counter = 0;
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
	x = [a for a in range(0,period+1)]
	for work_flow_id in prepare_data:
		for file_name in prepare_data[work_flow_id]:
			plt.plot(x, prepare_data[work_flow_id][file_name][:period+1], label = "file " + str(file_name))
			plt.title(work_flow_id)
			plt.xlabel('Days')
			plt.ylabel('Size of Backup in GB')
		plt.legend()
		plt.show()

def Q1(option):
	if option == 'a':
		problem1_plot(20)
	elif option == 'b':
		problem1_plot(105)

def encode_workflow(workflow):
	for i in range(len(workflow)):
		workflow[i] = int (workflow[i].split('_')[-1])
	return workflow 

def encode_files (files):
	for i in range(len(files)):
		files[i] = int (files[i].split('_')[-1])
	return files 

def encode_day(days):
	week_days = {'Monday' : 1, 'Tuesday' : 2, 'Wednesday' : 3 , 'Thursday' : 4, 'Friday' : 5,
'Saturday' : 6, 'Sunday' : 7 }
	for i in range(len(days)):
		days[i] = week_days ['Monday']
	return days

def Q2a():
	X = data_frame.ix [:,[0,1,2,3,4,6]].values
	X [:,1] = encode_day(X[:,1])
	X [:,3] = encode_workflow(X[:,3])
	X [:,4] = encode_files(X[:,4])

	y = data_frame.ix [:,5].values
	lr = linear_model.LinearRegression()

	kf = KFold(n_splits=10)

	rmse_test = []
	rmse_train = []

	# 10 folds 
	for train_index, test_index in kf.split(X):
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
		lr.fit(X_train, y_train)
		y_predicted = lr.predict(X_test)
		y_predicted_train = lr.predict(X_train)
		rmse_test.append(sqrt(mean_squared_error(y_test, y_predicted)))
		rmse_train.append(sqrt(mean_squared_error(y_train, y_predicted_train)))
	# TODO: need to double check how to report test and train rmse. refer to Piazza 
	print ('test rmse: ', np.mean(rmse_test))
	print ('train rmse: ', np.mean(rmse_train))

	# TODO: do we use all data to plot scatterplot? 
	# plot scatter 
	lr.fit(X, y)
	y_predicted = lr.predict(X)
	fig, ax = plt.subplots()
	ax.scatter(y, y_predicted, edgecolors=(0, 0, 0))
	ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
	ax.set_xlabel('Measured')
	ax.set_ylabel('Predicted')
	plt.show()

	# plot residual 
	y_residual = y - y_predicted 
	fig, ax = plt.subplots()
	ax.scatter(y_predicted, y_residual, edgecolors=(0, 0, 0))
	ax.set_xlabel('Fitted')
	ax.set_ylabel('Residual')
	plt.show()

	# TODO: 
	# standardize 
	# standardization shows no change right now. need to fix 

	x_scaler = StandardScaler()
	X_stan = x_scaler.fit_transform(X)
	y_scaler = StandardScaler()
	y_stan = y_scaler.fit_transform(y[:,None])[:,0]

	for train_index, test_index in kf.split(X_stan):
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
		lr.fit(X_train, y_train)
		y_predicted = lr.predict(X_test)
		y_predicted_train = lr.predict(X_train)
		rmse_test.append(sqrt(mean_squared_error(y_test, y_predicted)))
		rmse_train.append(sqrt(mean_squared_error(y_train, y_predicted_train)))
	print ('test rmse: ', np.mean(rmse_test))
	print ('train rmse: ', np.mean(rmse_train))

	# predicted_stan = cross_val_predict(lr, X_stan, y_stan, cv=10) 

	# fig_stan, ax_stan = plt.subplots()
	# ax_stan.scatter(predicted_stan, X, edgecolors=(0, 0, 0))
	# ax_stan.plot([predicted_stan.min(), predicted_stan.max()], [predicted_stan.min(), predicted_stan.max()], 'k--', lw=4)
	# ax_stan.set_xlabel('Measured')
	# ax_stan.set_ylabel('Predicted')
	# plt.show()


if __name__ == '__main__':
	#Q1('a')
	#Q1('b')
	Q2a()