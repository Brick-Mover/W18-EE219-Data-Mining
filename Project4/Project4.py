import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from math import sqrt


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
	# day of week 
	X = data_frame.ix [:,[0,1,2,3,4,6]].values
	# day 
	X [:,1] = encode_day(X[:,1])
	# work flow 
	X [:,3] = encode_workflow(X[:,3])
	X [:,4] = encode_files(X[:,4])

	y = data_frame.ix [:,5].values

	lr = linear_model.LinearRegression()
	predicted = cross_val_predict(lr, X, y, cv=10) 

	# rmse

	rmse = sqrt(mean_squared_error(y, predicted))
	print ('rmse: ', rmse)

	print ('plotting fitted values against true values...')

	fig, ax = plt.subplots()
	ax.scatter(y, predicted, edgecolors=(0, 0, 0))
	ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
	ax.set_xlabel('Measured')
	ax.set_ylabel('Predicted')
	plt.show()

	print ('plotting residuals versus fitted values...')
	# residual

	y_residual = y - predicted 
	fig, ax = plt.subplots()
	ax.scatter(predicted, y_residual, edgecolors=(0, 0, 0))
	ax.set_xlabel('Fitted')
	ax.set_ylabel('Residual')
	plt.show()

	#standardize numerical features 

	print ('plotting fitted values against true values after standardization...')

	x_scaler = StandardScaler()
	X_stan = x_scaler.fit_transform(X)
	y_scaler = StandardScaler()
	y_stan = y_scaler.fit_transform(y[:,None])[:,0]

	predicted_stan = cross_val_predict(lr, X_stan, y_stan, cv=10) 

	fig_stan, ax_stan = plt.subplots()
	ax_stan.scatter(y_stan, predicted_stan, edgecolors=(0, 0, 0))
	ax_stan.plot([y_stan.min(), y_stan.max()], [y_stan.min(), y_stan.max()], 'k--', lw=4)
	ax_stan.set_xlabel('Measured')
	ax_stan.set_ylabel('Predicted')
	plt.show()


	# file name 

	# print (X)

if __name__ == '__main__':
	#Q1('a')
	#Q1('b')
	Q2a()