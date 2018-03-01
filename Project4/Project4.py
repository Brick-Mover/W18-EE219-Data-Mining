import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt


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

if __name__ == '__main__':
	Q1('a')
	Q1('b')


