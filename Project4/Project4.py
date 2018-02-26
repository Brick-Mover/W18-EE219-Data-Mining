import numpy as np
import pandas as pd 

data = pd.read_csv("./data/network_backup_dataset.csv")
column_names = data.columns
#for i in column_names:
#	print(i)

def get_days(week_num, day_num):
	result = (week_num - 1) * 7 + day_num
	return result

def problem1_plot(period):
	x = range(period)