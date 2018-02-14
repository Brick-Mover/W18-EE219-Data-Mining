import numpy as np
import csv
import matplotlib.pyplot as plt

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

