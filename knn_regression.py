#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 10:17:38 2018

@author: christinakronser
"""


import csv
import numpy as np
import matplotlib.pyplot as plt

def import_data(ifname):
    """
    Imports data with file-name/-path ifname as a numpy array.
    """
    with open(ifname, 'r') as ifile:
        datareader = csv.reader(ifile, delimiter=';')
        # we want to avoid importing the header line.
        # instead we'll print it to screen
        header = next(datareader)
        #print("Importing data with fields:\n\t" + ",".join(header))
        # create an empty list to store each row of data
        data = []
        for row in datareader:
            # for each row of data 
            # convert each element (from string) to float type
            row_of_floats = list(map(float,row))
            # now store in our data list
            data.append(row_of_floats)
        # convert the data (list object) into a numpy array.
        data_as_array = np.array(data)
        # return this array to caller
        return data_as_array
  
def train_and_test_split(N, test_fraction=None):
    """
    Randomly generates a train/test split for data of size N.

    parameters
    ----------
    N - the dataset size
    test_fraction - a fraction (between 0 and 1) specifying the proportion of
        the data to use as test data.
    """
    if test_fraction is None:
        test_fraction = 0.5
    p = [test_fraction,(1-test_fraction)]
    train_part = np.random.choice([False,True],size=N, p=p)
    test_part = np.invert(train_part)
    return train_part, test_part

def train_and_test_partition(inputs, targets, train_part, test_part):
    """
    Splits a data matrix (or design matrix) and associated targets into train
    and test parts.

    parameters
    ----------
    inputs - a 2d numpy array whose rows are the datapoints, or can be a design
        matric, where rows are the feature vectors for data points.
    targets - a 1d numpy array whose elements are the targets.
    train_part - A list (or 1d array) of N booleans, where N is the number of
        data points. If the ith element is true then the ith data point will be
        added to the training data.
    test_part - (like train_part) but specifying the test points.

    returns
    -------     
    train_inputs - the training input matrix
    train_targets - the training targets
    test_inputs - the test input matrix
    test_targets - the test targtets
    """
    # get the indices of the train and test portion
    train_inputs = inputs[train_part,:]
    test_inputs = inputs[test_part,:]
    train_targets = targets[train_part]
    test_targets = targets[test_part]
    return train_inputs, train_targets, test_inputs, test_targets
    

def construct_knn_approx(train_inputs, train_targets, k, test_inputs, test_targets):  
    """
    Outputs the mean training value in the k-Neighbourhood of any input.
    """
    # Create Euclidean distance
    distance = lambda x,y: (x-y)**2   
    train_inputs = train_inputs.transpose() # 11 x 1209
#    print(np.shape(train_inputs)) 
    # 11 x-values of invented data point of which quality should be predicted 
#    test_inputs = np.array([4, 0.7, 0.1, 3, 0.07, 13, 40, 0.9964, 3, 0.56, 10])
    predicts = np.empty(test_targets.size)
#    print(test_targets.size)

    def prediction_function(test_inputs, predicts):
        # Reshape arrays of x-values into 11 x [amount of data points] column vector
        test_inputs = test_inputs.transpose() # 11 x 410
        
        for i in range(test_inputs.shape[1]): # 410 times
#            print("Shape train_inputs:", np.shape(train_inputs))
#            print("Shape of test inputs column: ", np.shape(test_inputs[:, 1]))
            test_inputs_col = test_inputs[:, i].reshape(test_inputs[:, i].size,1)
            # Calculates distance between training data points and test data points to predict (11 x 1599)
            distances = distance(train_inputs, test_inputs_col)
        
            # Sums up all distances per column (axis = 0), so that there's only 1 distance left per data point
            distanceSum = [np.sum(distances, axis = 0)]
            distanceSum = np.array(distanceSum) # not yet 1 x 1209 (training input length)
            distanceSum = distanceSum.reshape((1, distanceSum.size)) # 1 x 1209
#            print("distanceSum shape: ", np.shape(distanceSum))
        
            # All quality values of the data points 
            # TODO: correct the form of train_targets
            ys = train_targets.reshape(train_targets.size,1).T # 1 x 1209
#            print("ys shape: ", np.shape(ys))
            
            # Append the quality values to the distance values (making a 2 x 1209 array "distanceQuality")
            distanceQuality = np.append(distanceSum, ys, axis = 0)
#            print("distanceQuality shape: ", np.shape(distanceQuality))
        
            # Sort array with regard to first row (distanceSum)
            distanceQuality = distanceQuality.transpose() # 1209 x 2
            distanceQSorted = distanceQuality[distanceQuality[:,0].argsort()]
#            print("distanceQSorted shape: ", np.shape(distanceQSorted))
        
            # Average over k-nearest neighbours
            predicts[i] = np.round(np.mean(distanceQSorted[:k,1]),0)
#            print("predicts[i]: ", predicts[i])
            
        predicts = np.array(predicts)
#        print("Shape predicts xx: ", np.shape(predicts))
        print(predicts)
        return predicts
    # We return a handle to the locally defined function
    return prediction_function(test_inputs, predicts)
    
def sum_of_squared_errors(train_targets, predicts, test_targets):
    N = test_targets.size
    mse = np.sum((test_targets.flatten() - predicts.flatten())**2)/N
    return np.sqrt(mse)
    
def test_best_k(train_inputs, train_targets, test_inputs, test_targets, k_range):
    
    SSEs = np.empty(k_range)
    i = 2
    for i in range (k_range):
        k = i + 1
        predicts = construct_knn_approx(train_inputs, train_targets, k, test_inputs, test_targets)
        # collect SSE in array
        SSEs[i] = sum_of_squared_errors(train_targets, predicts, test_targets)
#        print("for loop round:", i)

    
    # Plot errors over different values of k
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    xs = np.array(np.linspace(1, k_range, k_range))
    ys = SSEs
    ax.plot(xs, ys, 'g-', linewidth=3)
    ax.set_xlabel("k")
    ax.set_ylabel("SSEs")
    fig.suptitle('Errors over different values of k')
    
    # Returns index of minimum SSE
    indexMinSSE = np.argmin(SSEs)
    # Returns min SSE
    print("type of SSEs:", type(SSEs))
    minSSE = SSEs[indexMinSSE]
    # Returns optimised k-value
    optK = xs[indexMinSSE]
    
    print("The value for k with the smallest error of %r is %r" % (minSSE, optK) )

    return fig, ax, minSSE, optK    
        
        # show in main fct
        # return k for which SSE is minimized
        # feed that into construct_knn_...


def main(ifname):
    data = import_data(ifname)
    if type(data) == np.ndarray:
        print("Data array loaded: there are %d rows" % data.shape[0])
        print ("first row:", data[0,:])
    
    inputs = data[:,[0,1,2,3,4,5,6,7,8,9,10]]
    targets = data[:,11]
    # get the train test split
    train_part, test_part = train_and_test_split(
        inputs.shape[0], test_fraction=0.25)
    # break the data into train and test parts
    train_inputs, train_targets, test_inputs, test_targets = \
        train_and_test_partition(inputs, targets, train_part, test_part)
    
    print ('Train Inputs: ', np.shape(train_inputs)) # 1209 x 11
    print ('Train Targets: ', np.shape(train_targets)) # 1209,
    print ('Test Inputs: ', np.shape(test_inputs)) #390 x 11
    print( 'Test inputs size: ', test_inputs.size)
    print ('Test Targets: ', np.shape(test_targets)) # 390,
    print( 'Test targets size: ', test_targets.size) # 390
    
    # Find k optimised for smallest error and plot errors over different values for k
    test_best_k(train_inputs, train_targets, test_inputs, test_targets, 15)
    plt.show()
    
    # Plot error over different test and trainings data ratio
    
    
    # Perform knn only with parameters most correlated to quality to reduce parameter amount
    
    
    # Plot error over different amounts of parameters
    

if __name__ == '__main__':
    import sys
    # this bit only runs when this script is called from the command line
    # but not when poly_fit_base.py is used as a library
    main(sys.argv[1])
