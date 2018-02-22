#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 10:17:38 2018

@author: christinakronser
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr   

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
    

def construct_knn_approx(train_inputs, train_targets, k, test_inputs, test_targets, seed = None):  
    """
    Outputs the mean training value in the k-Neighbourhood of any input.
    """
    if not seed is None:
        np.random.seed(seed)
    
    # Create Euclidean distance
    distance = lambda x,y: (x-y)**2   
    train_inputs = train_inputs.transpose() # 11 x 1209
#    print(np.shape(train_inputs)) 
    # 11 x-values of invented data point of which quality should be predicted 
#    test_inputs = np.array([4, 0.7, 0.1, 3, 0.07, 13, 40, 0.9964, 3, 0.56, 10])
    predictsRounded = np.empty(test_targets.size)
    predictsNotRounded = np.empty(test_targets.size)

    def prediction_function(test_inputs, predictsRounded, predictsNotRounded):
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
            predictsRounded[i] = np.round(np.mean(distanceQSorted[:k,1]),0)
#            print("predicts[i]: ", predicts[i])
            predictsNotRounded[i] = np.mean(distanceQSorted[:k,1])
            
        predictsRounded = np.array(predictsRounded)
#        print(predictsRounded)
        predictsNotRounded = np.array(predictsNotRounded)
#        print(predictsNotRounded)
#        print("Shape predicts xx: ", np.shape(predicts))
#        print(predicts)
        return predictsRounded, predictsNotRounded
    # We return a handle to the locally defined function
    return prediction_function(test_inputs, predictsRounded, predictsNotRounded)
    
def sum_of_squared_errors(train_targets, predicts, test_targets):
    N = test_targets.size
    mse = np.sum((test_targets.flatten() - predicts.flatten())**2)/N
    return np.sqrt(mse)
    
def test_best_k(train_inputs, train_targets, test_inputs, test_targets, k_range, seed):
    seed = seed
    SSEsRounded = np.empty(k_range)
    SSEsNotRounded = np.empty(k_range)
    for i in range (k_range):
        k = i + 1
        predictsRounded, predictsNotRounded = construct_knn_approx(train_inputs, train_targets, k, test_inputs, test_targets, seed )
        # collect SSE in array
#        print("predictsRounded: ", predictsRounded)
#        print("predictsNotRounded: ", predictsNotRounded)
        SSEsRounded[i] = sum_of_squared_errors(train_targets, predictsRounded, test_targets)
        SSEsNotRounded[i] = sum_of_squared_errors(train_targets, predictsNotRounded, test_targets)
#        print("for loop round:", i)

    print("SSEsRounded: ", SSEsRounded)
    print("SSEsNotRounded: ", SSEsNotRounded)
    # Plot errors over different values of k
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    xs = np.array(np.linspace(1, k_range, num=k_range))
    ys = SSEsRounded
    rounded_SSE_line, = ax.plot(xs, ys, 'g-', linewidth=3)
    ys = SSEsNotRounded
    notRounded_SSE_line, = ax.plot(xs, ys, 'r-', linewidth=3)
    ax.set_xlabel("k")
    ax.set_ylabel("SSEs")
    ax.legend([rounded_SSE_line, notRounded_SSE_line],["rounded predictions", "not rounded predictions"])
    fig.suptitle('Errors over different values of k')
    
    # Returns index of minimum SSE
    indexRoundedSSE = np.argmin(SSEsRounded)
    indexNotRoundedSSE = np.argmin(SSEsNotRounded)
    # Returns min SSE
#    print("type of SSEs:", type(SSEsRounded))
    minSSERounded = SSEsRounded[indexRoundedSSE]
    minSSENotRounded = SSEsNotRounded[indexNotRoundedSSE]
    
    # Returns optimised k-value
    optKRounded = xs[indexRoundedSSE]
    optKNotRounded = xs[indexNotRoundedSSE]
    
    print("The rounded value for k with the smallest error of %r is %r" % (minSSERounded, optKRounded) )
    print("The unrounded value for k with the smallest error of %r is %r" % (minSSENotRounded, optKNotRounded) )
    
    if minSSERounded < minSSENotRounded:
        minSSE = minSSERounded
        optK = optKRounded
    else:
        minSSE = minSSENotRounded
        optK = optKNotRounded


    return fig, ax, minSSE, optK    
        
        # show in main fct
        # return k for which SSE is minimized
        # feed that into construct_knn_...

def plot_errors_for_different_k(train_inputs, train_targets, test_inputs, test_targets, k_range, seed):
#    range = 100
#    SSEsRoundedArray = np.empty(range)
#    j = 0
#    for i in range(range):
#        seed = seed
    SSEsRounded = np.empty(k_range)
    SSEsNotRounded = np.empty(k_range)
    for i in range (k_range):
        k = i + 1
        predictsRounded, predictsNotRounded = construct_knn_approx(train_inputs, train_targets, k, test_inputs, test_targets, seed )
        # collect SSE in array
#        print("predictsRounded: ", predictsRounded)
#        print("predictsNotRounded: ", predictsNotRounded)
        SSEsRounded[i] = sum_of_squared_errors(train_targets, predictsRounded, test_targets)
        SSEsNotRounded[i] = sum_of_squared_errors(train_targets, predictsNotRounded, test_targets)
#        print("for loop round:", i)
        
    return SSEsRounded, SSEsNotRounded
    
  

def test_and_trainings_data(data):
    
    inputs = data[:,[0,1,2,3,4,5,6,7,8,9,10]]
    targets = data[:,11]
    
    # Prepare the data for normalisation
    fixed_acidity_inputs = inputs[:,0]
    volatile_acidity_inputs = inputs[:,1]
    citric_acid_inputs = inputs[:,2]
    residual_sugar_inputs = inputs[:,3]
    chlorides_inputs = inputs[:,4]
    free_sulfur_dioxide_inputs = inputs[:,5]
    total_sulfur_dioxide_inputs = inputs[:,6]
    density_inputs = inputs[:,7]
    pH_inputs = inputs[:,8]
    sulphates_inputs = inputs[:,9]
    alcohol_inputs = inputs[:,10]
    
    # Normalise inputs
    inputs[:,0] = (fixed_acidity_inputs - np.mean(fixed_acidity_inputs))/np.std(fixed_acidity_inputs)
    inputs[:,1] = (volatile_acidity_inputs - np.mean(volatile_acidity_inputs))/np.std(volatile_acidity_inputs)
    inputs[:,2] = (citric_acid_inputs - np.mean(citric_acid_inputs))/np.std(citric_acid_inputs)
    inputs[:,3] = (residual_sugar_inputs - np.mean(residual_sugar_inputs))/np.std(residual_sugar_inputs)
    inputs[:,4] = (chlorides_inputs - np.mean(chlorides_inputs))/np.std(chlorides_inputs)
    inputs[:,5] = (free_sulfur_dioxide_inputs - np.mean(free_sulfur_dioxide_inputs))/np.std(free_sulfur_dioxide_inputs)
    inputs[:,6] = (total_sulfur_dioxide_inputs - np.mean(total_sulfur_dioxide_inputs))/np.std(total_sulfur_dioxide_inputs)
    inputs[:,7] = (density_inputs - np.mean(density_inputs))/np.std(density_inputs)
    inputs[:,8] = (pH_inputs - np.mean(pH_inputs))/np.std(pH_inputs)
    inputs[:,9] = (sulphates_inputs - np.mean(sulphates_inputs))/np.std(sulphates_inputs)
    inputs[:,10] = (alcohol_inputs - np.mean(alcohol_inputs))/np.std(alcohol_inputs)
    
    # Get the train test split
    train_part, test_part = train_and_test_split(
        inputs.shape[0], test_fraction=0.25)
    # Break the data into train and test parts
    train_inputs, train_targets, test_inputs, test_targets = \
        train_and_test_partition(inputs, targets, train_part, test_part)
    
    return train_inputs, train_targets, test_inputs, test_targets, inputs

def correlation_parameters(inputs, data):
    # correlation mtx 
    targets = data[:,11]
    print("Correlation fixed acidity with quality: ", pearsonr(inputs[:,0], targets))
    
    



def main(ifname):
    data = import_data(ifname)
    if type(data) == np.ndarray:
        print("Data array loaded: there are %d rows" % data.shape[0])
        print ("first row:", data[0,:])
        
    train_inputs, train_targets, test_inputs, test_targets, inputs = test_and_trainings_data(data)

    # Find k optimised for smallest error and plot errors over different values for k
    test_best_k(train_inputs, train_targets, test_inputs, test_targets, 50, seed = 28)
    plt.show()
    
    # Calculate mean rounded and unrounded error over different values of k run 100 times
    runs = 100
    k_range = 20
    SSEs2dRounded = np.zeros(shape=(runs, k_range))
    SSEs2dNotRounded = np.zeros(shape=(runs, k_range))
    for i in range(runs):
        train_inputs, train_targets, test_inputs, test_targets, inputs = test_and_trainings_data(data)
        SSEsRounded, SSEsNotRounded = plot_errors_for_different_k(train_inputs, train_targets, test_inputs, test_targets, 20, None)
        SSEs2dRounded[i] = SSEsRounded # 100 x 20
        SSEs2dNotRounded[i] = SSEsNotRounded # 100 x 20
    
    SSEsRoundedMean = np.mean(SSEs2dRounded, axis = 0)
    SSEsNotRoundedMean = np.mean(SSEs2dNotRounded, axis = 0)
    
    # Plot it
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    xs = np.linspace(1, k_range, num=k_range)
    ys = SSEsRoundedMean
    rounded_SSE_line, = ax.plot(xs, ys, 'g-', linewidth=3)
    ys = SSEsNotRoundedMean
    notRounded_SSE_line, = ax.plot(xs, ys, 'r-', linewidth=3)
    ax.set_xlabel("k")
    ax.set_ylabel("SSEs")
    ax.legend([rounded_SSE_line, notRounded_SSE_line],["rounded predictions", "not rounded predictions"])
    fig.suptitle('Errors over different values of k run 100 times') 
    plt.show()
    
    # Returns index of minimum SSE
    indexRoundedMeanSSE = np.argmin(SSEsRoundedMean)
    indexNotRoundedMeanSSE = np.argmin(SSEsNotRoundedMean)
    # Returns min SSE
    minSSEMeanRounded = SSEsRoundedMean[indexRoundedMeanSSE]
    minSSEMeanNotRounded = SSEsNotRoundedMean[indexNotRoundedMeanSSE] 
    # Returns optimised k-value
    optKMeanRounded = xs[indexRoundedMeanSSE]
    optKMeanNotRounded = xs[indexNotRoundedMeanSSE]
    print("shape of minSSEMeanRounded: ", np.shape(minSSEMeanRounded))
    
    print("The smallest rounded mean error over 100 runs is ", minSSEMeanRounded, " with a k of ", optKMeanRounded, ".")
    print("The smallest unrounded mean error over 100 runs is ", minSSEMeanNotRounded, " with a k of ", optKMeanNotRounded, ".")
    
    # Perform knn only with parameters most correlated to quality to reduce parameter amount
    correlation_parameters(inputs, data)
    
    # Plot error over different amounts of parameters
    

if __name__ == '__main__':
    import sys
    # this bit only runs when this script is called from the command line
    # but not when poly_fit_base.py is used as a library
    main(sys.argv[1])
