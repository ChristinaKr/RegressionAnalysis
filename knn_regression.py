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
        inputs.shape[0], test_fraction=0.1)
    # Break the data into train and test parts
    train_inputs, train_targets, test_inputs, test_targets = \
        train_and_test_partition(inputs, targets, train_part, test_part)
    
    return train_inputs, train_targets, test_inputs, test_targets, inputs

def construct_knn_approx(train_inputs, train_targets, k, test_inputs, test_targets, seed = None):  
    """
    Outputs the mean training value in the k-Neighbourhood of any input.
    """
    if not seed is None:
        np.random.seed(seed)
    
    # Create Euclidean distance
    distance = lambda x,y: (x-y)**2   
    train_inputs = train_inputs.transpose() # 11 x 1209
    predictsRounded = np.empty(test_targets.size)
    predictsNotRounded = np.empty(test_targets.size)

    def prediction_function(test_inputs, predictsRounded, predictsNotRounded):
        # Reshape arrays of x-values into 11 x [amount of data points] column vector
        test_inputs = test_inputs.transpose()
        
        for i in range(test_inputs.shape[1]):
            test_inputs_col = test_inputs[:, i].reshape(test_inputs[:, i].size,1)
            # Calculates distance between training data points and test data points to predict (11 x 1599)
            distances = distance(train_inputs, test_inputs_col)
        
            # Sums up all distances per column (axis = 0), so that there's only 1 distance left per data point
            distanceSum = [np.sum(distances, axis = 0)]
            distanceSum = np.array(distanceSum)
            distanceSum = distanceSum.reshape((1, distanceSum.size))
        
            # All quality values of the data points 
            ys = train_targets.reshape(train_targets.size,1).T
            
            # Append the quality values to the distance values (making a 2 x 1209 array "distanceQuality")
            distanceQuality = np.append(distanceSum, ys, axis = 0)
        
            # Sort array with regard to first row (distanceSum)
            distanceQuality = distanceQuality.transpose() # 1209 x 2
            distanceQSorted = distanceQuality[distanceQuality[:,0].argsort()]
        
            # Average over k-nearest neighbours
            predictsRounded[i] = np.round(np.mean(distanceQSorted[:k,1]),0)
            predictsNotRounded[i] = np.mean(distanceQSorted[:k,1])
            
        predictsRounded = np.array(predictsRounded) # 1599
        predictsNotRounded = np.array(predictsNotRounded) # 1599

        return predictsRounded, predictsNotRounded
    # We return a handle to the locally defined function
    return prediction_function(test_inputs, predictsRounded, predictsNotRounded)
    
def sum_of_squared_errors(train_targets, predicts, test_targets):
    N = test_targets.size
    mse = np.sum((test_targets.flatten() - predicts.flatten())**2)/N
    return np.sqrt(mse)

def calculate_errors_for_different_k(train_inputs, train_targets, test_inputs, test_targets, k_range, seed):

    SSEsRounded = np.empty(k_range)
    SSEsNotRounded = np.empty(k_range)
    for i in range (k_range):
        k = i + 1
        predictsRounded, predictsNotRounded = construct_knn_approx(train_inputs, train_targets, k, test_inputs, test_targets, seed )
        # collect SSE in array
        SSEsRounded[i] = sum_of_squared_errors(train_targets, predictsRounded, test_targets)
        SSEsNotRounded[i] = sum_of_squared_errors(train_targets, predictsNotRounded, test_targets)
        
    return SSEsRounded, SSEsNotRounded
    

def correlation_parameters(inputs, data):
    # correlation between normalised inputs and quality
    targets = data[:,11]
    corr_0 = pearsonr(inputs[:,0], targets)
    corr_1 = pearsonr(inputs[:,1], targets)
    corr_2 = pearsonr(inputs[:,2], targets)
    corr_3 = pearsonr(inputs[:,3], targets)
    corr_4 = pearsonr(inputs[:,4], targets)
    corr_5 = pearsonr(inputs[:,5], targets)
    corr_6 = pearsonr(inputs[:,6], targets)
    corr_7 = pearsonr(inputs[:,7], targets)
    corr_8 = pearsonr(inputs[:,8], targets)
    corr_9 = pearsonr(inputs[:,9], targets)
    corr_10 = pearsonr(inputs[:,10], targets)
    
    correlations = [corr_0[0], corr_1[0], corr_2[0], corr_3[0], corr_4[0], corr_5[0], corr_6[0], corr_7[0], corr_8[0], corr_9[0], corr_10[0]]
    correlations = np.absolute(correlations)
    
    # Find indices of the highest correlating input parameters
    a=0
    indicesHighestCorr = []
    for i in correlations:
        if correlations[a] > 0.2:
            indicesHighestCorr = np.append(indicesHighestCorr, a)
        a+=1
    
    # Create a 2d array filled with the highest correlating input parameters
    inputCorr = np.zeros(shape=((inputs[:,0]).size, indicesHighestCorr.size)) # datapoints x 4
    b=0
    indicesHighestCorr = indicesHighestCorr.astype(int)
    for i in indicesHighestCorr:
        inputCorr[:, b] = inputs[:,indicesHighestCorr[b]]
        b+=1
    
    targetCorr = targets
    return inputCorr, targetCorr  

def error_with_highest_corr_inputs_only(inputCorr, targetCorr):  
    # Randomise training and test data for the highest correlating paramters
    # Get the train test split
    train_part, test_part = train_and_test_split(
        inputCorr.shape[0], test_fraction=0.1)
    # Break the data into train and test parts
    train_inputs, train_targets, test_inputs, test_targets = \
        train_and_test_partition(inputCorr, targetCorr, train_part, test_part)

    # Find k optimised for smallest error and plot errors over different values for k
    SSEsRounded, SSEsNotRounded = calculate_errors_for_different_k(train_inputs, train_targets, test_inputs, test_targets, 20, seed = 28)
    return SSEsRounded, SSEsNotRounded
    
def plot_all_inputs_vs_most_correlated_inputs(data, inputCorr, targetCorr, seed):
    
    runs = 100
    k_range = 20
    SSEs2dCorrInputs = np.zeros(shape=(runs, k_range))
    SSEs2dallInputs = np.zeros(shape=(runs, k_range))
    
    for i in range(runs):
        train_inputs, train_targets, test_inputs, test_targets, inputs = test_and_trainings_data(data)
        SSEsRounded, allInputs = calculate_errors_for_different_k(train_inputs, train_targets, test_inputs, test_targets, 20, None)  
        SSEsRounded, SSEsCorrInputs = error_with_highest_corr_inputs_only(inputCorr, targetCorr)
        SSEs2dCorrInputs[i] = SSEsCorrInputs # 100 x 20
        SSEs2dallInputs[i] = allInputs # 100 x 20

    SSEs2dallInputs = np.mean(SSEs2dallInputs, axis = 0)
    corrInputs = np.mean(SSEs2dCorrInputs, axis = 0)
    
    # Plot it
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    xs = np.linspace(1, k_range, num=k_range)
    ys = SSEs2dallInputs
    allInputs_line, = ax.plot(xs, ys, 'g-', linewidth=3)
    ys = corrInputs
    corrInputs_line, = ax.plot(xs, ys, 'r-', linewidth=3)
    ax.set_xlabel("k")
    ax.set_ylabel("SSEs")
    ax.legend([corrInputs_line, allInputs_line],[" 4 most highly correlated parameters", "All 11 input parameters"])
    fig.suptitle('Errors over different values of k run 100 times - Input parameter comparison') 
    plt.show()
    
    # Find and print the outcomes (smallest error and optimal k)
    indexCorrSSE = np.argmin(corrInputs)
    indexAllPSSE = np.argmin(SSEs2dallInputs)
    minSSECorr = corrInputs[indexCorrSSE]
    minSSEAllP = SSEs2dallInputs[indexAllPSSE] 
    optKCorr = xs[indexCorrSSE]
    optKAllP = xs[indexAllPSSE]
    
    print("The smallest mean error over 100 runs for only the most highly correlated parameters is ", minSSECorr, "with a k of", optKCorr)
    print("The smallest mean error over 100 runs for all parameters included is ", minSSEAllP, "with a k of", optKAllP)
    
def calculate_and_plot_rounded_vs_unrounded_mse(data, seed):
    runs = 100
    k_range = 20
    SSEs2dRounded = np.zeros(shape=(runs, k_range))
    SSEs2dNotRounded = np.zeros(shape=(runs, k_range))
    for i in range(runs):
        train_inputs, train_targets, test_inputs, test_targets, inputs = test_and_trainings_data(data)
        SSEsRounded, SSEsNotRounded = calculate_errors_for_different_k(train_inputs, train_targets, test_inputs, test_targets, 20, None)
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
    fig.suptitle('Errors over different values of k run 100 times - Rounding comparison') 
    plt.show()
    
    # Find and print the outcomes (smallest error and optimal k)
    indexRoundedMeanSSE = np.argmin(SSEsRoundedMean)
    indexNotRoundedMeanSSE = np.argmin(SSEsNotRoundedMean)
    minSSEMeanRounded = SSEsRoundedMean[indexRoundedMeanSSE]
    minSSEMeanNotRounded = SSEsNotRoundedMean[indexNotRoundedMeanSSE] 
    optKMeanRounded = xs[indexRoundedMeanSSE]
    optKMeanNotRounded = xs[indexNotRoundedMeanSSE]
    
    print("The smallest mean error over 100 runs with rounded predictions is ", minSSEMeanRounded, "with a k of", optKMeanRounded)
    print("The smallest mean error over 100 runs with unrounded predictions is ", minSSEMeanNotRounded, "with a k of", optKMeanNotRounded)
     
    
def main(ifname):
    data = import_data(ifname)
    if type(data) == np.ndarray:
        print("Data array loaded: there are %d rows" % data.shape[0])
    
    # Split remaining 85% of data into test and trainings data, inputs are the normalized input values
    train_inputs, train_targets, test_inputs, test_targets, inputs = test_and_trainings_data(data)
    
    # Calculate rounded and unrounded mean error over different values of k run 100 times
    # to check whether it makes a difference to round the predictions as they're discrete values
    # Also plots the errors over different values of k averaged over 100 runs for both - rounded and unrounded
    calculate_and_plot_rounded_vs_unrounded_mse(data, seed = 24)
    
    # Perform knn only with parameters most correlated to quality to reduce parameter amount
    # Plot error over different amounts of parameters
    inputCorr, targetCorr = correlation_parameters(inputs, data)
    error_with_highest_corr_inputs_only(inputCorr, targetCorr)
    plot_all_inputs_vs_most_correlated_inputs (data, inputCorr, targetCorr, seed = 24)
    
    

if __name__ == '__main__':
    import sys
    # this bit only runs when this script is called from the command line
    # but not when poly_fit_base.py is used as a library
    main(sys.argv[1])
