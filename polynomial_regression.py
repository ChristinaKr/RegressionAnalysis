import csv
import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt


# for fitting
from regression_models import ml_weights
from regression_models import regularised_ml_weights
from regression_models import linear_model_predict

#for evaluating
from regression_train_test import simple_evaluation_linear_model
from regression_train_test import train_and_test
from regression_train_test import cv_evaluation_linear_model

def split_data(ifname, delimiter=None, has_header=False, columns=None, seed=42, fraction=0.15):
    """
    Imports a tab/comma/semi-colon/... separated data file as an array of 
    floating point numbers.

    parameters
    ----------
    ifname -- filename/path of data file.
    delimiter -- delimiter of data values
    has_header -- does the data-file have a header line
    columns -- a list of integers specifying which columns of the file to import
        (counting from 0)
    seed -- the seed for the pseudo-random number generator
    fraction -- the fraction of the data used for testing (between 0 and 1)

    returns
    -------
    training_data_as_array -- the training data as a numpy.array object
    test_data_as_array --  the test data as a numpy.array object 
    """
    
    np.random.seed(seed)
    test_rows = np.unique(np.array(np.random.uniform(size = int(fraction*1599))*1599).astype(int))

    if delimiter is None:
        delimiter = '\t'
    with open(ifname, 'r') as ifile:
        datareader = csv.reader(ifile, delimiter=delimiter)
        # if the data has a header line we want to avoid trying to import it.
        if has_header:
            field_names = next(datareader)
        # create empty lists to store each row of data
        training_data = []
        test_data = []
        count = 0
        for row in datareader:
            # for each row of data only take the columns we are interested in
            if not columns is None:
                row = [row[c] for c in columns]
            # now store in our data lists
            if(count in test_rows):
                test_data.append(row)
            else:
                training_data.append(row)
            count+=1
    #print("There are %d training entries" % len(training_data))
    #print("There are %d test entries" % len(test_data))
    #print("Each row has %d elements" % len(training_data[0]))
    # convert the data (list object) into a numpy array.
    training_data_as_array = np.array(training_data).astype(float)
    test_data_as_array = np.array(test_data).astype(float)
    # return the two data sets to caller
    return training_data_as_array, test_data_as_array

def normalize(data):
    for i in range (1,len(data[0])):
        data[:,i] = (data[:,i] - np.mean(data[:,i]))/np.std(data[:,i])
    return data

def second_degree_designmtx(data):
    """
    Converts the raw data matrix into a designmtx 
    for second degree polynomial regression.
        
    parameters
    ----------
    data -- the data matrix as numpy array
    
    returns
    -------
    designmtx -- the design matrix as a numpy array
    """
    
    N = len(data[0])
    designmtx = []
    for row in data:
        temp=[]
        for i in range (0,N):
            for j in range(i,N):
                temp.append(row[i]*row[j])
        designmtx.append(np.append(row,temp))
    designmtx = np.array(designmtx)
    designmtx = np.hstack([np.ones((designmtx.shape[0],1)), designmtx])
    return designmtx

def third_degree_designmtx(data):
    """
    Converts the raw data matrix into a designmtx 
    for third degree polynomial regression.
        
    parameters
    ----------
    data -- the data matrix as numpy array
    
    returns
    -------
    designmtx -- the design matrix as a numpy array
    """
    data = np.hstack([np.ones((data.shape[0],1)), data])
    N = len(data[0])
    designmtx = []
    
    for row in data:
        temp=[]
        for i in range (0,N):
            for j in range(i,N):
                for k in range(j,N):
                    temp.append(row[i]*row[j]*row[k])
        designmtx.append(temp)
    designmtx = np.array(designmtx)
    return designmtx

def fourth_degree_designmtx(data):
    """
    Converts the raw data matrix into a designmtx 
    for third degree polynomial regression.
        
    parameters
    ----------
    data -- the data matrix as numpy array
    
    returns
    -------
    designmtx -- the design matrix as a numpy array
    """
    data = np.hstack([np.ones((data.shape[0],1)), data])
    N = len(data[0])
    designmtx = []
    
    for row in data:
        temp=[]
        for i in range (0,N):
            for j in range(i,N):
                for k in range(j,N):
                    for l in range(k,N):
                        temp.append(row[i]*row[j]*row[k]*row[l])
        designmtx.append(temp)
    designmtx = np.array(designmtx)
    return designmtx


def evaluate_reg_param(designmtx, targets, folds, reg_params=None):
    """
      Evaluate then plot the performance of different regularisation parameters
    """

    # choose a range of regularisation parameters
    if reg_params is None:
        reg_params = np.logspace(-2,0)
    num_values = reg_params.size
    num_folds = len(folds)
    # create some arrays to store results
    train_mean_errors = np.zeros(num_values)
    test_mean_errors = np.zeros(num_values)
    train_stdev_errors = np.zeros(num_values)
    test_stdev_errors = np.zeros(num_values)
    #
    min_value = 1000000
    optimal_reg = 0
    for r, reg_param in enumerate(reg_params):
        # r is the index of reg_param, reg_param is the regularisation parameter
        # cross validate with this regularisation parameter
        train_errors, test_errors = cv_evaluation_linear_model(
            designmtx, targets, folds, reg_param=reg_param)
        # we're interested in the average (mean) training and testing errors
        train_mean_error = np.mean(train_errors)
        test_mean_error = np.mean(test_errors)
        train_stdev_error = np.std(train_errors)
        test_stdev_error = np.std(test_errors)
        # store the results
        train_mean_errors[r] = train_mean_error
        if (test_mean_error < min_value):
            min_value = test_mean_error
            optimal_reg = reg_param
        test_mean_errors[r] = test_mean_error
        train_stdev_errors[r] = train_stdev_error
        test_stdev_errors[r] = test_stdev_error
    print("Min train error: ", min(train_mean_errors))
    print("Min test error: ", min(test_mean_errors)," for lambda = ", optimal_reg)
    # Now plot the results
    fig, ax = plot_train_test_errors(
        "$\lambda$", reg_params, train_mean_errors, test_mean_errors)
    # Here we plot the error ranges too: mean plus/minus 1 standard error.
    # 1 standard error is the standard deviation divided by sqrt(n) where
    # n is the number of samples. 
    # (There are other choices for error bars.)
    # train error bars
    lower = train_mean_errors - train_stdev_errors/np.sqrt(num_folds)
    upper = train_mean_errors + train_stdev_errors/np.sqrt(num_folds)
    ax.fill_between(reg_params, lower, upper, alpha=0.2, color='b')
    # test error bars
    lower = test_mean_errors - test_stdev_errors/np.sqrt(num_folds)
    upper = test_mean_errors + test_stdev_errors/np.sqrt(num_folds)
    ax.fill_between(reg_params, lower, upper, alpha=0.2, color='r')
    ax.set_xscale('log')
    return optimal_reg

def plot_train_test_errors(
        control_var, experiment_sequence, train_errors, test_errors):
    """
    Plot the train and test errors for a sequence of experiments.

    parameters
    ----------
    control_var - the name of the control variable, e.g. degree (for polynomial)
        degree.
    experiment_sequence - a list of values applied to the control variable.
    """
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    train_line, = ax.plot(experiment_sequence, train_errors,'b-')
    test_line, = ax.plot(experiment_sequence, test_errors, 'r-')
    ax.set_xlabel(control_var)
    ax.set_ylabel("$E_{RMS}$")
    ax.legend([train_line, test_line], ["train", "test"])
    return fig, ax

def create_cv_folds(N, num_folds):
    """
    Defines the cross-validation splits for N data-points into num_folds folds.
    Returns a list of folds, where each fold is a train-test split of the data.
    Achieves this by partitioning the data into num_folds (almost) equal
    subsets, where in the ith fold, the ith subset will be assigned to testing,
    with the remaining subsets assigned to training.

    parameters
    ----------
    N - the number of datapoints
    num_folds - the number of folds

    returns
    -------
    folds - a sequence of num_folds folds, each fold is a train and test array
        indicating (with a boolean array) whether a datapoint belongs to the
        training or testing part of the fold.
        Each fold is a (train_part, test_part) pair where:

        train_part - a boolean vector of length N, where if ith element is
            True if the ith data-point belongs to the training set, and False if
            otherwise.
        test_part - a boolean vector of length N, where if ith element is
            True if the ith data-point belongs to the testing set, and False if
            otherwise.
    """
    # if the number of datapoints is not divisible by folds then some parts
    # will be larger than others (by 1 data-point). min_part is the smallest
    # size of a part (uses integer division operator //)
    min_part = N//num_folds
    # rem is the number of parts that will be 1 larger
    rem = N % num_folds
    # create an empty array which will specify which part a datapoint belongs to 
    parts = np.empty(N, dtype=int)
    start = 0
    for part_id in range(num_folds):
        # calculate size of the part
        n_part = min_part
        if part_id < rem:
            n_part += 1
        # now assign the part id to a block of the parts array
        parts[start:start+n_part] = part_id*np.ones(n_part)
        start += n_part
    # now randomly reorder the parts array (so that each datapoint is assigned
    # a random part.
    np.random.shuffle(parts)
    # we now want to turn the parts array, into a sequence of train-test folds
    folds = []
    for f in range(num_folds):
        train = (parts != f)
        test = (parts == f)
        folds.append((train,test))
    return folds
    
def main(ifname='winequality-red.csv', delimiter=';', has_header=True, folds = 10):
    
    Train_Data, Test_Data = split_data(ifname, delimiter, has_header, [0,1,2,3,4,5,6,7,8,9,10])
    Train_Targets, Test_Targets = split_data(ifname, delimiter, has_header, [11])
    Train_Second = second_degree_designmtx(Train_Data)
    Test_Data_Second = second_degree_designmtx(Test_Data)
    Train_Third = third_degree_designmtx(Train_Data)
    Test_Data_Third = third_degree_designmtx(Test_Data)
    Train_Fourth = fourth_degree_designmtx(Train_Data)
    Test_Data_Fourth = fourth_degree_designmtx(Test_Data)
    
    #print("Second degree polynomial regression")
    #evaluate_reg_param(Train_Second, Train_Targets, create_cv_folds(len(Train_Second), folds), reg_params = np.logspace(-2,5))
    #print("Third degree polynomial regression")
    #evaluate_reg_param(Train_Third, Train_Targets, create_cv_folds(len(Train_Third), folds), reg_params = np.logspace(2,7))
    #print("Fourth degree polynomial regression")
    #evaluate_reg_param(Train_Fourth, Train_Targets, create_cv_folds(len(Train_Third), folds), reg_params = np.logspace(6,10))
    
    
    #print("Second degree polynomial regression (normalized)")
    reg_second = evaluate_reg_param(normalize(Train_Second), Train_Targets, create_cv_folds(len(Train_Second), folds), reg_params = np.logspace(-1,3))
    #print("Third degree polynomial regression (normalized)")
    reg_third = evaluate_reg_param(normalize(Train_Third), Train_Targets, create_cv_folds(len(Train_Third), folds), reg_params = np.logspace(-1,3))
    #print("Fourth degree polynomial regression (normalized)")
    reg_fourth =evaluate_reg_param(normalize(Train_Fourth), normalize(Train_Targets), create_cv_folds(len(Train_Third), folds), reg_params = np.logspace(-1,3))
    
    print("Train, Test error (degree 2): ", train_and_test(normalize(Train_Second), Train_Targets, normalize(Test_Data_Second), Test_Targets, reg_param = reg_second))
    
    print("Train, Test error (degree 3): ", train_and_test(normalize(Train_Third), Train_Targets, normalize(Test_Data_Third), Test_Targets, reg_param = reg_third))
          
    print("Train, Test error (degree 4): ", train_and_test(normalize(Train_Fourth), Train_Targets, normalize(Test_Data_Fourth), Test_Targets, reg_param = reg_fourth))


if __name__ == '__main__':
    """
    To run this script on just synthetic data use:

        python regression_external_data.py

    You can pass the data-file name as the first argument when you call
    your script from the command line. E.g. use:

        python regression_external_data.py datafile.tsv

    If you pass a second argument it will be taken as the delimiter, e.g.
    for comma separated values:

        python regression_external_data.py comma_separated_data.csv ","

    for semi-colon separated values:

        python regression_external_data.py comma_separated_data.csv ";"

    If your data has more than 2 columns you must specify which columns
    you wish to plot as a comma separated pair of values, e.g.

        python regression_external_data.py comma_separated_data.csv ";" 8,9

    For the wine quality data you will need to specify which columns to pass.
    """
    import sys
    if len(sys.argv) == 1:
        # calls the main function with no arguments
        main() 
    elif len(sys.argv) == 2:
        # assumes that the first argument is the input filename/path
        main(ifname=sys.argv[1])
    elif len(sys.argv) == 3:
        # assumes that the second argument is the data delimiter
        main(ifname=sys.argv[1], delimiter=sys.argv[2])
    elif len(sys.argv) == 4:
        # assumes that the third argument is the has_header boolean value
        main(ifname=sys.argv[1], delimiter=sys.argv[2], has_header=sys.argv[3])
    elif len(sys.argv) == 5:
        # assumes that the fourth argument is the list of columns to import
        columns = list(map(int, sys.argv[4].split(","))) 
        main(ifname=sys.argv[1], delimiter=sys.argv[2], has_header=sys.argv[3], columns=columns)
    elif len(sys.argv) == 6:
        # assumes that the fifth argument is the number of folds for the cross-validation
        columns = list(map(int, sys.argv[4].split(","))) 
        main(ifname=sys.argv[1], delimiter=sys.argv[2], has_header=sys.argv[3], columns=columns, folds=sys.argv[5])
