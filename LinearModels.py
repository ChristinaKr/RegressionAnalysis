import csv
import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
import scipy as sp

#from LinearModelsMethods import plot_train_test_errors
from LinearModelsMethods import simple_evaluation_linear_model
#from LinearModelsMethods import exploratory_plots
from LinearModelsMethods import ml_weights
from LinearModelsMethods import train_and_test_partition
from LinearModelsMethods import train_and_test
from regression_models import construct_polynomial_approx

from regression_train_test import create_cv_folds

#from gradientDescent import gradientDescent
#from gradientDescent import computeCost

def main(
        ifname, delimiter=None, columns=None, has_header=True,
        test_fraction=0.25):
    """
    To be called when the script is run. This function creates, fits and plots
    synthetic data, and then fits and plots imported data (if a filename is
    provided). In both cases, data is 2 dimensional real valued data and is fit
    with maximum likelihood 2d gaussian.

    parameters
    ----------
    ifname -- filename/path of data file. 
    delimiter -- delimiter of data values
    has_header -- does the data-file have a header line
    columns -- a list of integers specifying which columns of the file to import
        (counting from 0)    
    """
    # if no file name is provided then use synthetic data
    training_data_as_array, test_data_as_array, field_names = split_data(
            ifname, delimiter=delimiter, has_header=has_header, columns=columns, seed=42)
    exploratory_plots(training_data_as_array, field_names)
    N = training_data_as_array.shape[0]
    inputs = training_data_as_array[:,[0,1,2,3,4,5,6,7,8,9,10]]
    targets = training_data_as_array[:,11]
    norm_inputs = training_data_as_array[:,[0,1,2,3,4,5,6,7,8,9,10]]
    test_inputs = test_data_as_array[:,[0,1,2,3,4,5,6,7,8,9,10]]
    test_targets = test_data_as_array[:,11]
    
    # let's inspect the data a little more
    fixed_acid = norm_inputs[:,0]
    vol_acid = norm_inputs[:,1]
    citric_acid = norm_inputs[:,2]
    resid_sugar = norm_inputs[:,3]
    chlor = norm_inputs[:,4]
    f_SO2 = norm_inputs[:,5]
    t_SO2 = norm_inputs[:,6]
    density = norm_inputs[:,7]
    pH = norm_inputs[:,8]
    sulph = norm_inputs[:,9]
    alco = norm_inputs[:,10]

    '''print("np.mean(fixed_acid) = %r" % (np.mean(fixed_acid),))
    print("np.std(fixed_acid) = %r" % (np.std(fixed_acid),))
    print("np.mean(vol_acid) = %r" % (np.mean(vol_acid),))
    print("np.std(vol_acid) = %r" % (np.std(vol_acid),))
    print("np.mean(citric_acid) = %r" % (np.mean(citric_acid),))
    print("np.std(citric_acid) = %r" % (np.std(citric_acid),))'''
    
    
    # normalise inputs (meaning radial basis functions are more helpful)
    # normalising even for linear can be quicker computationally as scales and units are different for different attributes
    norm_inputs[:,0] = (fixed_acid - np.mean(fixed_acid))/np.std(fixed_acid)
    norm_inputs[:,1] = (vol_acid - np.mean(vol_acid))/np.std(vol_acid)
    norm_inputs[:,2] = (citric_acid - np.mean(citric_acid))/np.std(citric_acid)
    norm_inputs[:,3] = (resid_sugar - np.mean(resid_sugar))/np.std(resid_sugar)
    norm_inputs[:,4] = (chlor - np.mean(chlor))/np.std(chlor)
    norm_inputs[:,5] = (f_SO2 - np.mean(f_SO2))/np.std(f_SO2)
    norm_inputs[:,6] = (t_SO2 - np.mean(t_SO2))/np.std(t_SO2)
    norm_inputs[:,7] = (density - np.mean(density))/np.std(density)
    norm_inputs[:,8] = (pH - np.mean(pH))/np.std(pH)
    norm_inputs[:,9] = (sulph - np.mean(sulph))/np.std(sulph)
    norm_inputs[:,10] = (alco - np.mean(alco))/np.std(alco)
    
    # add col of ones to normalised inputs
    N, D = norm_inputs.shape
    col = np.ones((N,1))
    norm_inputs = np.hstack((col,norm_inputs))
    
    print "Linear Regression with Wine Data:"
    
    # Random split of data into train and test
    # non normalised inputs
    train_errors, test_errors, fig,ax = evaluate_linear_approx(
        inputs, targets, test_fraction)
    
    fig.suptitle('Train and Test Errors of raw data over different values of \n lambda (the regularisation parameter) ')
    
    # use cross validation
    num_folds = 5
    folds = create_cv_folds(N, num_folds)

    # evaluate then plot the performance of different reg params using cross validatation to select
    # non-normalised
    fig, ax = cv_evaluate_reg_param(inputs, targets, folds)

    fig.suptitle('A plot to show how train and test errors of raw data calculated using \n cross validation vary with different regularisation parameters')
    
    print "Linear Regression with normalised Wine Data:"
    
    # Random split of data into train and test
    # normalised 
    train_errors2, test_errors2, fig2,ax2 = evaluate_linear_approx(
        norm_inputs, targets, test_fraction)
    
    fig2.suptitle('Train and Test Errors of normalised data over different values of \n lambda (the regularisation parameter) ')
    
    # using cross evaluation
    # normalised
    fig2, ax2 = cv_evaluate_reg_param(norm_inputs, targets, folds)
    
    fig2.suptitle('A plot to show how train and test errors of normalised data calculated using \n cross validation vary with different regularisation parameters')
    
    # generate train and test errors against the initally seperated 10% of the data
    
    print "Generate test error with the 10% of test data initally seperated 10% of data:"
    
    train_error_final, test_error_final = train_and_test(
        inputs, targets, test_inputs, test_targets,
        reg_param=0.1)
    
    print "    Test error: {} ".format(test_error_final)
    
    # gradient descent 

    # set X (training data) and y (target variable)
    X2 = norm_inputs
    y2 = targets
    
    # convert to matrices and initialize theta
    X2 = np.matrix(X2)
    y2 = (np.matrix(targets)).T
    #m = np.shape(X2)
    theta = np.zeros((1,12))
    theta = np.matrix(theta)
    #theta = theta.T
    #print theta
    alpha = 0.01
    iters = 1000

    # perform linear regression on the data set
    theta, cost2 = gradientDescent(X2, y2, theta, alpha, iters)
    
    # plot
    fig, ax = plt.subplots()
    ax.plot(np.arange(iters), cost2, 'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Error')
    ax.set_title('Error vs Iterations')
    
    # predict    
    quality = (X2 * theta.T)
    errors = np.sum((np.array(quality) - np.array(y2))**2)/len(y2)
    test_error_gd = np.sqrt(errors)
    
    print "Error with gradient descent:", test_error_gd
    
    '''error_gd, fig, ax = calc_gd(inputs, targets)'''
    
    # plot linear approximation for a input variable (the first arg)
    fig, ax = plot_linear_approx(10, inputs, field_names, targets)
    
    plt.show()

def exploratory_plots(data, field_names=None):
    # the number of dimensions in the data
    dim = data.shape[1]
    # create an empty figure object
    fig = plt.figure(figsize=(14,10))
    # create a grid of four axes
    z = 1
    for i in range(dim):
        for j in range(dim):
            ax = fig.add_subplot(4,3,z)            
            # if it is a plot on the diagonal we histogram the data
            if i == j: 
                ax.hist(data[:,i])
                z = z+1
                ax.set_xlabel(field_names[j])
                ax.set_ylabel("Observations")
                ax.set_yticks(ax.get_yticks()[::2])
                ax.set_xticks(ax.get_xticks()[::2])          
    plt.tight_layout()
    
def cv_evaluate_reg_param(inputs, targets, folds, reg_params=None):
    """
      Evaluate then plot the performance of different regularisation parameters
    """
    
    #degree = 1
    # create the feature mappoing and then the design matrix 

    #designmtx = expand_to_monomials(inputs, degree) 
    # choose a range of regularisation parameters
    if reg_params is None:
        reg_params = np.logspace(-15, -4,11)
    num_values = reg_params.size
    num_folds = len(folds)
    # create some arrays to store results
    train_mean_errors = np.zeros(num_values)
    test_mean_errors = np.zeros(num_values)
    train_stdev_errors = np.zeros(num_values)
    test_stdev_errors = np.zeros(num_values)
    #    
    for r, reg_param in enumerate(reg_params):
        # r is the index of reg_param, reg_param is the regularisation parameter
        # cross validate with this regularisation parameter
        train_errors, test_errors = cv_evaluation_linear_model(
            inputs, targets, folds, reg_param=reg_param)
        # we're interested in the average (mean) training and testing errors
        train_mean_error = np.mean(train_errors)
        test_mean_error = np.mean(test_errors)
        train_stdev_error = np.std(train_errors)
        test_stdev_error = np.std(test_errors)
        # store the results
        train_mean_errors[r] = train_mean_error
        test_mean_errors[r] = test_mean_error
        train_stdev_errors[r] = train_stdev_error
        test_stdev_errors[r] = test_stdev_error
    print "Linear Regression with Cross Validation: "
    print ("train mean error: {}, train error standard deviation: {}, test mean error: {}, test error standard deviation: {}".format(train_mean_error, train_stdev_error, test_mean_error, test_stdev_error))

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
    
    #ax.legend([train_line, test_line], ["train", "test"], loc=1)
    ax.set_xscale('log')
    
    return fig, ax

def cv_evaluation_linear_model(
        inputs, targets, folds, reg_param=None):
    """
    Will split inputs and targets into train and test parts, then fit a linear
    model to the training part, and test on the both parts.

    Inputs can be a data matrix (or design matrix), targets should
    be real valued.

    parameters
    ----------
    inputs - the input design matrix (any feature mapping should already be
        applied)
    targets - the targets as a vector
    num_folds - the number of folds
    reg_param (optional) - the regularisation strength. If provided, then
        regularised least squares fitting is uses with this regularisation
        strength. Otherwise, (non-regularised) least squares is used.

    returns
    -------
    train_errors - the training errors for the approximation
    test_errors - the test errors for the approximation
    """
    # get the number of datapoints
    N = inputs.shape[0]
    # get the number of folds
    num_folds = len(folds)
    train_errors = np.empty(num_folds)
    test_errors = np.empty(num_folds)
    for f,fold in enumerate(folds):
        # f is the fold id, fold is the train-test split
        train_part, test_part = fold
        # break the data into train and test sets
        train_inputs, train_targets, test_inputs, test_targets = \
            train_and_test_partition(inputs, targets, train_part, test_part)
        # now train and evaluate the error on both sets
        train_error, test_error = train_and_test(
            train_inputs, train_targets, test_inputs, test_targets,
            reg_param=reg_param)
        #print("train_error = %r" % (train_error,))
        #print("test_error = %r" % (test_error,))
        train_errors[f] = train_error
        test_errors[f] = test_error
    return train_errors, test_errors

def plot_linear_approx(i, inputs, field_names,targets):
    
    ''' 
    i = the attribute column number 
    
    '''
    
    degree = 1
    
    attribute = inputs[:,i]
    field_name_i = field_names[i]
    
    # convert our inputs into a matrix where each row
    # is a vector of monomials of the corresponding input
    processed_inputs = expand_to_monomials(attribute, degree)
    #
    # find the weights that fit the data in a least squares way
    weights = ml_weights(processed_inputs, targets)
    # use weights to create a function that takes inputs and returns predictions
    # in python, functions can be passed just like any other object
    # those who know MATLAB might call this a function handle
    linear_approx = construct_polynomial_approx(degree, weights)
    fig, ax = plot_function_data_and_approximation(
        linear_approx, attribute, targets)
    ax.legend(['data', 'linear approx'])
    ax.set_xlabel(field_name_i)
    ax.set_ylabel('Quality')
    ax.set_title('A simple linear approximation of \n how quality changes with {}'.format(field_name_i))
    ax.set_ylim( 2.5, 8.5 )
    #ax.set_xticks([])
    #ax.set_yticks([])
    fig.tight_layout()
    
    return fig,ax

# this is the feature mapping for a polynomial of given degree in 1d
def expand_to_monomials(inputs, degree):
    """
    Create a design matrix from a 1d array of input values, where columns
    of the output are powers of the inputs from 0 to degree (inclusive)

    So if input is: inputs=np.array([x1, x2, x3])  and degree = 4 then
    output will be design matrix:
        np.array( [[  1.    x1**1   x1**2   x1**3   x1**4   ]
                   [  1.    x2**1   x2**2   x2**3   x2**4   ]
                   [  1.    x3**1   x3**2   x3**3   x3**4   ]])
    """
    expanded_inputs = []
    for i in range(degree+1):
        expanded_inputs.append(inputs**i)
    return np.array(expanded_inputs).transpose()

def plot_function_and_data(inputs, targets, markersize=5, **kwargs):
    """
    Plot a function and some associated regression data in a given range

    parameters
    ----------
    inputs - the input data
    targets - the targets
    markersize (optional) - the size of the markers in the plotted data
    <for other optional arguments see plot_function>

    returns
    -------
    fig - the figure object for the plot
    ax - the axes object for the plot
    lines - a list of the line objects on the plot
    """
    #fig, ax, lines = plot_function(true_func)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(inputs, targets, 'bo', markersize=markersize)
    return fig, ax

def plot_function_data_and_approximation(
        predict_func, inputs, targets, linewidth=3, xlim=None,
        **kwargs):
    """
    Plot a function, some associated regression data and an approximation
    in a given range

    parameters
    ----------
    predict_func - the approximating function
    inputs - the input data
    targets - the targets
    <for optional arguments see plot_function_and_data>

    returns
    -------
    fig - the figure object for the plot
    ax - the axes object for the plot
    lines - a list of the line objects on the plot
    """
    if xlim is None:
        xlim = (0,1)
    fig, ax = plot_function_and_data(
        inputs, targets, linewidth=linewidth, xlim=xlim, **kwargs)
    xs = np.linspace(min(inputs), max(inputs), 101)
    ys = predict_func(xs)
    ax.plot(xs, ys, 'r-', linewidth=linewidth)
    return fig, ax

def evaluate_linear_approx(inputs, targets, test_fraction):
    '''
    This evaluates the linear performance of the data. 
    This takes in an input variable of runs - this is the number of times the errors are calculated for each reg param,
        this is then averaged and the mean is plot, therefore producing a smooth and accurate curve.
    '''
    # the linear performance
    reg_params = np.logspace(-15,-4, 11)
    train_errors = []
    test_errors = []
    
    # plot the average for a number of runs
    for reg_param in reg_params:
        #print("Evaluating reg_para " + str(reg_param))
        train_error, test_error = simple_evaluation_linear_model(
            inputs, targets, test_fraction=test_fraction, reg_param=reg_param)
        train_errors.append(train_error)
        test_errors.append(test_error)
        # once errors calculated for no. of runs, calculate average, append and clear array for next reg param
    
    fig , ax = plot_train_test_errors(
        "$\lambda$", reg_params, train_errors, test_errors)
    # we also want to plot a straight line showing the linear performance
    xlim = ax.get_xlim()
    #test, = ax.plot(xlim, test_error*np.ones(2), ':g', label='test')
    #ax.legend([train_line, test_line, test], ["train", "test", "test error"], loc=1)
    ax.set_xscale('log')
    
    print("Linear Regression:")
    print("\t(train_error, test_error) = %r" % ((train_error, test_error),))
    return train_errors, test_errors, fig, ax

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
    ax.legend([train_line, test_line], ["train", "test"], loc=1)
    return fig, ax

def predict_wine(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,theta):
    x_i = sp.matrix([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11])
    x_n = (x_i - np.mean(x_i))/np.std(x_i) # normalise
    x = sp.hstack((sp.matrix([1]), x_n))
    quality = (x * theta.T)
    print ("Prediction using Gradient Descent: \n    Quality of wine with fixed acidity {}, volitile acidity {}, citric acid {}, residual sugar {}, chlorides {}, free sulfur dioxide {}, total sulfur dioxide {}, density {}, pH {}, sulphates {} and alcohol {} has a predicted quality of {}".format(x1, x2, x3,x4,x5,x6,x7,x8,x9,x10,x11,quality))

def gradientDescent(X, y, theta2, alpha, iters):
    
    #theta2 = theta2.T  
    #print "theta2: ", theta2
    
    temp = np.matrix(np.zeros(theta2.shape))
    parameters = int(theta2.ravel().shape[1])
    cost = np.zeros(iters)

    for i in range(iters):
        error = (X * theta2.T) - y

        for j in range(parameters):
            term = np.multiply(error, X[:,j])
            temp[0,j] = theta2[0,j] - ((alpha / len(X)) * np.sum(term))

        theta2 = temp
        cost[i] = computeCost(X, y, theta2)

    return theta2, cost

def computeCost(X, y, theta2): 
    #theta2 = theta2.T   
    inner = np.power(((X * theta2.T) - y), 2)
    return np.sum(inner) / (2 * len(X))
    
def split_data(ifname, delimiter=None, has_header=False, columns=None, seed=42, fraction=0.15):
    """
    Imports a tab/comma/semi-colon/... separated data file as an array of 
    floating point numbers. If the import file has a header then this should
    be specified, and the field names will be returned as the second argument.

    parameters
    ----------
    ifname -- filename/path of data file.
    delimiter -- delimiter of data values
    has_header -- does the data-file have a header line
    columns -- a list of integers specifying which columns of the file to import
        (counting from 0)

    returns
    -------
    data_as_array -- the data as a numpy.array object  
    field_names -- if file has header, then this is a list of strings of the
      the field names imported. Otherwise, it is a None object.
    """
    '''if not seed is None:
        np.random.seed(seed)
    if delimiter is None:
        delimiter = '\t'
    with open(ifname, 'r') as ifile:
        datareader = csv.reader(ifile, delimiter=delimiter)
        # if the data has a header line we want to avoid trying to import it.
        # instead we'll print it to screen
        if has_header:
            field_names = next(datareader)
            #print("Importing data with field_names:\n\t" + ",".join(field_names))
        else:
            # if there is no header then the field names is a dummy variable
            field_names = None
        # create an empty list to store each row of data
        data = []
        for row in datareader:
            # print("row = %r" % (row,))
            # for each row of data only take the columns we are interested in
            if not columns is None:
                row = [row[c] for c in columns]
            # now store in our data list
            data.append(row)
        #print("There are %d entries" % len(data))
        #print("Each row has %d elements" % len(data[0]))
    # convert the data (list object) into a numpy array.
    data_as_array = np.array(data).astype(float)
    if not columns is None and not field_names is None:
        # thin the associated field names if needed
        field_names = [field_names[c] for c in columns]
    # return this array to caller (and field_names if provided)
    return data_as_array, field_names'''
    
    np.random.seed(seed)
    test_rows = np.unique(np.array(np.random.uniform(size = int(fraction*1599))*1599).astype(int))
    #print(test_rows)
    
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
    return training_data_as_array, test_data_as_array, field_names


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
        main() # calls the main function with no arguments
    elif len(sys.argv) == 2:
        # assumes that the first argument is the input filename/path
        main(ifname=sys.argv[1])
    elif len(sys.argv) == 3:
        # assumes that the second argument is the data delimiter
        main(ifname=sys.argv[1], delimiter=sys.argv[2])
    elif len(sys.argv) == 4:
        # assumes that the third argument is the list of columns to import
        columns = list(map(int, sys.argv[3].split(","))) 
        main(ifname=sys.argv[1], delimiter=sys.argv[2], columns=columns)

