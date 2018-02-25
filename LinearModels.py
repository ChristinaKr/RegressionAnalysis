import csv
import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
import scipy as sp

def linear_models(
        training_data_as_array,test_data_as_array, test_fraction):
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
    #training_data_as_array, test_data_as_array, field_names = split_data(
            #ifname, delimiter=delimiter, has_header=has_header, columns=columns, seed=42)
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
    train_errors, test_errors, train_sd, test_sd, fig,ax = evaluate_linear_approx(
        inputs, targets, test_fraction)
    
    fig.suptitle('Train and Test Errors of raw data over different values of \n lambda (the regularisation parameter), averaged over 1000 runs ')
    
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
    train_errors2, test_errors2, train_sd2, test_sd2,  fig2,ax2 = evaluate_linear_approx(
        norm_inputs, targets, test_fraction)
    
    fig2.suptitle('Train and Test Errors of normalised data over different values of \n lambda (the regularisation parameter), averaged over 1000 runs')
    
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

    # set inputs and targets 
    grad_inputs = norm_inputs
    grad_targets = targets
    
    # convert to matrices, initialise theta
    grad_inputs = np.matrix(grad_inputs)
    grad_targets = (np.matrix(targets)).T
    theta = np.zeros((1,12))
    theta = np.matrix(theta)
    alpha = 0.01
    iters = 1000

    # perform linear regression on the data set
    theta, cost2 = gradientDescent(grad_inputs, grad_targets, theta, alpha, iters)
    
    # plot error vs iterations to see the minimisation
    fig, ax = plt.subplots()
    ax.plot(np.arange(iters), cost2, 'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Error')
    ax.set_title('Error vs Iterations')
    
    # predict overall error using gradient descent  
    quality = (grad_inputs * theta.T)
    errors = np.sum((np.array(quality) - np.array(grad_targets))**2)/len(grad_targets)
    test_error_gd = np.sqrt(errors)
    
    print "Error with gradient descent:", test_error_gd
    
    plt.show()

def evaluate_linear_approx(inputs, targets, test_fraction):
    '''
    This evaluates the linear performance of the data. 
    The test and train errors are then plot against different regularisation parameters.
    '''
    # the linear performance
    reg_params = np.logspace(-12,8, 20)
    train_errors = []
    test_errors = []
    mean_test_error = []
    mean_train_error = []
    std_test_error = []
    std_train_error = []
    
    # plot the average for a number of runs
    for reg_param in reg_params:
        for i in range(1000):
            #print("Evaluating reg_para " + str(reg_param))
            train_error, test_error = simple_evaluation_linear_model(
                inputs, targets, test_fraction=test_fraction, reg_param=reg_param)
            train_errors.append(train_error)
            test_errors.append(test_error)
            # once errors calculated for no. of runs, calculate average, append and clear array for next reg param
        mean_train_error.append(np.mean(train_errors))
        mean_test_error.append(np.mean(test_errors))
        std_train_error.append(np.std(train_errors))
        std_test_error.append(np.std(test_errors))
        train_errors = []
        test_errors = []
    
    fig , ax = plot_train_test_errors(
        "$\lambda$", reg_params, mean_train_error, mean_test_error)
    ax.set_xscale('log')
    
    # train error bars
    lower = mean_train_error - std_train_error/np.sqrt(1000)
    upper = mean_train_error + std_train_error/np.sqrt(1000)
    ax.fill_between(reg_params, lower, upper, alpha=0.2, color='b')
    # test error bars
    lower = mean_test_error - std_test_error/np.sqrt(1000)
    upper = mean_test_error + std_test_error/np.sqrt(1000)
    ax.fill_between(reg_params, lower, upper, alpha=0.2, color='r')
    
    print("Linear Regression:")
    print("\t(mean_train_error, mean_test_error) = %r" % ((np.mean(mean_train_error), np.mean(mean_test_error)),))
    print("\t(std_train_error, std_test_error) = %r" % ((np.mean(std_train_error), np.mean(std_test_error)),))
    return mean_train_error, mean_test_error, std_train_error, std_test_error, fig, ax

def simple_evaluation_linear_model(
        inputs, targets, test_fraction=None, reg_param=None):
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
    reg_param (optional) - the regularisation strength. If provided, then
        regularised least squares fitting is uses with this regularisation
        strength. Otherwise, (non-regularised) least squares is used.

    returns
    -------
    train_error - the training error for the approximation
    test_error - the test error for the approximation
    """
    # get the train test split
    train_part, test_part = train_and_test_split(
        inputs.shape[0], test_fraction=test_fraction)
    # break the data into train and test parts
    train_inputs, train_targets, test_inputs, test_targets = \
        train_and_test_partition(inputs, targets, train_part, test_part)
    # now train and evaluate the error on both sets
    train_error, test_error = train_and_test(
        train_inputs, train_targets, test_inputs, test_targets,
        reg_param=0.1)
    return train_error, test_error    

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
        test_fraction = 0.1
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

def train_and_test(
        train_inputs, train_targets, test_inputs, test_targets, reg_param=None):
    """
    Will fit a linear model with either least squares, or regularised least 
    squares to the training data, then evaluate on both test and training data

    parameters
    ----------
    train_inputs - the input design matrix for training
    train_targets - the training targets as a vector
    test_inputs - the input design matrix for testing
    test_targets - the test targets as a vector
    reg_param (optional) - the regularisation strength. If provided, then
        regularised maximum likelihood fitting is uses with this regularisation
        strength. Otherwise, (non-regularised) least squares is used.

    returns
    -------
    train_error - the training error for the approximation
    test_error - the test error for the approximation
    """
    # Find the optimal weights (depends on regularisation)
    if reg_param is None:
        # use simple least squares approach
        weights = ml_weights(
            train_inputs, train_targets)
    else:
        # use regularised least squares approach
        weights = regularised_ml_weights(
          train_inputs, train_targets,  reg_param)
    # predictions are linear functions of the inputs, we evaluate those here
    train_predicts = linear_model_predict(train_inputs, weights)
    test_predicts = linear_model_predict(test_inputs, weights)
    # evaluate the error between the predictions and true targets on both sets
    train_error = root_mean_squared_error(train_targets, train_predicts)
    test_error = root_mean_squared_error(test_targets, test_predicts)
    return train_error, test_error

def ml_weights(inputmtx, targets):
    """
    This method returns the weights that give the best linear fit between
    the processed inputs and the targets.
    """
    Phi = np.matrix(inputmtx)
    targets = np.matrix(targets).reshape((len(targets),1))
    weights = linalg.inv(Phi.transpose()*Phi)*Phi.transpose()*targets
    return np.array(weights).flatten()

def linear_model_predict(designmtx, weights):
    ys = np.matrix(designmtx)*np.matrix(weights).reshape((len(weights),1))
    return np.array(ys).flatten()

def root_mean_squared_error(y_true, y_pred):
    """
    Evaluate how closely predicted values (y_pred) match the true values
    (y_true, also known as targets)

    Parameters
    ----------
    y_true - the true targets
    y_pred - the predicted targets

    Returns
    -------
    mse - The root mean squared error between true and predicted target
    """
    N = len(y_true)
    # be careful, square must be done element-wise (hence conversion
    # to np.array)
    mse = np.sum((np.array(y_true).flatten() - np.array(y_pred).flatten())**2)/N
    return np.sqrt(mse)

def regularised_ml_weights(
        inputmtx, targets, reg_param):
    """
    This method returns the weights that give the best linear fit between
    the processed inputs and the targets penalised by some regularisation term
    (reg_param)
    """
    Phi = np.matrix(inputmtx)
    targets = np.matrix(targets).reshape((len(targets),1))
    I = np.identity(Phi.shape[1])
    weights = linalg.inv(reg_param*I + Phi.transpose()*Phi)*Phi.transpose()*targets
    return np.array(weights).flatten()
   
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

def gradientDescent(X, y, theta, alpha, iters):
    """
    This method uses the gradient descent algorithm to minmise the error function
    
    This code has been adapted from a tutorial online: 
        johnwittenauer.net. (2015). 
        Machine Learning Exercises In Python, Part 2. [Online]. 
        Available at: http://www.johnwittenauer.net/machine-learning-exercises-in-python-part-2/ (accessed 24th February 2018).

    """
    
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)

    for i in range(iters):
        error = (X * theta.T) - y

        for j in range(parameters):
            term = np.multiply(error, X[:,j])
            temp[0,j] = theta[0,j] - ((alpha / len(X)) * np.sum(term))

        theta = temp
        cost[i] = computeCost(X, y, theta)

    return theta, cost

def computeCost(X, y, theta2): 
    """
    This method computes the cost function for gradient descent
    """ 
    inner = np.power(((X * theta2.T) - y), 2)
    return np.sum(inner) / (2 * len(X))
    
'''def split_data(ifname, delimiter=None, has_header=False, columns=None, seed=42, fraction=0.15):
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
    return training_data_as_array, test_data_as_array, field_names'''


'''if __name__ == '__main__':
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
        main(ifname=sys.argv[1], delimiter=sys.argv[2], columns=columns)'''

