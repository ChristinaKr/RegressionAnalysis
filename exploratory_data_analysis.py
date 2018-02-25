import csv
import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt

def exp_data_ana(
        training_data_as_array,field_names):
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
    exploratory_plots(training_data_as_array, field_names)
    
    scatter_ph_versus_ca(training_data_as_array)
    scatter_quality_pH(training_data_as_array)
    scatter_quality_sugar(training_data_as_array)
    scatter_totalSO2_freeSO2(training_data_as_array)
    scatter_alcohol_density(training_data_as_array)
    scatter_quality_alcohol(training_data_as_array)
    scatter_quality_acidity(training_data_as_array)
    scatter_quality_vacidity(training_data_as_array)
    scatter_quality_ca(training_data_as_array)
    scatter_quality_chlor(training_data_as_array)
    scatter_quality_fSO2(training_data_as_array)
    scatter_quality_totSO2(training_data_as_array)
    scatter_quality_density(training_data_as_array)
    scatter_quality_sulphates(training_data_as_array)
    scatter_quality_sulphates(training_data_as_array)
    
    plt.show()

def exploratory_plots(data, field_names=None):
    '''
    This method takes the input data and generates histograms for all of the variables
    '''
    
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

def scatter_quality_pH(data):
    # create an empty figure object
    fig = plt.figure()
    # create a single axis on that figure
    ax = fig.add_subplot(1,1,1)
    # histogram the data and label the axes
    ax.set_xlabel("pH")
    ax.set_ylabel("Quality")
    fig.suptitle('How pH varies in different quality wines')
    x_val = np.array(data[:,8]) 
    ax.set_ylim([2.5,8.5])
    y_val = np.array(data[:,11])
    ax.plot(x_val, y_val, 'x')
    ax.plot(x_val, np.poly1d(np.polyfit(x_val, y_val, 1))(x_val), color = 'r', linewidth = 2.0)

def scatter_quality_alcohol(data):
    # create an empty figure object
    fig = plt.figure()
    # create a single axis on that figure
    ax = fig.add_subplot(1,1,1)
    # histogram the data and label the axes
    ax.set_xlabel("Alcohol")
    ax.set_ylabel("Quality")
    fig.suptitle('How alcohol content varies in different quality wines')
    x_val = np.array(data[:,10]) 
    ax.set_ylim([2.5,8.5])
    y_val = np.array(data[:,11])
    ax.plot(x_val, y_val, 'x')
    ax.plot(x_val, np.poly1d(np.polyfit(x_val, y_val, 1))(x_val), color = 'r', linewidth = 2.0)

def scatter_quality_acidity(data):
    # create an empty figure object
    fig = plt.figure()
    # create a single axis on that figure
    ax = fig.add_subplot(1,1,1)
    # histogram the data and label the axes
    ax.set_xlabel("Acidity")
    ax.set_ylabel("Quality")
    fig.suptitle('How acidity content varies in different quality wines')
    x_val = np.array(data[:,0]) 
    ax.set_ylim([2.5,8.5])
    y_val = np.array(data[:,11])
    ax.plot(x_val, y_val, 'x')
    ax.plot(x_val, np.poly1d(np.polyfit(x_val, y_val, 1))(x_val), color = 'r', linewidth = 2.0)

def scatter_quality_vacidity(data):
    # create an empty figure object
    fig = plt.figure()
    # create a single axis on that figure
    ax = fig.add_subplot(1,1,1)
    # histogram the data and label the axes
    ax.set_xlabel("Volitile Acidity")
    ax.set_ylabel("Quality")
    fig.suptitle('How volitile acidity content varies in different quality wines')
    x_val = np.array(data[:,1]) 
    ax.set_ylim([2.5,8.5])
    y_val = np.array(data[:,11])
    ax.plot(x_val, y_val, 'x')
    ax.plot(x_val, np.poly1d(np.polyfit(x_val, y_val, 1))(x_val), color = 'r', linewidth = 2.0)

def scatter_quality_ca(data):
    # create an empty figure object
    fig = plt.figure()
    # create a single axis on that figure
    ax = fig.add_subplot(1,1,1)
    # histogram the data and label the axes
    ax.set_xlabel("Citric Acid")
    ax.set_ylabel("Quality")
    fig.suptitle('How citric acid content varies in different quality wines')
    x_val = np.array(data[:,2]) 
    ax.set_ylim([2.5,8.5])
    y_val = np.array(data[:,11])
    ax.plot(x_val, y_val, 'x')
    ax.plot(x_val, np.poly1d(np.polyfit(x_val, y_val, 1))(x_val), color = 'r', linewidth = 2.0)

def scatter_quality_chlor(data):
    # create an empty figure object
    fig = plt.figure()
    # create a single axis on that figure
    ax = fig.add_subplot(1,1,1)
    # histogram the data and label the axes
    ax.set_xlabel("Chlorides")
    ax.set_ylabel("Quality")
    fig.suptitle('How Chloride content varies in different quality wines')
    x_val = np.array(data[:,4]) 
    ax.set_ylim([2.5,8.5])
    y_val = np.array(data[:,11])
    ax.plot(x_val, y_val, 'x')
    ax.plot(x_val, np.poly1d(np.polyfit(x_val, y_val, 1))(x_val), color = 'r', linewidth = 2.0)
    
def scatter_quality_fSO2(data):
    # create an empty figure object
    fig = plt.figure()
    # create a single axis on that figure
    ax = fig.add_subplot(1,1,1)
    # histogram the data and label the axes
    ax.set_xlabel("Free SO2")
    ax.set_ylabel("Quality")
    fig.suptitle('How Free SO2 content varies in different quality wines')
    x_val = np.array(data[:,5]) 
    ax.set_ylim([2.5,8.5])
    y_val = np.array(data[:,11])
    ax.plot(x_val, y_val, 'x')
    ax.plot(x_val, np.poly1d(np.polyfit(x_val, y_val, 1))(x_val), color = 'r', linewidth = 2.0)

def scatter_quality_totSO2(data):
    # create an empty figure object
    fig = plt.figure()
    # create a single axis on that figure
    ax = fig.add_subplot(1,1,1)
    # histogram the data and label the axes
    ax.set_xlabel("Total SO2")
    ax.set_ylabel("Quality")
    fig.suptitle('How total SO2 content varies in different quality wines')
    x_val = np.array(data[:,6]) 
    ax.set_ylim([2.5,8.5])
    y_val = np.array(data[:,11])
    ax.plot(x_val, y_val, 'x')
    ax.plot(x_val, np.poly1d(np.polyfit(x_val, y_val, 1))(x_val), color = 'r', linewidth = 2.0)                                        

def scatter_quality_density(data):
    # create an empty figure object
    fig = plt.figure()
    # create a single axis on that figure
    ax = fig.add_subplot(1,1,1)
    # histogram the data and label the axes
    ax.set_xlabel("Density")
    ax.set_ylabel("Quality")
    fig.suptitle('How density varies in different quality wines')
    x_val = np.array(data[:,7]) 
    ax.set_ylim([2.5,8.5])
    y_val = np.array(data[:,11])
    ax.plot(x_val, y_val, 'x')
    ax.plot(x_val, np.poly1d(np.polyfit(x_val, y_val, 1))(x_val), color = 'r', linewidth = 2.0)                                                                                                                        

def scatter_quality_sulphates(data):
    # create an empty figure object
    fig = plt.figure()
    # create a single axis on that figure
    ax = fig.add_subplot(1,1,1)
    # histogram the data and label the axes
    ax.set_xlabel("Sulphates")
    ax.set_ylabel("Quality")
    fig.suptitle('How sulphate content varies in different quality wines')
    x_val = np.array(data[:,9]) 
    ax.set_ylim([2.5,8.5])
    y_val = np.array(data[:,11])
    ax.plot(x_val, y_val, 'x')
    ax.plot(x_val, np.poly1d(np.polyfit(x_val, y_val, 1))(x_val), color = 'r', linewidth = 2.0)                                                                                                                                                                                                                                                                                                                                                                        
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
def scatter_ph_versus_ca(data):
    # write the function
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(1,1,1)
    ax2.set_xlabel("Citric Acid")
    ax2.set_ylabel("pH")
    x_val = np.array(data[:,2])
    y_val = np.array(data[:,8])
    fig2.suptitle('How Citric Acid effects pH')
    ax2.plot(x_val, y_val, 'x')
    # linear line of best fit
    ax2.plot(x_val, np.poly1d(np.polyfit(x_val, y_val, 1))(x_val), color = 'r', linewidth = 2.0)

def scatter_quality_sugar(data):
    # create an empty figure object
    fig = plt.figure()
    # create a single axis on that figure
    ax = fig.add_subplot(1,1,1)
    # histogram the data and label the axes
    ax.set_xlabel("Sugar")
    ax.set_ylabel("Quality")
    fig.suptitle('How Sugar varies in different quality wines')
    x_val = np.array(data[:,3])
    ax.set_ylim([2.5,8.5])
    y_val = np.array(data[:,11])
    ax.plot(x_val, y_val, 'x')
    ax.plot(x_val, np.poly1d(np.polyfit(x_val, y_val, 1))(x_val), color = 'r', linewidth = 2.0)

def scatter_totalSO2_freeSO2(data):
    # create an empty figure object
    fig = plt.figure()
    # create a single axis on that figure
    ax = fig.add_subplot(1,1,1)
    # histogram the data and label the axes
    ax.set_xlabel("Total SO2")
    ax.set_ylabel("Free SO2")
    fig.suptitle('How total SO2 effects free SO2 in wine')
    x_val = np.array(data[:,6])
    y_val = np.array(data[:,5])
    ax.plot(x_val, y_val, 'x')
    ax.plot(x_val, np.poly1d(np.polyfit(x_val, y_val, 1))(x_val), color = 'r', linewidth = 2.0)

def scatter_alcohol_density(data):
    # create an empty figure object
    fig = plt.figure()
    # create a single axis on that figure
    ax = fig.add_subplot(1,1,1)
    # histogram the data and label the axes
    ax.set_xlabel("Density")
    ax.set_ylabel("Alcohol")
    fig.suptitle('The relationship between Density and Alcohol')
    x_val = np.array(data[:,7])
    y_val = np.array(data[:,10])
    ax.plot(x_val, y_val, 'x')
    ax.plot(x_val, np.poly1d(np.polyfit(x_val, y_val, 1))(x_val), color = 'r', linewidth = 2.0)

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
        main(ifname=sys.argv[1], delimiter=sys.argv[2], columns=columns)'''
