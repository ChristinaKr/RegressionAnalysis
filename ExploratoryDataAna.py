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

def histogram_quality(data):
    # create an empty figure object
    fig = plt.figure()
    # create a single axis on that figure
    ax = fig.add_subplot(1,1,1)
    # histogram the data and label the axes
    ax.set_xlabel("Quality")
    ax.set_ylabel("Number of Observations")
    fig.suptitle('Histogram of Wine Quality')
    quality = data[:,11]
    ax.hist(quality,bins=10)
    
def hist_citricA(data):
    # create an empty figure object
    fig = plt.figure()
    # create a single axis on that figure
    ax = fig.add_subplot(1,1,1)
    # histogram the data and label the axes
    ax.set_xlabel("Citric Acid")
    ax.set_ylabel("Number of Observations")
    fig.suptitle('Distribution of Citric Acid')
    quality = data[:,2]
    ax.hist(quality,bins=10)

def scatter_quality_pH(data):
    # create an empty figure object
    fig = plt.figure()
    # create a single axis on that figure
    ax = fig.add_subplot(1,1,1)
    # histogram the data and label the axes
    ax.set_xlabel("Quality")
    ax.set_ylabel("pH")
    fig.suptitle('How pH varies in different quality wines')
    x_val = np.array(data[:,11]) 
    ax.set_xlim([2.5,8.5])
    y_val = np.array(data[:,8])
    ax.plot(x_val, y_val, 'x')
    ax.plot(x_val, np.poly1d(np.polyfit(x_val, y_val, 1))(x_val), color = 'r', linewidth = 2.0)

def scatter_quality_alcohol(data):
    # create an empty figure object
    fig = plt.figure()
    # create a single axis on that figure
    ax = fig.add_subplot(1,1,1)
    # histogram the data and label the axes
    ax.set_xlabel("Quality")
    ax.set_ylabel("Alcohol")
    fig.suptitle('How alcohol content varies in different quality wines')
    x_val = np.array(data[:,11]) 
    ax.set_xlim([2.5,8.5])
    y_val = np.array(data[:,10])
    ax.plot(x_val, y_val, 'x')
    ax.plot(x_val, np.poly1d(np.polyfit(x_val, y_val, 1))(x_val), color = 'r', linewidth = 2.0)

def scatter_quality_acidity(data):
    # create an empty figure object
    fig = plt.figure()
    # create a single axis on that figure
    ax = fig.add_subplot(1,1,1)
    # histogram the data and label the axes
    ax.set_xlabel("Quality")
    ax.set_ylabel("Acidity")
    fig.suptitle('How acidity content varies in different quality wines')
    x_val = np.array(data[:,11]) 
    ax.set_xlim([2.5,8.5])
    y_val = np.array(data[:,0])
    ax.plot(x_val, y_val, 'x')
    ax.plot(x_val, np.poly1d(np.polyfit(x_val, y_val, 1))(x_val), color = 'r', linewidth = 2.0)

def scatter_quality_vacidity(data):
    # create an empty figure object
    fig = plt.figure()
    # create a single axis on that figure
    ax = fig.add_subplot(1,1,1)
    # histogram the data and label the axes
    ax.set_xlabel("Quality")
    ax.set_ylabel("Volitile Acidity")
    fig.suptitle('How volitile acidity content varies in different quality wines')
    x_val = np.array(data[:,11]) 
    ax.set_xlim([2.5,8.5])
    y_val = np.array(data[:,1])
    ax.plot(x_val, y_val, 'x')
    ax.plot(x_val, np.poly1d(np.polyfit(x_val, y_val, 1))(x_val), color = 'r', linewidth = 2.0)

def scatter_quality_ca(data):
    # create an empty figure object
    fig = plt.figure()
    # create a single axis on that figure
    ax = fig.add_subplot(1,1,1)
    # histogram the data and label the axes
    ax.set_xlabel("Quality")
    ax.set_ylabel("Citric Acid")
    fig.suptitle('How citric acid content varies in different quality wines')
    x_val = np.array(data[:,11]) 
    ax.set_xlim([2.5,8.5])
    y_val = np.array(data[:,2])
    ax.plot(x_val, y_val, 'x')
    ax.plot(x_val, np.poly1d(np.polyfit(x_val, y_val, 1))(x_val), color = 'r', linewidth = 2.0)

def scatter_quality_chlor(data):
    # create an empty figure object
    fig = plt.figure()
    # create a single axis on that figure
    ax = fig.add_subplot(1,1,1)
    # histogram the data and label the axes
    ax.set_xlabel("Quality")
    ax.set_ylabel("Chlorides")
    fig.suptitle('How Chloride content varies in different quality wines')
    x_val = np.array(data[:,11]) 
    ax.set_xlim([2.5,8.5])
    y_val = np.array(data[:,4])
    ax.plot(x_val, y_val, 'x')
    ax.plot(x_val, np.poly1d(np.polyfit(x_val, y_val, 1))(x_val), color = 'r', linewidth = 2.0)
    
def scatter_quality_fSO2(data):
    # create an empty figure object
    fig = plt.figure()
    # create a single axis on that figure
    ax = fig.add_subplot(1,1,1)
    # histogram the data and label the axes
    ax.set_xlabel("Quality")
    ax.set_ylabel("Free SO2")
    fig.suptitle('How Free SO2 content varies in different quality wines')
    x_val = np.array(data[:,11]) 
    ax.set_xlim([2.5,8.5])
    y_val = np.array(data[:,5])
    ax.plot(x_val, y_val, 'x')
    ax.plot(x_val, np.poly1d(np.polyfit(x_val, y_val, 1))(x_val), color = 'r', linewidth = 2.0)

def scatter_quality_totSO2(data):
    # create an empty figure object
    fig = plt.figure()
    # create a single axis on that figure
    ax = fig.add_subplot(1,1,1)
    # histogram the data and label the axes
    ax.set_xlabel("Quality")
    ax.set_ylabel("Total SO2")
    fig.suptitle('How total SO2 content varies in different quality wines')
    x_val = np.array(data[:,11]) 
    ax.set_xlim([2.5,8.5])
    y_val = np.array(data[:,6])
    ax.plot(x_val, y_val, 'x')
    ax.plot(x_val, np.poly1d(np.polyfit(x_val, y_val, 1))(x_val), color = 'r', linewidth = 2.0)                                        

def scatter_quality_density(data):
    # create an empty figure object
    fig = plt.figure()
    # create a single axis on that figure
    ax = fig.add_subplot(1,1,1)
    # histogram the data and label the axes
    ax.set_xlabel("Quality")
    ax.set_ylabel("Density")
    fig.suptitle('How density varies in different quality wines')
    x_val = np.array(data[:,11]) 
    ax.set_xlim([2.5,8.5])
    y_val = np.array(data[:,7])
    ax.plot(x_val, y_val, 'x')
    ax.plot(x_val, np.poly1d(np.polyfit(x_val, y_val, 1))(x_val), color = 'r', linewidth = 2.0)                                                                                                                        

def scatter_quality_sulphates(data):
    # create an empty figure object
    fig = plt.figure()
    # create a single axis on that figure
    ax = fig.add_subplot(1,1,1)
    # histogram the data and label the axes
    ax.set_xlabel("Quality")
    ax.set_ylabel("Sulphates")
    fig.suptitle('How sulphate content varies in different quality wines')
    x_val = np.array(data[:,11]) 
    ax.set_xlim([2.5,8.5])
    y_val = np.array(data[:,9])
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
    ax2.plot(np.unique(x_val), np.poly1d(np.polyfit(x_val, y_val, 1))(np.unique(x_val)), color = 'r', linewidth = 2.0)

def scatter_quality_sugar(data):
    # create an empty figure object
    fig = plt.figure()
    # create a single axis on that figure
    ax = fig.add_subplot(1,1,1)
    # histogram the data and label the axes
    ax.set_xlabel("Quality")
    ax.set_ylabel("Sugar")
    fig.suptitle('How Sugar varies in different quality wines')
    x_val = np.array(data[:,11])
    ax.set_xlim([2.5,8.5])
    y_val = np.array(data[:,3])
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


def main(ifname):
    data = import_data(ifname)
    if type(data) == np.ndarray:
        print("Data array loaded: there are %d rows" % data.shape[0])
        print "first row:", data[0,:]
    histogram_quality(data)
    hist_citricA(data)
    scatter_ph_versus_ca(data)
    scatter_quality_pH(data)
    scatter_quality_sugar(data)
    scatter_totalSO2_freeSO2(data)
    scatter_alcohol_density(data)
    scatter_quality_alcohol(data)
    scatter_quality_acidity(data)
    scatter_quality_vacidity(data)
    scatter_quality_ca(data)
    scatter_quality_chlor(data)
    scatter_quality_fSO2(data)
    scatter_quality_totSO2(data)
    scatter_quality_density(data)
    scatter_quality_sulphates(data)
    scatter_quality_sulphates(data)
    plt.show()
   

if __name__ == '__main__':
    import sys
    # this allows you to pass the file name as the first argument when you call
    # your script from the command line
    # so to run this script use:
    # python old_faithful.py old_faithful.tsv
    try:
        main(sys.argv[1])
    except IndexError:
        print(
            "[ERROR] Please give the data-file location as the first argument.")

