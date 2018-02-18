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


def construct_knn_approx(train_inputs, train_targets, k):  
    """
    For 1 dimensional training data, it produces a function:reals-> reals
    that outputs the mean training value in the k-Neighbourhood of any input.
    """
    # Create Euclidean distance.
    distance = lambda x,y: (x-y)**2
    print(np.shape(train_inputs))
#    train_inputs = np.resize(train_inputs, (1, train_inputs.size)) #convert into row-vector (1,19188 (11x1599))
    train_inputs = train_inputs.transpose() # 12 x 1599
    print("shape train_inputs old", np.shape(train_inputs))
    data = train_inputs[:11,:]
    print("shape train_inputs new aka data ", np.shape(data))
    Xx = np.array([4, 0.7, 0.1, 3, 0.07, 13, 40, 0.9964, 3, 0.56, 10])
    print("Xx shape:", np.shape(Xx))
    def prediction_function(Xx):
        Xx = Xx.reshape((Xx.size,1))
        print(Xx.shape)
        print(Xx)
        distances = distance(data, Xx)
        print("distance shape old: ", np.shape(distances))
        distanceSum = [np.sum(distances, axis = 0)]
        distanceSum = np.array(distanceSum)
        distanceSum = distanceSum.reshape((1, distanceSum.size))
        print("distanceSum shape new: ", np.shape(distanceSum)) # 1 x 1599 (row vector will distances of all datapoints)
        ys = train_inputs[11,:].reshape(1, 1599)
        print("Shape of train_inputs[11,:]", np.shape(ys))
        
        # append the quality values to the distance values (making a 2 x 1599 array)
        distanceQuality = np.append(distanceSum, ys, axis = 0)
        print("distanceQuality appended shape: ", np.shape(distanceQuality))
        
        #sort array with regard to first row (distanceSum)
        distanceQuality = distanceQuality.transpose()
        distanceQSorted = distanceQuality[distanceQuality[:,0].argsort()]
        print("distanceQSorted shape: ", np.shape(distanceQSorted))
        
        #average over k-nearest neighbours
        predicts = np.mean(distanceQSorted[:k,1])
        print(predicts)
        
        
        #print("Xx size:", Xx.size)
        #predicts = np.empty(Xx.size)
        #for i, neighbourhood in enumerate(np.argpartition(distances, k)[:,:k]):
            # the neighbourhood is the indices of the closest inputs to xs[i]
            # the prediction is the mean of the targets for this neighbourhood
            # predicts[i] = np.mean(train_targets[neighbourhood])
        return predicts
    # We return a handle to the locally defined function
    return prediction_function(Xx)
    

def main(ifname):
    data = import_data(ifname)
    if type(data) == np.ndarray:
        print("Data array loaded: there are %d rows" % data.shape[0])
        print ("first row:", data[0,:])
    targets = data[:,11]
    construct_knn_approx(data, targets, 3)
#    fig = plt.figure()
#    ax = fig.add_subplot(1,1,1)
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

