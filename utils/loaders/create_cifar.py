import numpy as np
import cPickle

def unpickle(file):
    '''
    Taken from https://www.cs.toronto.edu/~kriz/cifar.html
    '''
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict




train_data   = []
train_labels = []

# Get training data
for i in range(5):
    file = 'datasets/cifar-10-batches-py/data_batch_' + str(i + 1)
    data_batch = unpickle(file)
   
    train_data.append(data_batch['data'])
    train_labels.append(data_batch['labels'])

# Get test data
test_batch  = unpickle('datasets/cifar-10-batches-py/test_batch')
test_data   = test_batch['data']
test_labels = test_batch['labels']

# Get npz arrays for train and test data
train_data   = np.transpose(np.concatenate(train_data).reshape(50000, 3, 32, 32), axes=[0,2,3,1])
train_labels = np.concatenate(train_labels)
test_data    = np.transpose(test_data.reshape(10000, 3, 32, 32), axes=[0,2,3,1])

# Save out train/test data and labels
np.savez('cifar-10', train_data=train_data, test_data=test_data, train_labels=train_labels, test_labels=test_labels)
