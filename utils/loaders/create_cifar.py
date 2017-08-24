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
    file = 'cifar-10-batches-py/data_batch_' + str(i + 1)
    data_batch = unpickle(file)
   
    train_data.append(data_batch['data'])
    train_labels.append(data_batch['labels'])

# Get test data
test_batch  = unpickle('cifar-10-batches-py/test_batch')
test_data   = test_batch['data']
test_labels = test_batch['labels']

# Get npz arrays for train and test data
train_data   = np.concatenate(train_data).reshape(50000, 32, 32, 3)
train_labels = np.concatenate(train_labels)
test_data    = test_data.reshape(10000, 32, 32, 3)

# Save out train/test data and labels
np.savez('cifar-10', train_data=train_data, test_data=test_data, train_labels=train_labels, test_labels=test_labels)
