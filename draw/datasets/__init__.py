
from __future__ import division

supported_datasets = ['bmnist', 'silhouettes']
# ToDo: # 'mnist' and 'tfd' are not normalized (0<= x <=1.)


def get_data(data_name):
    if data_name == 'mnist':
        from fuel.datasets import MNIST

        img_size = (28, 28)

        data_train = MNIST(which_set="train", sources=['features'])
        data_valid = MNIST(which_set="test", sources=['features'])
        data_test  = MNIST(which_set="test", sources=['features'])
    elif data_name == 'bmnist':
        from fuel.datasets.binarized_mnist import BinarizedMNIST

        img_size = (28, 28)

        data_train = BinarizedMNIST(which_set='train', sources=['features'])
        data_valid = BinarizedMNIST(which_set='valid', sources=['features'])
        data_test  = BinarizedMNIST(which_set='test', sources=['features'])
    elif data_name == 'silhouettes':
        from fuel.datasets.caltech101_silhouettes import CalTech101Silhouettes

        size = 28
        img_size = (size, size)

        data_train = CalTech101Silhouettes(which_set=['train'], size=size, sources=['features'])
        data_valid = CalTech101Silhouettes(which_set=['valid'], size=size, sources=['features'])
        data_test  = CalTech101Silhouettes(which_set=['test'], size=size, sources=['features'])
    elif data_name == 'tfd':
        from fuel.datasets.toronto_face_database import TorontoFaceDatabase

        size = 28
        img_size = (size, size)

        data_train = TorontoFaceDatabase(which_set=['unlabeled'], size=size, sources=['features'])
        data_valid = TorontoFaceDatabase(which_set=['valid'], size=size, sources=['features'])
        data_test  = TorontoFaceDatabase(which_set=['test'], size=size, sources=['features'])
        
    elif data_name == 'speech':
        from SynthesisTaskData import SynthesisTaskData
        
        img_size = (28,28)
        
        data_train = SynthesisTaskData(sources = ['features'])
        data_valid = SynthesisTaskData(sources = ['features'])
        data_test = SynthesisTaskData(sources = ['features'])
    else:
        raise ValueError("Unknown dataset %s" % data_name)

    return img_size, data_train, data_valid, data_test
