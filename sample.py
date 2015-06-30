#!/usr/bin/env python

from __future__ import print_function, division

import logging
import theano
import theano.tensor as T
import cPickle as pickle

import numpy as np
import os

from PIL import Image
from blocks.main_loop import MainLoop
from blocks.model import AbstractModel
from blocks.config_parser import config


from fuel.streams import DataStream
from fuel.schemes import SequentialScheme
from fuel.transformers import Transformer
from abc import ABCMeta, abstractmethod
from six import add_metaclass, iteritems

from draw.datasets.SynthesisTaskData import SynthesisTaskData

FORMAT = '[%(asctime)s] %(name)-15s %(message)s'
DATEFMT = "%H:%M:%S"
logging.basicConfig(format=FORMAT, datefmt=DATEFMT, level=logging.INFO)

def scale_norm(arr):
    arr = arr - arr.min()
    scale = (arr.max() - arr.min())
    return arr / scale

def img_grid(arr, global_scale=True):
    N, height, width = arr.shape

    rows = int(np.sqrt(N))
    cols = int(np.sqrt(N))

    if rows*cols < N:
        cols = cols + 1

    if rows*cols < N:
        rows = rows + 1

    total_height = rows * height
    total_width  = cols * width

    if global_scale:
        arr = scale_norm(arr)

    I = np.zeros((total_height, total_width))

    for i in xrange(N):
        r = i // cols
        c = i % cols

        if global_scale:
            this = arr[i]
        else:
            this = scale_norm(arr[i])

        offset_y, offset_x = r*height, c*width
        I[offset_y:(offset_y+height), offset_x:(offset_x+width)] = this

    I = (255*I).astype(np.uint8)
    return Image.fromarray(I)

def encode_features(p):
    if isinstance(p, AbstractModel):
        model = p
    else:
        print("Don't know how to handle unpickled %s" % type(p))
        return
    
    draw = model.get_top_bricks()[0]
    
    std = SynthesisTaskData(sources = ['features'])
    test_stream = Flatten(DataStream(std, iteration_scheme=SequentialScheme(std.num_examples, 1)))
    
    z_dim = draw.sampler.mean_transform.get_dim('output')
    
    features = T.ftensor3("features")
    encodings = draw.get_feature_encoding(features)
    do_encoding = theano.function([features], outputs = encodings, allow_input_downcast=True)
    encodings = do_encoding(test_stream.get_epoch_iterator().__next__())
    n_iter, N, D = encodings.shape
    print(encodings.shape)
    
    np.save("encodings",encodings)
    
def reconstruct(p):
    if isinstance(p, AbstractModel):
        model = p
    else:
        print("Don't know how to handle unpickled %s" % type(p))
        return
    
    draw = model.get_top_bricks()[0]
    
    std = SynthesisTaskData(sources = ['features'])
    test_stream = ExtraFlat(DataStream(std, iteration_scheme=SequentialScheme(std.num_examples, 100)))
    
    z_dim = draw.sampler.mean_transform.get_dim('output')
    
    features = T.matrix("features")
    recons, kl = draw.reconstruct(features)
    do_encoding = theano.function([features], outputs = [recons, kl], allow_input_downcast=True)
    recons, kl = do_encoding(test_stream.get_epoch_iterator().__next__()[0])
#     n_iter, N, D = recons.shape
    print(recons.shape)
    
    np.save("orig",test_stream.get_epoch_iterator().__next__()[0])
    np.save("recons",recons)

def generate_samples(p, subdir, output_size):
    if isinstance(p, AbstractModel):
        model = p
    else:
        print("Don't know how to handle unpickled %s" % type(p))
        return

    draw = model.get_top_bricks()[0]
    # reset the random generator
    del draw._theano_rng
    del draw._theano_seed
    draw.seed_rng = np.random.RandomState(config.default_seed)

    #------------------------------------------------------------
    logging.info("Compiling sample function...")

    n_samples = T.iscalar("n_samples")
    samples = draw.sample(n_samples)

    do_sample = theano.function([n_samples], outputs=samples, allow_input_downcast=True)

    #------------------------------------------------------------
    logging.info("Sampling and saving images...")

    samples = do_sample(16*16)
    #samples = np.random.normal(size=(16, 100, 28*28))

    n_iter, N, D = samples.shape

    np.save(os.path.join(subdir,"samples"), samples)
    
    samples = samples.reshape( (n_iter, N, output_size, output_size) )

    if(n_iter > 0):
        img = img_grid(samples[n_iter-1,:,:,:])
        img.save("{0}/sample.png".format(subdir))

    for i in xrange(n_iter):
        img = img_grid(samples[i,:,:,:])
        img.save("{0}/sample-{1:03d}.png".format(subdir, i))

    #with open("centers.pkl", "wb") as f:
    #    pikle.dump(f, (center_y, center_x, delta))
    os.system("convert -delay 5 -loop 0 {0}/sample-*.png {0}/samples.gif".format(subdir))
        
# def chunk(n_of_chunks, second):
#     second_chunk_length = len(second)/n_of_chunks

#     second_chunks = []
#     for i in range(n_of_chunks):
#         second_chunks.append(second[i*second_chunk_length:(i+1)*second_chunk_length])
#     npsecond = np.array(second_chunks, dtype = theano.config.floatX)
#     return np.reshape(npsecond, (chunk_size,784))

#----------------------------------------------------------------------------
"""
Inclusion of SingleMapping transformer from newer version of Fuel
"""
@add_metaclass(ABCMeta)
class SingleMapping(Transformer):
    """Applies a single mapping to multiple sources.

    Parameters
    ----------
    data_stream : instance of :class:`DataStream`
        The wrapped data stream.
    which_sources : tuple of str, optional
        Which sources to apply the mapping to. Defaults to `None`, in
        which case the mapping is applied to all sources.

    """
    def __init__(self, data_stream, which_sources=None):
        if which_sources is None:
            which_sources = data_stream.sources
        self.which_sources = which_sources
        super(SingleMapping, self).__init__(data_stream)

    @abstractmethod
    def mapping(self, source):
        """Applies a single mapping to selected sources.

        Parameters
        ----------
        source : :class:`numpy.ndarray`
            Input source.

        """

    def get_data(self, request=None):
        if request is not None:
            raise ValueError
        data = list(next(self.child_epoch_iterator))
        for i, source_name in enumerate(self.data_stream.sources):
            if source_name in self.which_sources:
                data[i] = self.mapping(data[i])
        return tuple(data)


class Flatten(SingleMapping):
    """Flattens selected sources along all but the first axis."""
    def __init__(self, data_stream, **kwargs):
        super(Flatten, self).__init__(data_stream, **kwargs)

    def mapping(self, source):
        return source.reshape((source.shape[0], -1))
    
class ExtraFlat(SingleMapping):
    def __init__(self, data_stream, **kwargs):
        super(ExtraFlat, self).__init__(data_stream, **kwargs)
    def mapping(self, source):
        return source.reshape((100,784))
#----------------------------------------------------------------------------

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("model_file", help="filename of a pickled DRAW model")
    parser.add_argument("--mode", help="either sample or feature encoding", default = "sample", dest="mode")
    parser.add_argument("--data", help="filename of your data", default = "data.npy", dest="data")
    parser.add_argument("--size", type=int,
                default=28, help="Output image size (width and height)")
    args = parser.parse_args()

    logging.info("Loading file %s..." % args.model_file)
    with open(args.model_file, "rb") as f:
        p = pickle.load(f)

    if(args.mode == "sample"):
        subdir = "sample"
        if not os.path.exists(subdir):
            os.makedirs(subdir)

        generate_samples(p, subdir, args.size)
    else:
        reconstruct(p)
