
# coding: utf-8

# In[31]:

import urllib
from scipy import fft, arange, ifft
import math
import os
from collections import OrderedDict
import numpy as np
from fuel import config
from fuel.datasets import IndexableDataset
from fuel.utils import do_not_pickle_attributes
from progressbar import ProgressBar
import theano


# In[53]:

class SynthesisTaskData(IndexableDataset):
    """
    Dataset where a sequence of character indexes are the features
    and a sequence of audio magnitudes are the targets
    warning that the audio sequence is ~500x longer than the text sequence
    """
    # switched targets and features so that features is now the audio
    provides_sources = ('targets', 'features')
    
    def __init__(self, **kwargs):
        super(SynthesisTaskData, self).__init__(
            OrderedDict(zip(self.provides_sources,
                           self._load_data())),
            **kwargs
        )
        
    def load(self):
        self.indexables = [data[self.start:self.stop] for source, data
                          in zip(self.provides_sources, self._load_data())
                          if source in self.sources]
  

    def _load_data(self):
        self.vocab = OrderedDict()
        
        text = open(config.data_path+"/speech_synthesis/preface.txt").read()
        audio = np.load(config.data_path+"/speech_synthesis/preface.npy")
#         audio = np.reshape(audio, audio.shape+(1,))
        
        print audio.shape
        print len(text)
        
        chunk_size = 2570
          
        def vocab_add_and_lookup(character):
            if character not in self.vocab:
                self.vocab[character] = 0
            return self.vocab.keys().index(character)

        def encode(sequence):
            return np.array(map(lambda c: vocab_add_and_lookup(c), text))
        
        def chunk(n_of_chunks, first, second):
            first_chunk_length = len(first)/n_of_chunks
            second_chunk_length = len(second)/n_of_chunks
            
            first_chunks = []
            second_chunks = []
            for i in range(n_of_chunks):
                first_chunks.append(first[i*first_chunk_length:(i+1)*first_chunk_length])
                second_chunks.append(second[i*second_chunk_length:(i+1)*second_chunk_length])
            npfirst = np.array(first_chunks, dtype = "int64")
            npsecond = np.array(second_chunks, dtype = theano.config.floatX)
            return (
                np.reshape(npfirst, (chunk_size, 1)),
                np.reshape(npsecond, (chunk_size,784))
            )
            
        chunks = chunk (chunk_size, encode(text), audio)
        print len(chunks[0]), len(chunks[1])
        print chunks[1].shape
        print chunks[0].shape
        return chunks