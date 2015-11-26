#!/usr/bin/env python
"""
 The definition of the Convolutional Neural Network for the sentence representation by Duyu Tang
 "Learning Semantic Representations of Users and Products for Document Level Sentiment Classification" (from ACL2015)

 Three filters of size 1,2 and 3
  Each filter is a lookup layer, linear, average, tanh,
 Average filter
 Softmax at the top

 @author Yaroslav Nechaev (remper@me.com)
"""

from neon.backends import gen_backend
from neon.data import DataIterator
from neon.layers import Convolution
from neon.initializers import Uniform
from neon.layers import GeneralizedCost, Affine, Recurrent
from neon.models import Model
from neon.optimizers import RMSProp
from neon.transforms import Tanh, Softmax, CrossEntropyMulti
from neon.callbacks.callbacks import Callbacks
from neon.util.argparser import NeonArgparser
import numpy as np

parser = NeonArgparser(__doc__)
args = parser.parse_args(gen_be=False)

word_dim = 50

batch_size = 50
num_epochs = args.epochs

# these hyperparameters are from the paper
time_steps = 150
hidden_size = 500
clip_gradients = False

# setup backend
be = gen_backend(backend=args.backend,
                 batch_size=batch_size,
                 rng_seed=args.rng_seed,
                 device_id=args.device_id,
                 default_dtype=args.datatype)

# loading the dataset
train_set = []
labels = []
with open(args.data_dir, 'rb') as f:
    for line in f:
        tokens = line.split(" ")
        labels.append(tokens[0])
        sentence = []
        word = []
        for token in tokens[1:]:
            word.append(float(token))
            if len(word) >= word_dim:
                sentence.append(word)
                word = []
        train_set.append(sentence)
train_set = [train_set]
train_set = DataIterator(X=train_set, Y=labels)
# load data and parse


# weight initialization
init = Uniform(low=-0.01, high=0.01)

# model initialization
filters = [Convolution((1,1), (1), init)]
layers = [Recurrent(hidden_size, init, Tanh()),
          Affine(len(train_set.vocab), init, bias=init, activation=Softmax())]

cost = GeneralizedCost(costfunc=CrossEntropyMulti(usebits=True))

model = Model(layers=layers)

optimizer = RMSProp(clip_gradients=clip_gradients, stochastic_round=args.rounding)

# configure callbacks
callbacks = Callbacks(model, train_set, args)

# train model
model.fit(train_set, optimizer=optimizer, num_epochs=num_epochs, cost=cost, callbacks=callbacks)
