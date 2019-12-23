from __future__ import absolute_import

from .vgg import *


def initialize_model(name):
    if name == 'vgg_v1':
        return vgg_v1
    if name == 'vgg_v0':
        return vgg_v0
