import numpy as np
import sys
import os
import argparse
import pprint

import _init_paths
from handleData import get_from_mat
from config import cfg
from timer import Timer

import caffe
from caffe.proto import caffe_pb2
import google.protobuf as pb2


def test_mnist(net, data):
    """Test a Fast R-CNN network on an image database."""
    imagesum = data['data'].shape[0]
    accuracy = []
    for i in xrange(imagesum):
    	im = np.zeros((1,1,20,20), dtype = np.float32)
    	im[0,0,:,:] = data['data'][i,0,:,:]
    	# label = data['label'][i]
        label = np.zeros((1,1,1,1), dtype = np.float32)
        label[0,0,0,0] = data['label'][i]
    	net.blobs['data'].reshape(*(im.shape))
    	net.blobs['label'].reshape(*(1,1,1,1))
    	blob_out = net.forward(data=im.astype(np.float32, copy=False),
                               label=label.astype(np.float32, copy=False))
        accuracy.append(blob_out['accuracy'].tolist())
    return accuracy


if __name__ == '__main__':

    testfile = '/net/liuwenran/caffe_learn/data/lwr_test_img_mat.mat'
    test_prototxt = '/net/liuwenran/caffe_learn/proto/lwr_lenet_test.prototxt'
    output_dir = '/net/liuwenran/caffe_learn/data/output/'
    pretrained_model = '/net/liuwenran/caffe_learn/data/output/lenet_iter_10000.caffemodel'
    gpu_id = 3

    mnist_test_data = get_from_mat(testfile)

    print('Using config:')
    pprint.pprint(cfg)

    # set up caffe
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)
    net = caffe.Net(test_prototxt, pretrained_model, caffe.TEST)
    net.name = os.path.splitext(os.path.basename(pretrained_model))[0]

    accuracy = test_mnist(net, mnist_test_data)
    final_score = sum(accuracy) / mnist_test_data['data'].shape[0]
    print 'final_score is ' + str(final_score)