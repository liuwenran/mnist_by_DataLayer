import scipy.io as sio
import os
import os.path as osp
import sys
import numpy as np

def get_from_mat(imagefile):
    assert os.path.exists(imagefile), \
            'data file not found at: {}'.format(imagefile)
    image = sio.loadmat(imagefile)
    keys = image.keys()
    image = image[keys[0]]
    imagenum = []
    for i in xrange(10):
    	subnum = image[0,i].shape[2]
    	imagenum.append(subnum)

    imagesum = sum(imagenum)
    blob_image = np.zeros((imagesum,1, 20, 20), dtype = np.float32)
    blob_label = np.zeros(imagesum,dtype = np.int8)
    imagecount = 0
    for i in xrange(10):
    	for j in xrange(image[0,i].shape[2]):
    		blob_image[imagecount,0,:,:] = image[0,i][:,:,j]
    		blob_label[imagecount] = i
    		imagecount += 1

    result = {}
    result['data'] = blob_image
    result['label'] = blob_label
    return result

if __name__ == '__main__':
    trainfile = '/net/liuwenran/caffe_learn/data/lwr_train_img_mat.mat'
    data = get_from_mat(trainfile)
    perm = np.random.permutation(np.arange(len(data['data'])))
    inds = perm[0:99]
    data['data'] = data['data'][inds,:,:,:]
    data['label'] = data['label'][inds]

