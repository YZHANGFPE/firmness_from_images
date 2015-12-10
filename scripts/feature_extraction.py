import numpy as np
import matplotlib.pyplot as plt
import sys
import os

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

caffe_root = '/home/yi/caffe/'
image_root = '../saved_images/'
output_path = 'features.npy'
sys.path.insert(0, caffe_root + 'python')

import caffe

caffe.set_mode_cpu()
net = caffe.Net(caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt',
                caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
                caffe.TEST)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

# create image folder list
folder_list = [folder for folder in os.listdir(image_root)]

res = []
for i in range(len(folder_list)):
    for file_name in os.listdir(image_root + folder_list[i]):
        image_path = image_root + folder_list[i] + '/' + file_name
        f = {'image_path' : image_path, 'category' : folder_list[i]}
        res.append(f)

size = len(res)

# set net to batch size
net.blobs['data'].reshape(size,3,227,227)

for i in range(size):
    net.blobs['data'].data[i] = transformer.preprocess('data', caffe.io.load_image(res[i]['image_path']))

out = net.forward()

for i in range(size):

    # print "The input image is : ", folder_list[i]
    # print("Predicted class is #{}.".format(out['prob'][i].argmax()))
    # 
    # # load labels
    # imagenet_labels_filename = caffe_root + 'data/ilsvrc12/synset_words.txt'
    # labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')
    # 
    # # sort top k predictions from softmax output
    # top_k = net.blobs['prob'].data[i].flatten().argsort()[-1]
    # print labels[top_k]
    # # plt.imshow(transformer.deprocess('data', net.blobs['data'].data[0]))
    # # plt.show()

    res[i]['feature'] = net.blobs['fc7'].data[i]

print "Number of input: ", size
print "Save result to: ", output_path
np.save(output_path, res)
