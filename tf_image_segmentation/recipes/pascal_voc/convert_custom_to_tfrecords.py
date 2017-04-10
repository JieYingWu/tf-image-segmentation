import os, sys, random
import numpy as np


sys.path.append("/home/jieying/CIS2/tf-image-segmentation/")
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

image_root = '/home/jieying/CIS2/data/composed/'
mask_root = '/home/jieying/CIS2/data/composed_mask/'

from tf_image_segmentation.utils.tf_records import write_image_annotation_pairs_to_tfrecord

# Returns a list of (image, annotation) filename pairs (filename.jpg, filename.png)
image_names = os.listdir(image_root)
mask_names = os.listdir(mask_root)

overall_val_image_annotation_filename_pairs = []
overall_train_image_annotation_filename_pairs = []
indices = range(0,np.size(image_names))
np.random.shuffle(indices)

for i in indices[0:100000]:
    overall_train_image_annotation_filename_pairs.append((image_root + image_names[i], mask_root + mask_names[i]))

for i in indices[100000:125000]:
    overall_val_image_annotation_filename_pairs.append((image_root + image_names[i], mask_root + mask_names[i]))

# You can create your own tfrecords file by providing
# your list with (image, annotation) filename pairs here
write_image_annotation_pairs_to_tfrecord(filename_pairs=overall_val_image_annotation_filename_pairs,
                                         tfrecords_filename='custom_augmented_value.tfrecords')

write_image_annotation_pairs_to_tfrecord(filename_pairs=overall_train_image_annotation_filename_pairs,
                                         tfrecords_filename='custom_augmented_train.tfrecords')
