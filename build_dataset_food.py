"""Split the SIGNS dataset into train/dev/test and resize images to 64x64.

The SIGNS dataset comes in the following format:
    train_signs/
        0_IMG_5864.jpg
        ...
    test_signs/
        0_IMG_5942.jpg
        ...

Original images have size (3024, 3024).
Resizing to (64, 64) reduces the dataset size from 1.16 GB to 4.7 MB, and loading smaller images
makes training faster.

We already have a test set created, so we only need to split "train_signs" into train and dev sets.
Because we don't have a lot of images and we want that the statistics on the dev set be as
representative as possible, we'll take 20% of "train_signs" as dev set.
"""

import argparse
import random
import os

from PIL import Image
from tqdm import tqdm
import numpy as np


SIZE = 299

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='C:\data\\food_05_300x300', help="Directory with the SIGNS dataset")
parser.add_argument('--output_dir', default='data/food-10-300x300', help="Where to write the new data")


def class_to_index_mapping(data_dir):
    class_to_ix = {}
    ix_to_class = {}
    classes =[]
    with open(os.path.join(data_dir, "meta/classes.txt")) as txt:
        classes = [l.strip() for l in txt.readlines()]
        class_to_ix = dict(zip(classes, range(len(classes))))
        ix_to_class = dict(zip(range(len(classes)), classes))
        class_to_ix = {v: k for k, v in ix_to_class.items()}
    # sorted_class_to_ix = collections.OrderedDict(sorted(class_to_ix.items()))
        return class_to_ix, ix_to_class

def get_train_images_data(data_dir, opt):
    imagesPaths = []
    labels = []
    class_to_ix, ix_to_class = class_to_index_mapping(data_dir)
    classes_list = os.listdir(os.path.join(data_dir, opt))
    for d in classes_list:
        for f in os.listdir(os.path.join(data_dir, opt, d)):
            if f.endswith('.jpg'):
                labels.append(class_to_ix[d])
                imagesPaths.append(os.path.join(data_dir, opt, d, f))
    # Get the filenames in each directory (train and test)

    return imagesPaths, labels

def get_images_data(data_dir, opt):
    imagesPaths = []
    labels = []
    class_to_ix, ix_to_class = class_to_index_mapping(data_dir)
    with open(os.path.join(data_dir, "meta", opt + ".txt")) as t:
        train_data = t.read().splitlines()
        for d in train_data:
            class_name = d.split('/')[0]
            labels.append(class_to_ix[class_name])
            imagesPaths.append(os.path.join(data_dir, opt, d + ".jpg"))
    return imagesPaths, labels

def create_meta_data(source_data_dir, dist_data_dir):
    class_to_ix, ix_to_class = class_to_index_mapping(source_data_dir)
    classes_file = open(os.path.join(dist_data_dir, "meta/classes.txt"), "w")
    if not os.path.exists(dist_data_dir):
        os.mkdir(dist_data_dir)
        os.mkdir(os.path.join(dist_data_dir, "meta"))
    train_file = open(os.path.join(dist_data_dir, "meta/train.txt"), "w")
    with open(os.path.join(source_data_dir, "meta/train.txt"))as t:
        train_pahts = t.read().splitlines()
        for i in range(len(ix_to_class)):
            classes_file.write(ix_to_class[i]+"\n")
            for p in train_pahts:
                class_name = p.split('/')[0]
                if ix_to_class[i] == class_name:
                    train_file.write(p+"\n")
    test_file = open(os.path.join(dist_data_dir, "meta/test.txt"), "w")
    with open(os.path.join(source_data_dir, "meta/test.txt"))as t:
        test_pahts = t.read().splitlines()
        for i in range(len(ix_to_class)):
            for p in test_pahts:
                class_name = p.split('/')[0]
                if ix_to_class[i] == class_name:
                    test_file.write(p+"\n")

def resize_and_save(filename, output_dir, size=SIZE):
    """Resize the image contained in `filename` and save it to the `output_dir`"""
    image = Image.open(filename)
    # Use bilinear interpolation instead of the default "nearest neighbor" method
    image = image.resize((size, size), Image.BILINEAR)
    # print(os.path.join(output_dir, (filename.split('/')[-1]).split('\\')[-1]))
    image.save(os.path.join(output_dir, (filename.split('/')[-1]).split('\\')[-1]))


if __name__ == '__main__':
    args = parser.parse_args()

    assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)
    data_dir ="C:\data\\food_05_300x300"
    images, labels = get_train_images_data(data_dir, "train")
    print(len(images), len(labels))
"""
    # Create Meta data files
    create_meta_data(args.data_dir, args.output_dir)

    # Creat tow floder for train and test
    if not(os.path.exists(os.path.join(args.output_dir, "train"))):
        os.mkdir(os.path.join(args.output_dir, "train"))
        os.mkdir(os.path.join(args.output_dir, "test"))


    # Define the data directories
    all_data_dir = os.path.join(args.data_dir, 'images')
    train_data_dir = os.path.join(args.data_dir, 'train')
    test_data_dir = os.path.join(args.data_dir, 'test')

    # Get the filenames in each directory (train and test)

    train_filenames, train_labels = get_images_data(args.data_dir, "train")
    eval_filenames, eval_labels = get_images_data(args.data_dir, "test")

    # Preprocess train, dev and test
    idx = 0
    train_dir = os.path.join(args.output_dir, "train")
    test_dir = os.path.join(args.output_dir, "test")
    with open(os.path.join(args.output_dir, "meta/classes.txt")) as classes_list:
        classes_list = classes_list.read().splitlines()
        for c in classes_list:
            output_train_dir_split = os.path.join(train_dir, c)
            if not os.path.exists(output_train_dir_split):
                os.mkdir(output_train_dir_split)
            else:
                print("Warning: dir {} already exists".format(output_train_dir_split))
            print("Processing {} data, saving preprocessed data to")
            for filename in tqdm(train_filenames):
                class_name = filename.split('\\')[8].split('/')[0]
                print(class_name)
                if class_name == c:
                    resize_and_save(filename, output_train_dir_split, size=SIZE)

            # Copy and resize test images
            output_test_dir_split = os.path.join(test_dir, c)
            if not os.path.exists(output_test_dir_split):
                os.mkdir(output_test_dir_split)
            else:
                print("Warning: dir {} already exists".format(output_test_dir_split))
            for filename in tqdm(eval_filenames):
                if filename.split('\\')[8].split('/')[0] == c:
                    resize_and_save(filename, output_test_dir_split, size=SIZE)

    print("Done building dataset")
"""