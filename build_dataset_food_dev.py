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
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm
import numpy as np


SIZE = 299

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='D:\Metwalli\master\\reasearches proposals\Computer vision\Materials\\food41', help="Directory with the SIGNS dataset")
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

def create_meta_data(source_data_dir, dist_data_dir, filenames):
    class_to_ix, ix_to_class = class_to_index_mapping(source_data_dir)
    classes_file = open(os.path.join(dist_data_dir, "meta/classes.txt"), "w")
    if not os.path.exists(dist_data_dir):
        os.mkdir(dist_data_dir)
        os.mkdir(os.path.join(dist_data_dir, "meta"))
    train_file = open(os.path.join(dist_data_dir, "train.txt"), "w")
    with open(os.path.join(source_data_dir, "meta/train.txt"))as t:
        train_pahts = t.read().splitlines()
        for i in range(len(ix_to_class)):
            classes_file.write(ix_to_class[i]+"\n")
            for p in train_pahts:
                class_name = p.split('/')[0]
                if ix_to_class[i] == class_name:
                    train_file.write(p+"\n")
    test_file = open(os.path.join(dist_data_dir, "dev.txt"), "w")
    with open(os.path.join(source_data_dir, "meta/dev.txt"))as t:
        test_pahts = t.read().splitlines()
        for i in range(len(ix_to_class)):
            for p in test_pahts:
                class_name = p.split('/')[0]
                if ix_to_class[i] == class_name:
                    test_file.write(p+"\n")

def resize_and_save(full_path, filename, output_dir, size=SIZE):
    """Resize the image contained in `filename` and save it to the `output_dir`"""
    image = Image.open(full_path)
    # Use bilinear interpolation instead of the default "nearest neighbor" method
    # image = image.resize((size, size), Image.BILINEAR)
    # print(os.path.join(output_dir, (filename.split('/')[-1]).split('\\')[-1]))
    image.save(os.path.join(output_dir, filename, ".jpg"))

def split_images(class_name, opt):
    # Creat Dir for Class in specified opt {train, dev or test}
    if not os.path.exists(os.path.join(args.output_dir, opt, class_name)):
        os.mkdir(os.path.join(args.output_dir, opt, class_name))
    with open(os.path.join(args.output_dir, "meta", opt, ".txt")) as t:
        filenames = t.read().splitlines()
        print("Processing {} data, saving preprocessed data to")
        for filename in tqdm(filenames):
            if filename.split('/')[0] == class_name:
                full_path = os.path.join(args.data_dir, "images", filename + ".jpg")
                resize_and_save(full_path, filename, os.path.join(args.output_dir, opt), SIZE)

if __name__ == '__main__':
    args = parser.parse_args()

    assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)
    if not (os.path.exists(os.path.join(args.output_dir, "train"))):
        os.mkdir(os.path.join(args.output_dir, "train"))
        os.mkdir(os.path.join(args.output_dir, "dev"))
        os.mkdir(os.path.join(args.output_dir, "test"))
        os.mkdir(os.path.join(args.output_dir, "meta"))
    # Split Train Data to train and dev and copy to output dir
    with open(os.path.join(args.data_dir, "meta/train.txt")) as train_list:
        split = int(0.8 * len(train_list))
        train_filenames = train_list[:split]
        dev_filenames = train_list[split:]
        train_file = open(os.path.join(args.output_dir, "meta/train.txt"), "w")
        train_file.write(train_filenames)
        dev_file = open(os.path.join(args.output_dir, "meta/dev.txt"), "w")
        dev_file.write(dev_filenames)
    # copy classes and test to output dir
    shutil.copy(os.path.join(args.data_dir, "meta/test.txt"), os.path.join(args.output_dir, "meta"))
    shutil.copy(os.path.join(args.data_dir, "meta/classes.txt"), os.path.join(args.output_dir, "meta"))


    # Define the data directories
    image_data_dir = os.path.join(args.data_dir, 'images')
    train_data_dir = os.path.join(args.data_dir, 'train')
    test_data_dir = os.path.join(args.data_dir, 'test')

    # Apply Data Augmentation


    with open(os.path.join(args.data_dir, "meta/classes.txt")) as classes_list:
        classes_list = classes_list.read().splitlines()
        for c in classes_list:
            split_images(c, "train")
            split_images(c, "dev")
            split_images(c, "test")

    print("Done building dataset")