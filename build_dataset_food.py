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
parser.add_argument('--data_dir', default='data/SIGNS', help="Directory with the SIGNS dataset")
parser.add_argument('--output_dir', default='data/64x64_SIGNS', help="Where to write the new data")

def get_images_data(data_dir, file_name, no_per_class, num_labels):
    imagesPaths = []

    with open(os.path.join(data_dir, "meta", file_name)) as t:
        train_data = t.read().splitlines()
        for d in train_data:
            imagesPaths.append(os.path.join(data_dir, "images", d + ".jpg"))
    labels = np.ones(len(imagesPaths), dtype='int32')
    labels[:no_per_class] = 0
    for i in range(num_labels - 1):
        labels[(i + 1) * no_per_class:(i + 2) * no_per_class] = i + 1
    # labels[params.num_labels * no_per_class:] = params.num_labels
    return imagesPaths, labels

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

    # Define the data directories
    train_data_dir = os.path.join(args.data_dir, 'train_signs')
    test_data_dir = os.path.join(args.data_dir, 'test_signs')

    # Get the filenames in each directory (train and test)

    train_filenames, train_labels = get_images_paths(args.data_dir, "train.txt", 750, 5)
    eval_filenames, eval_labels = get_images_paths(args.data_dir, "test.txt", 750, 5)

    '''
    filenames = os.listdir(train_data_dir)
    filenames = [os.path.join(train_data_dir, f) for f in filenames if f.endswith('.jpg')]

    test_filenames = os.listdir(test_data_dir)
    test_filenames = [os.path.join(test_data_dir, f) for f in test_filenames if f.endswith('.jpg')]

    # Split the images in 'train_signs' into 80% train and 20% dev
    # Make sure to always shuffle with a fixed seed so that the split is reproducible
    random.seed(230)
    filenames.sort()
    random.shuffle(filenames)

    split = int(0.8 * len(filenames))
    train_filenames = filenames[:split]
    dev_filenames = filenames[split:]

    filenames = {'train': train_filenames,
                 'dev': dev_filenames,
                 'test': test_filenames}

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    else:
        print("Warning: output dir {} already exists".format(args.output_dir))
    '''
    # Preprocess train, dev and test
    idx = 0
    with open(os.path.join(args.data_dir, "meta", "classes.txt")) as classes_list:
        classes_list = classes_list.read().splitlines()
        for c in classes_list:
            output_dir_split = os.path.join(args.output_dir, c)
            if not os.path.exists(output_dir_split):
                os.mkdir(output_dir_split)
            else:
                print("Warning: dir {} already exists".format(output_dir_split))

            print("Processing {} data, saving preprocessed data to")
            for filename in tqdm(train_filenames):
                if filename.split('/')[2].split('\\')[-1] == c:
                    resize_and_save(filename, output_dir_split, size=SIZE)
            for filename in tqdm(eval_filenames):
                if filename.split('/')[2].split('\\')[-1] == c:
                    resize_and_save(filename, output_dir_split, size=SIZE)

    print("Done building dataset")
