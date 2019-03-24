"""Train the model"""

import argparse
import logging
import os
import random
from imutils import paths
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
# from build_dataset_food import get_images_data, get_train_images_data
from model.input_fn import input_fn, get_labels
from model.utils import Params
from model.utils import set_logger
from model.utils import save_dict_to_json
from model.model_fn import model_fn
from model.training import train_and_evaluate
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--device_name', default='cpu')
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default='c:\data\\food_05_300x300\\all-train',
                    help="Directory containing the dataset")
parser.add_argument('--restore_from', default=None,
                    help="Optional, directory or file containing weights to reload before training")


if __name__ == '__main__':
    # Set the random seed for the whole graph for reproductible experiments
    tf.set_random_seed(230)

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Check that we are not overwriting some previous experiment
    # Comment these lines if you are developing your model and don't care about overwritting
    model_dir_has_best_weights = os.path.isdir(os.path.join(args.model_dir, "best_weights"))
    overwritting = model_dir_has_best_weights and args.restore_from is None
    assert not overwritting, "Weights found in model_dir, aborting to avoid overwrite"

    # Set the logger
    set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Creating the datasets...")
    data_dir = args.data_dir

    # Get the filenames from the train and dev sets
    image_paths = sorted(list(paths.list_images(data_dir)))
    random.seed(42)
    random.shuffle(image_paths)

    # binarize the labels
    # lb = LabelBinarizer()
    # labels = lb.fit_transform(labels)

    # partition the data into training and testing splits using 80% of
    # the data for training and the remaining 20% for testing
    split = int(0.8 * len(image_paths))
    train_filenames = image_paths[:split]
    eval_filenames = image_paths[split:]
    # (train_filenames, eval_filenames, train_labels, eval_labels) = train_test_split(image_paths,
    #                                                 labels, test_size=0.2, random_state=42)
    print(len(train_filenames), len(eval_filenames))
    classes_list = os.listdir(data_dir)
    train_labels = get_labels(train_filenames, classes_list)
    eval_labels = get_labels(eval_filenames, classes_list)
    # eval_filenames, eval_labels = get_images_data(data_dir, "dev")
    # eval_filenames, eval_labels = get_train_images_data(data_dir, "test")
    # Specify the sizes of the dataset we train on and evaluate on
    params.train_size = len(train_filenames)
    params.eval_size = len(eval_filenames)

    # Create the two iterators over the two datasets
    train_inputs = input_fn(True, train_filenames, train_labels, params)
    eval_inputs = input_fn(False, eval_filenames, eval_labels, params)
    # Define the model
    logging.info("Creating the model...")
    train_model_spec = model_fn('train', train_inputs, params)
    eval_model_spec = model_fn('eval', eval_inputs, params, reuse=True)

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(train_model_spec, eval_model_spec, args.model_dir, params, args.restore_from)
