# Image preprocessing in Keras

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
from tqdm import tqdm
from build_dataset_food_dev import get_images_data
datagen = ImageDataGenerator(
        rotation_range=45,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        fill_mode='nearest')

def img_augmentation(img, output_dir, filename):
    x = img_to_array(img)  # convert image to numpy array
    x = x.reshape((1,) + x.shape)  # reshape image to (1, ..,..,..) to fit keras' standard shape

    # Use flow() to apply data augmentation randomly according to the datagenerator
    # and saves the results to the `preview/` directory
    num_image_generated = 0
    for batch in datagen.flow(x, batch_size=1, save_to_dir=output_dir, save_prefix=filename, save_format='jpg'):
        break
        # num_image_generated += 1
        # if num_image_generated == 2:
        #     break # stop the loop after num_image_generated iterations

data_dir = "C:\data\\food_05_300x300"
train_filenames, labels = get_images_data(data_dir, "train")

with open(os.path.join(data_dir, "meta/classes.txt")) as classes_list:
    classes_list = classes_list.read().splitlines()
    for c in classes_list:
        output_train_dir_split = os.path.join(data_dir, "train", c)
        print("Processing {} data, saving preprocessed data to")
        for filename in tqdm(train_filenames):
            class_name = filename.split('\\')[4].split('/')[0]
            prefix = filename.split('\\')[4].split('/')[1].split('.')[0]
            # print(class_name)
            if class_name == c:
                img = load_img(filename)
                img_augmentation(img, output_train_dir_split, prefix)

        # Copy and resize test images
