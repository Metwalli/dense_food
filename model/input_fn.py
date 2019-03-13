"""Create the input data pipeline using `tf.data`"""

import tensorflow as tf
import math

def _parse_function(filename, label, size):
    """Obtain the image from the filename (for both training and validation).

    The following operations are applied:
        - Decode the image from jpeg format
        - Convert to float and to range [0, 1]
    """
    image_string = tf.read_file(filename)

    # Don't use tf.image.decode_image, or the output shape will be undefined
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)

    # This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image_decoded, tf.float32)

    resized_image = tf.image.resize_images(image, [size, size])

    return resized_image, label


def train_preprocess(image, label, use_random_flip):
    """Image preprocessing for training.

    Apply the following operations:
        - Horizontally flip the image with probability 1/2
        - Apply random brightness and saturation
    """
    if use_random_flip:
        image = tf.image.random_flip_left_right(image)

    # image = tf.image.central_crop(image, 0.3)
    image = tf.image.random_crop(image, image.shape)
    image = tf.image.rot90(image, 1)
    # image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
    # image = tf.image.random_saturation(image, lower=0.5, upper=1.5)

    # Make sure the image is still in [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, label


def augment(images, labels,
            resize=None,  # (width, height) tuple or None
            horizontal_flip=False,
            vertical_flip=False,
            rotate=0,  # Maximum rotation angle in degrees
            crop_probability=0,  # How often we do crops
            crop_min_percent=0.6,  # Minimum linear dimension of a crop
            crop_max_percent=1.,  # Maximum linear dimension of a crop
            mixup=0):  # Mixup coeffecient, see https://arxiv.org/abs/1710.09412.pdf
    if resize is not None:
        images = tf.image.resize_bilinear(images, resize)

    # My experiments showed that casting on GPU improves training performance
    if images.dtype != tf.float32:
        images = tf.image.convert_image_dtype(images, dtype=tf.float32)
        images = tf.subtract(images, 0.5)
        images = tf.multiply(images, 2.0)
    labels = tf.to_float(labels)

    with tf.name_scope('augmentation'):
        shp = tf.shape(images)
        batch_size, height, width = shp[0], shp[1], shp[2]
        width = tf.cast(width, tf.float32)
        height = tf.cast(height, tf.float32)

        # The list of affine transformations that our image will go under.
        # Every element is Nx8 tensor, where N is a batch size.
        transforms = []
        identity = tf.constant([1, 0, 0, 0, 1, 0, 0, 0], dtype=tf.float32)
        if horizontal_flip:
            coin = tf.less(tf.random_uniform([batch_size], 0, 1.0), 0.5)
            flip_transform = tf.convert_to_tensor(
                [-1., 0., width, 0., 1., 0., 0., 0.], dtype=tf.float32)
            transforms.append(
                tf.where(coin,
                         tf.tile(tf.expand_dims(flip_transform, 0), [batch_size, 1]),
                         tf.tile(tf.expand_dims(identity, 0), [batch_size, 1])))

        if vertical_flip:
            coin = tf.less(tf.random_uniform([batch_size], 0, 1.0), 0.5)
            flip_transform = tf.convert_to_tensor(
                [1, 0, 0, 0, -1, height, 0, 0], dtype=tf.float32)
            transforms.append(
                tf.where(coin,
                         tf.tile(tf.expand_dims(flip_transform, 0), [batch_size, 1]),
                         tf.tile(tf.expand_dims(identity, 0), [batch_size, 1])))

        if rotate > 0:
            angle_rad = rotate / 180 * math.pi
            angles = tf.random_uniform([batch_size], -angle_rad, angle_rad)
            transforms.append(
                tf.contrib.image.angles_to_projective_transforms(
                    angles, height, width))

        if crop_probability > 0:
            crop_pct = tf.random_uniform([batch_size], crop_min_percent,
                                         crop_max_percent)
            left = tf.random_uniform([batch_size], 0, width * (1 - crop_pct))
            top = tf.random_uniform([batch_size], 0, height * (1 - crop_pct))
            crop_transform = tf.stack([
                crop_pct,
                tf.zeros([batch_size]), top,
                tf.zeros([batch_size]), crop_pct, left,
                tf.zeros([batch_size]),
                tf.zeros([batch_size])
            ], 1)

            coin = tf.less(
                tf.random_uniform([batch_size], 0, 1.0), crop_probability)
            transforms.append(
                tf.where(coin, crop_transform,
                         tf.tile(tf.expand_dims(identity, 0), [batch_size, 1])))

        if transforms:
            images = tf.contrib.image.transform(
                images,
                tf.contrib.image.compose_transforms(*transforms),
                interpolation='BILINEAR')  # or 'NEAREST'

        def cshift(values):  # Circular shift in batch dimension
            return tf.concat([values[-1:, ...], values[:-1, ...]], 0)

        if mixup > 0:
            mixup = 1.0 * mixup  # Convert to float, as tf.distributions.Beta requires floats.
            beta = tf.distributions.Beta(mixup, mixup)
            lam = beta.sample(batch_size)
            ll = tf.expand_dims(tf.expand_dims(tf.expand_dims(lam, -1), -1), -1)
            images = ll * images + (1 - ll) * cshift(images)
            labels = lam * labels + (1 - lam) * cshift(labels)

    return images, labels

def input_fn(is_training, filenames, labels, params):
    """Input function for the SIGNS dataset.

    The filenames have format "{label}_IMG_{id}.jpg".
    For instance: "data_dir/2_IMG_4584.jpg".

    Args:
        is_training: (bool) whether to use the train or test pipeline.
                     At training, we shuffle the data and have multiple epochs
        filenames: (list) filenames of the images, as ["data_dir/{label}_IMG_{id}.jpg"...]
        labels: (list) corresponding list of labels
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """
    num_samples = len(filenames)
    assert len(filenames) == len(labels), "Filenames and labels should have same length"

    # Create a Dataset serving batches of images and labels
    # We don't repeat for multiple epochs because we always train and evaluate for one epoch
    parse_fn = lambda f, l: _parse_function(f, l, params.image_size)
    train_fn = lambda f, l: augment(images=f, labels=l, horizontal_flip=True, rotate=15, crop_probability=0.8, mixup=4)

    if is_training:
        dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(filenames), tf.constant(labels)))
            .shuffle(num_samples)  # whole dataset into the buffer ensures good shuffling
            .map(parse_fn, num_parallel_calls=params.num_parallel_calls)
            .map(train_fn, num_parallel_calls=params.num_parallel_calls)
            .batch(params.batch_size)
            .prefetch(1)  # make sure you always have one batch ready to serve
        )
    else:
        dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(filenames), tf.constant(labels)))
            .map(parse_fn)
            .batch(params.batch_size)
            .prefetch(1)  # make sure you always have one batch ready to serve
        )

    # Create reinitializable iterator from dataset
    iterator = dataset.make_initializable_iterator()
    images, labels = iterator.get_next()
    iterator_init_op = iterator.initializer

    inputs = {'images': images, 'labels': labels, 'iterator_init_op': iterator_init_op}
    return inputs
