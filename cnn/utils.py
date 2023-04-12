import tensorflow as tf
from keras.models import load_model
import numpy as np


def latent_vector(no_of_samples, latent_dim=100, n_cats=10):
    """Sample points in latent space for input to the generator"""
    # Generate points in the latent space
    latent_input = np.random.randn(latent_dim * no_of_samples)
    # Reshape into a batch of inputs for the network
    latent_input = latent_input.reshape(no_of_samples, latent_dim)
    # Generate category labels
    cat_labels = np.random.randint(0, n_cats, no_of_samples)
    return [latent_input, cat_labels]


def generate_gan_samples(dataset_name, no_of_samples):
    """generate samples from the generator using regular labels (0-9)"""
    # Generate latent points
    latent_points, _ = latent_vector(no_of_samples)
    n_cats = 10
    repeated = int(no_of_samples / n_cats)
    # Specify labels to generate (0-9 repeated "repeated" times)
    labels = np.asarray([x for _ in range(repeated) for x in range(10)], dtype=np.int64)
    # Load previously saved generator model
    model = load_model("./{}_gan.h5".format(dataset_name))
    # Generate images
    gen_imgs = model.predict([latent_points, labels])
    # Scale from [-1, 1] to [0, 1]
    gen_imgs = (gen_imgs + 1) / 2.0
    # Scale from [0,1] to [0,255]
    gen_imgs = (gen_imgs * 255).astype("uint8")
    return tf.data.Dataset.from_tensor_slices((gen_imgs, labels))


def generate_gan_samples_onehot(dataset_name, no_of_samples):
    """generate samples from the generator using one-hot encoded labels"""
    noise = tf.random.normal([no_of_samples, 100])
    n_cats = 10
    repeated = int(no_of_samples / n_cats)
    labels_original = np.asarray(
        [x for _ in range(repeated) for x in range(10)], dtype=np.int64
    )
    labels = tf.one_hot(labels_original, 10)
    generator = load_model("./{}_generator.h5".format(dataset_name))
    predictions = generator.predict([noise, labels])
    predictions = (predictions + 1) / 2.0
    predictions = tf.cast(predictions * 255, tf.uint8)
    return tf.data.Dataset.from_tensor_slices((predictions, labels_original))
