import tensorflow as tf
import matplotlib.pyplot as plt
from gridmask_tf import GridMask

def apply_grid_mask(image, image_shape, d1, d2, rotate_angle, ratio):
    mask = GridMask(image_shape[0], image_shape[1], d1, d2, rotate_angle, ratio)
    if image_shape[-1] == 3:
        mask = tf.concat([mask, mask, mask], axis=-1)

    return image * tf.cast(mask, tf.uint8)


def show_before_after(image, image_masked):
    plt.figure(figsize=(10,6))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1, 2, 2)
    plt.imshow(image_masked)
    plt.xticks([])
    plt.yticks([])
    plt.show()

if __name__ == '__main__':
    daisy_bits = tf.io.read_file('./imgs/daisy.png')
    dog_bits = tf.io.read_file('./imgs/dog.png')

    daisy = tf.image.decode_png(daisy_bits)
    dog = tf.image.decode_png(dog_bits)

    daisy_masked = apply_grid_mask(daisy, daisy.get_shape().as_list(), 90, 150, 23, 0.5)
    dog_masked = apply_grid_mask(dog, dog.get_shape().as_list(), 90, 150, 23, 0.5)

    show_before_after(daisy, daisy_masked)
    show_before_after(dog, dog_masked)
