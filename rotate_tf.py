import math
import tensorflow as tf

def transform(image, inv_mat, image_shape):

      h, w, c = image_shape
      cx, cy = w//2, h//2

      new_xs = tf.repeat( tf.range(-cx, cx, 1), h)
      new_ys = tf.tile( tf.range(-cy, cy, 1), [w])
      new_zs = tf.ones([h*w], dtype=tf.int32)

      old_coords = tf.matmul(inv_mat, tf.cast(tf.stack([new_xs, new_ys, new_zs]), tf.float32))
      old_coords_x, old_coords_y = tf.round(old_coords[0, :] + w//2), tf.round(old_coords[1, :] + h//2)

      clip_mask_x = tf.logical_or(old_coords_x<0, old_coords_x>w-1)
      clip_mask_y = tf.logical_or(old_coords_y<0, old_coords_y>h-1)
      clip_mask = tf.logical_or(clip_mask_x, clip_mask_y)

      old_coords_x = tf.boolean_mask(old_coords_x, tf.logical_not(clip_mask))
      old_coords_y = tf.boolean_mask(old_coords_y, tf.logical_not(clip_mask))
      new_coords_x = tf.boolean_mask(new_xs+cx, tf.logical_not(clip_mask))
      new_coords_y = tf.boolean_mask(new_ys+cy, tf.logical_not(clip_mask))

      old_coords = tf.cast(tf.stack([old_coords_y, old_coords_x]), tf.int32)
      new_coords = tf.cast(tf.stack([new_coords_y, new_coords_x]), tf.int64)
      rotated_image_values = tf.gather_nd(image, tf.transpose(old_coords))
      rotated_image_channel = list()
      for i in range(c):
          vals = rotated_image_values[:,i]
          sparse_channel = tf.SparseTensor(tf.transpose(new_coords), vals, [h, w])
          rotated_image_channel.append(tf.sparse.to_dense(sparse_channel, default_value=0, validate_indices=False))

      return tf.transpose(tf.stack(rotated_image_channel), [1,2,0])

def random_rotate(image, angle, image_shape):

    def get_rotation_mat_inv(angle):
          #transform to radian
          angle = math.pi * angle / 180

          cos_val = tf.math.cos(angle)
          sin_val = tf.math.sin(angle)
          one = tf.constant([1], tf.float32)
          zero = tf.constant([0], tf.float32)

          rot_mat_inv = tf.concat([cos_val, sin_val, zero,
                                     -sin_val, cos_val, zero,
                                     zero, zero, one], axis=0)
          rot_mat_inv = tf.reshape(rot_mat_inv, [3,3])

          return rot_mat_inv
    angle = float(angle) * tf.random.normal([1],dtype='float32')
    rot_mat_inv = get_rotation_mat_inv(angle)
    return transform(image, rot_mat_inv, image_shape)
