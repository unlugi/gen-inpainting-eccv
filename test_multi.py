import argparse

import cv2
import numpy as np
import tensorflow as tf
import multiprocessing
import os

import neuralgym as ng
from inpaint_model import InpaintCAModel


parser = argparse.ArgumentParser()
parser.add_argument('--image_dir', default='', type=str,
                    help='The folder containing images to be completed.')
parser.add_argument('--mask_dir', default='', type=str,
                    help='The folder containing masks, value 255 indicates mask.')
parser.add_argument('--output_dir', default='', type=str,
                    help='Where to write output.')
parser.add_argument('--checkpoint_dir', default='', type=str,
                    help='The directory of tensorflow checkpoint.')

def complete(image_file):
    ng.get_gpus(1,verbose=False)
    tf.reset_default_graph()
    model = InpaintCAModel()
    image = cv2.imread(os.path.join(args.image_dir, image_file))
    mask = cv2.imread(os.path.join(args.mask_dir, image_file))
    assert image.shape == mask.shape

    h, w, _ = image.shape
    grid = 8
    image_rs = image[:h // grid * grid, :w // grid * grid, :]
    mask_rs = mask[:h // grid * grid, :w // grid * grid, :]
    print('Shape of image: {}'.format(image_rs.shape))

    image_rs = np.expand_dims(image_rs, 0)
    mask_rs = np.expand_dims(mask_rs, 0)
    input_image = np.concatenate([image_rs, mask_rs], axis=2)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
        input_image = tf.constant(input_image, dtype=tf.float32)
        output = model.build_server_graph(input_image)
        output = (output + 1.) * 127.5
        output = tf.reverse(output, [-1])
        output = tf.saturate_cast(output, tf.uint8)
        # load pretrained model
        vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        assign_ops = []
        for var in vars_list:
            vname = var.name
            from_name = vname
            var_value = tf.contrib.framework.load_variable(args.checkpoint_dir, from_name)
            assign_ops.append(tf.assign(var, var_value))
        sess.run(assign_ops)

        result = sess.run(output)

        image[:h // grid * grid, :w // grid * grid, :] = result[0][:, :, ::-1]
        save_value = cv2.imwrite(os.path.join(args.output_dir, image_file), image)
        print("Image saved:", save_value)
        sess.close()


if __name__ == '__main__':
    args = parser.parse_args()

    image_files = sorted(os.listdir(args.image_dir))
    mask_files = sorted(os.listdir(args.mask_dir))
    print("places2-256 finetune people mask50 prediction")

    for i in range(len(image_files)):
        image_file = image_files[i]
        mask_file = mask_files[i]
        p = multiprocessing.Process(target=complete, args=(image_file,))
        p.start()
        p.join()



