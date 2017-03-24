from re import T

import numpy as np
from functools import partial
import PIL.Image
import tensorflow as tf
import matplotlib.pyplot as plt
import urllib.request
import os
import zipfile
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def main():
    model_fn='tensorflow_inception_graph.pb'
    img_noise = np.random.uniform(size=(224, 224, 3)) + 100.0
    # Step 2 - Creating Tensorflow session and loading the model
    graph = tf.Graph()
    sess = tf.InteractiveSession(graph=graph)
    with tf.gfile.FastGFile(model_fn) as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    t_input = tf.placeholder(np.float32, name='input')  # define the input tensor
    imagenet_mean = 117.0
    t_preprocessed = tf.expand_dims(t_input - imagenet_mean, 0)
    tf.import_graph_def(graph_def, {'input': t_preprocessed})

    layers = [op.name for op in graph.get_operations() if op.type == 'Conv2D' and 'import/' in op.name]
    feature_nums = [int(graph.get_tensor_by_name(name + ':0').get_shape()[-1]) for name in layers]

    print('Number of layers', len(layers))
    print('Total number of feature channels:', sum(feature_nums))


    def showarray(a):
        a = np.uint8(np.clip(a, 0, 1) * 255)
        plt.imshow(a)
        plt.show()

    def calc_grad_tiled(img, t_grad, tile_size=512):
        '''Compute the value of tensor t_grad over the image in a tiled way.
        Random shifts are applied to the image to blur tile boundaries over
        multiple iterations.'''
        sz = tile_size
        h, w = img.shape[:2]
        sx, sy = np.random.randint(sz, size=2)
        img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
        grad = np.zeros_like(img)
        for y in range(0, max(h-sz//2, sz),sz):
            for x in range(0, max(w-sz//2, sz),sz):
                sub = img_shift[y:y+sz,x:x+sz]
                g = sess.run(t_grad, {t_input:sub})
                grad[y:y+sz,x:x+sz] = g
        return np.roll(np.roll(grad, -sx, 1), -sy, 0)


    def render_deepdream(t_obj, img0=img_noise,iter_n=10, step=1.5, octave_n=4, octave_scale=1.4):

            t_score = tf.reduce_mean(t_obj)  # defining the optimization objective
            t_grad = tf.gradients(t_score, t_input)[0]  # behold the power of automatic differentiation!

            # split the image into a number of octaves
            img = img0
            octaves = []
            for _ in range(octave_n - 1):
                hw = img.shape[:2]
                lo = np.resize(img, np.int32(np.float32(hw) / octave_scale))
                hi = img - np.resize(lo, hw)
                img = lo
                octaves.append(hi)

            # generate details octave by octave
            for octave in range(octave_n):
                if octave > 0:
                    hi = octaves[-octave]
                    img = np.resize(img, hi.shape[:2]) + hi
                for _ in range(iter_n):
                    g = calc_grad_tiled(img, t_grad)
                    img += g * (step / (np.abs(g).mean() + 1e-7))

                # this will usually be like 3 or 4 octaves
                # Step 5 output deep dream image via matplotlib
                showarray(img / 255.0)

            # Step 3 - Pick a layer to enhance our image
            layer = 'mixed4d_3x3_bottleneck_pre_relu'
            channel = 139  # picking some feature channel to visualize

            # open image
            img0 = PIL.Image.open('pilatus800.jpg')
            img0 = np.float32(img0)

            # Step 4 - Apply gradient ascent to that layer
            plt.imshow(render_deepdream(tf.square(T('mixed4c')), img0))
            plt.show()

if __name__ == '__main__':
        main()
