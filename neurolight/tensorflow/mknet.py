import tensorflow as tf
import json
from funlib.learn.tensorflow.models import unet, conv_pass
import numpy as np

from typing import Dict, Any


def mknet(config: Dict[str, Any]):

    INPUT_SHAPE = tuple(config["INPUT_SHAPE"])
    EMBEDDING_DIMS = config["EMBEDDING_DIMS"]
    NUM_FMAPS = config["NUM_FMAPS"]
    FMAP_INC_FACTORS = config["FMAP_INC_FACTORS"]

    raw = tf.placeholder(tf.float32, shape=INPUT_SHAPE)
    raw_batched = tf.reshape(raw, (1, 1) + INPUT_SHAPE)

    with tf.variable_scope("embedding"):
        embedding_unet = unet(
            raw_batched,
            num_fmaps=NUM_FMAPS,
            fmap_inc_factors=FMAP_INC_FACTORS,
            downsample_factors=[[2, 2], [2, 2], [2, 2]],
            kernel_size_up=[[3], [3], [3]],
            constant_upsample=True,
        )
    with tf.variable_scope("fg"):
        fg_unet = unet(
            raw_batched,
            num_fmaps=6,
            fmap_inc_factors=3,
            downsample_factors=[[2, 2], [2, 2], [2, 2]],
            kernel_size_up=[[3], [3], [3]],
            constant_upsample=True,
        )

    embedding_batched = conv_pass(
        embedding_unet[0],
        kernel_sizes=[1],
        num_fmaps=EMBEDDING_DIMS,
        activation=None,
        name="embedding",
    )

    embedding_norms = tf.norm(embedding_batched[0], axis=1, keep_dims=True)
    embedding_scaled = embedding_batched[0] / embedding_norms

    fg_batched = conv_pass(
        fg_unet[0], kernel_sizes=[1], num_fmaps=1, activation="sigmoid", name="fg"
    )

    output_shape_batched = embedding_scaled.get_shape().as_list()
    output_shape = tuple(
        output_shape_batched[2:]
    )  # strip the batch and channel dimension

    assert all(
        np.isclose(np.array(output_shape), np.array(config["OUTPUT_SHAPE"]))
    )

    embedding = tf.reshape(embedding_scaled, (EMBEDDING_DIMS,) + output_shape)
    fg = tf.reshape(fg_batched[0], output_shape)
    gt_labels = tf.placeholder(tf.int64, shape=output_shape)

    tf.train.export_meta_graph(filename="train_net.meta")
    names = {
        "raw": raw.name,
        "embedding": embedding.name,
        "fg": fg.name,
        "gt_labels": gt_labels.name,
    }
    with open("tensor_names.json", "w") as f:
        json.dump(names, f)
