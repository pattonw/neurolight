import tensorflow as tf
import json
from funlib.learn.tensorflow.models import unet, conv_pass
import numpy as np

from typing import Dict, Any
from pathlib import Path


def mknet(config: Dict[str, Any], output_path: Path = Path()):
    if config["DISTANCES"]:
        mknet_distances(config, output_path)
    else:
        mknet_binary(config, output_path)


def mknet_distances(config: Dict[str, Any], output_path: Path = Path()):

    input_shape = (2,) + tuple(config["INPUT_SHAPE"])
    num_fmaps_foreground = config["NUM_FMAPS_FOREGROUND"]
    fmap_inc_factors_foreground = config["FMAP_INC_FACTORS_FOREGROUND"]
    downsample_factors = config["DOWNSAMPLE_FACTORS"]
    kernel_size_up = config["KERNEL_SIZE_UP"]

    output_shape = np.array(config["OUTPUT_SHAPE"])

    raw = tf.placeholder(tf.float32, shape=input_shape, name="raw_input")
    gt_distances = tf.placeholder(tf.int64, shape=output_shape, name="gt_distances")
    loss_weights = tf.placeholder(tf.int64, shape=output_shape, name="loss_weights")

    raw_batched = tf.reshape(raw, (1,) + input_shape)

    with tf.variable_scope("dist"):
        dist_unet = unet(
            raw_batched,
            num_fmaps=num_fmaps_foreground,
            fmap_inc_factors=fmap_inc_factors_foreground,
            downsample_factors=downsample_factors,
            kernel_size_up=kernel_size_up,
            constant_upsample=True,
        )

    dist_batched = conv_pass(
        dist_unet[0], kernel_sizes=[1], num_fmaps=1, activation="sigmoid"
    )[0]

    output_shape_batched = dist_batched.get_shape().as_list()
    # strip the batch and channel dimension
    output_shape = tuple(output_shape_batched[2:])

    assert all(np.isclose(np.array(output_shape), np.array(config["OUTPUT_SHAPE"])))

    fg_dist = tf.reshape(dist_batched[0], output_shape, name="fg_dist")

    fg_loss = tf.losses.mean_squared_error(gt_distances, fg_dist, loss_weights)

    opt = tf.train.AdamOptimizer(
        learning_rate=0.5e-5, beta1=0.95, beta2=0.999, epsilon=1e-8
    )

    optimizer = opt.minimize(fg_loss)

    tf.summary.scalar("fg_loss", fg_loss)

    summaries = tf.summary.merge_all()

    tf.train.export_meta_graph(filename=output_path / "train_net_foreground.meta")
    names = {
        "raw": raw.name,
        "gt_distances": gt_distances.name,
        "gt_fg": gt_distances.name,
        "fg_pred": fg_dist.name,
        "fg_loss": fg_loss.name,
        "optimizer": optimizer.name,
        "loss_weights": loss_weights.name,
        "summaries": summaries.name,
    }

    with (output_path / "tensor_names.json").open("w") as f:
        json.dump(names, f)


def mknet_binary(config: Dict[str, Any], output_path: Path = Path()):

    input_shape = (2,) + tuple(config["INPUT_SHAPE"])
    num_fmaps_foreground = config["NUM_FMAPS_FOREGROUND"]
    fmap_inc_factors_foreground = config["FMAP_INC_FACTORS_FOREGROUND"]
    downsample_factors = config["DOWNSAMPLE_FACTORS"]
    kernel_size_up = config["KERNEL_SIZE_UP"]

    output_shape = np.array(config["OUTPUT_SHAPE"])

    raw = tf.placeholder(tf.float32, shape=input_shape, name="raw_input")
    loss_weights = tf.placeholder(tf.float32, shape=output_shape, name="loss_weights")
    gt_labels = tf.placeholder(tf.int64, shape=output_shape, name="gt_labels")

    raw_batched = tf.reshape(raw, (1,) + input_shape)

    with tf.variable_scope("fg"):
        fg_unet = unet(
            raw_batched,
            num_fmaps=num_fmaps_foreground,
            fmap_inc_factors=fmap_inc_factors_foreground,
            downsample_factors=downsample_factors,
            kernel_size_up=kernel_size_up,
            constant_upsample=True,
        )

    fg_batched = conv_pass(fg_unet[0], kernel_sizes=[1], num_fmaps=1, activation=None)[
        0
    ]

    output_shape_batched = fg_batched.get_shape().as_list()
    output_shape = tuple(
        output_shape_batched[2:]
    )  # strip the batch and channel dimension

    assert all(
        np.isclose(np.array(output_shape), np.array(config["OUTPUT_SHAPE"]))
    ), "output shapes don't match"

    fg_logits = tf.reshape(fg_batched[0], output_shape, name="fg_logits")
    fg = tf.sigmoid(fg_logits, name="fg")
    gt_fg = tf.not_equal(gt_labels, 0, name="gt_fg")

    fg_loss = tf.losses.sigmoid_cross_entropy(
        gt_fg, fg_logits, weights=loss_weights, scope="fg_loss"
    )

    opt = tf.train.AdamOptimizer(
        learning_rate=0.5e-5, beta1=0.95, beta2=0.999, epsilon=1e-8
    )

    optimizer = opt.minimize(fg_loss)

    tf.summary.scalar("fg_loss", fg_loss)

    summaries = tf.summary.merge_all()

    tf.train.export_meta_graph(filename=output_path / "train_net_foreground.meta")
    names = {
        "raw": raw.name,
        "gt_labels": gt_labels.name,
        "gt_fg": gt_fg.name,
        "fg_pred": fg.name,
        "fg_logits": fg_logits.name,
        "loss_weights": loss_weights.name,
        "fg_loss": fg_loss.name,
        "optimizer": optimizer.name,
        "summaries": summaries.name,
    }

    with (output_path / "tensor_names.json").open("w") as f:
        json.dump(names, f)
