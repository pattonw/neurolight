import tensorflow as tf
import json
from funlib.learn.tensorflow.models import unet, conv_pass
import numpy as np

from typing import Dict, Any
from pathlib import Path


def mknet(config: Dict[str, Any], output_path: Path = Path()):

    input_shape = (2,) + tuple(config["INPUT_SHAPE"])
    embedding_dims = config["EMBEDDING_DIMS"]
    num_fmaps_embedding = config["NUM_FMAPS_EMBEDDING"]
    fmap_inc_factors_embedding = config["FMAP_INC_FACTORS_EMBEDDING"]
    downsample_factors = config["DOWNSAMPLE_FACTORS"]
    kernel_size_up = config["KERNEL_SIZE_UP"]

    raw = tf.placeholder(tf.float32, shape=input_shape, name="raw_input")
    raw_batched = tf.reshape(raw, (1,) + input_shape)

    with tf.variable_scope("embedding"):
        embedding_unet = unet(
            raw_batched,
            num_fmaps=num_fmaps_embedding,
            fmap_inc_factors=fmap_inc_factors_embedding,
            downsample_factors=downsample_factors,
            kernel_size_up=kernel_size_up,
            constant_upsample=True,
        )

    embedding_batched = conv_pass(
        embedding_unet[0],
        kernel_sizes=[1],
        num_fmaps=embedding_dims,
        activation=None,
        name="embedding",
    )

    embedding_norms = tf.norm(embedding_batched[0], axis=1, keep_dims=True)
    embedding_scaled = embedding_batched[0] / embedding_norms

    output_shape_batched = embedding_scaled.get_shape().as_list()
    output_shape = tuple(
        output_shape_batched[2:]
    )  # strip the batch and channel dimension

    assert all(
        np.isclose(np.array(output_shape), np.array(config["OUTPUT_SHAPE"]))
    ), "output shapes don't match"

    embedding = tf.reshape(embedding_scaled, (embedding_dims,) + output_shape)

    tf.train.export_meta_graph(filename=output_path / "train_net_embedding.meta")
    names = {"raw": raw.name, "embedding": embedding.name}
    with (output_path / "tensor_names.json").open("w") as f:
        json.dump(names, f)
