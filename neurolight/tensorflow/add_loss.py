from .nms import max_detection

from funlib.learn.tensorflow.losses import ultrametric_loss_op

import tensorflow as tf
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

emst_name = "PyFuncStateless:0"
edges_u_name = "GatherV2:0"
edges_v_name = "GatherV2_1:0"
ratio_pos_name = "PyFuncStateless_1:1"
ratio_neg_name = "PyFuncStateless_1:2"
dist_name = "Sqrt:0"
num_pos_pairs_name = "PyFuncStateless_1:3"
num_neg_pairs_name = "PyFuncStateless_1:4"
summaries_name = "Merge/MergeSummary:0"

TENSOR_NAMES = {
    "emst": emst_name,
    "edges_u": edges_u_name,
    "edges_v": edges_v_name,
    "ratio_pos": ratio_pos_name,
    "ratio_neg": ratio_neg_name,
    "dist": dist_name,
    "num_pos_pairs": num_pos_pairs_name,
    "num_neg_pairs": num_neg_pairs_name,
    "summaries": summaries_name,
}


def create_custom_loss(tensor_names: Dict[str, str], config: Dict[str, Any]):
    def add_loss(graph):
        OUTPUT_SHAPE = config["OUTPUT_SHAPE"]
        MAX_FILTER_SIZE = config["MAX_FILTER_SIZE"]
        MAXIMA_THRESHOLD = config["MAXIMA_THRESHOLD"]
        COORDINATE_SCALE = config["COORDINATE_SCALE"]
        ALPHA = config["ALPHA"]
        CONSTRAINED = config["CONSTRAINED"]

        # k, h, w
        embedding = graph.get_tensor_by_name(tensor_names["embedding"])

        # h, w
        fg = graph.get_tensor_by_name(tensor_names["fg"])

        # h, w
        gt_labels = graph.get_tensor_by_name(tensor_names["gt_labels"])

        # h, w
        gt_fg = tf.not_equal(gt_labels, 0, name="gt_fg")

        # h, w

        _, maxima = max_detection(
            tf.reshape(fg, (1, *OUTPUT_SHAPE, 1)),
            window_size=(1, *MAX_FILTER_SIZE),
            threshold=MAXIMA_THRESHOLD,
        )

        # 1, k, h, w
        embedding = tf.reshape(embedding, (1,) + tuple(embedding.get_shape().as_list()))
        # k, 1, h, w
        embedding = tf.transpose(embedding, perm=[1, 0, 2, 3])

        um_loss, emst, edges_u, edges_v, _ = ultrametric_loss_op(
            embedding,
            gt_labels,
            mask=maxima,
            coordinate_scale=COORDINATE_SCALE,
            alpha=ALPHA,
            constrained_emst=CONSTRAINED,
        )

        ratio_pos = graph.get_tensor_by_name(ratio_pos_name)
        ratio_neg = graph.get_tensor_by_name(ratio_neg_name)
        dist = graph.get_tensor_by_name(dist_name)
        ratio_pos = tf.concat([ratio_pos, [0]], axis=0)
        ratio_neg = tf.concat([ratio_neg, [0]], axis=0)
        dist = tf.concat([dist, [0]], axis=0)

        # Reorder edges to process mst in the proper order
        # In the case of a constrained um_loss, edges won't be in ascending order
        order = tf.argsort(dist, axis=-1, direction="ASCENDING")
        ratio_pos = tf.gather(ratio_pos, order)
        ratio_neg = tf.gather(ratio_neg, order)
        dist = tf.gather(dist, order)

        # Calculate a score:

        # false positives are just ratio_neg
        # false negatives are 1 - ratio_pos

        false_pos = tf.math.cumsum(ratio_neg)
        false_neg = 1 - tf.math.cumsum(ratio_pos)
        scores = false_pos + false_neg
        best_score = tf.math.reduce_min(scores)
        best_score_index = tf.math.argmin(scores)
        best_alpha = tf.gather(dist, best_score_index)

        assert emst.name == emst_name, f"{emst.name} != {emst_name}"
        assert edges_u.name == edges_u_name, f"{edges_u.name} != {edges_u_name}"
        assert edges_v.name == edges_v_name, f"{edges_v.name} != {edges_v_name}"

        fg_loss = tf.losses.mean_squared_error(gt_fg, fg)

        loss = um_loss + fg_loss

        tf.summary.scalar("um_loss", um_loss)
        tf.summary.scalar("fg_loss", fg_loss)
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("best_alpha", best_alpha)
        tf.summary.scalar("best_score", best_score)

        summaries = tf.summary.merge_all()

        assert summaries.name == summaries_name, f"{summaries.name} != {summaries_name}"

        opt = tf.train.AdamOptimizer(
            learning_rate=0.5e-5, beta1=0.95, beta2=0.999, epsilon=1e-8
        )

        optimizer = opt.minimize(loss)

        return (loss, optimizer)

    return add_loss
