import gunpowder as gp

from .pipeline_pieces import (
    get_training_inputs,
    get_mouselight_data_sources,
    get_snapshot_source,
    get_neuron_pair,
    add_label_processing,
    add_data_augmentation,
    add_caching,
    add_embedding_prediction,
    add_foreground_prediction,
    add_embedding_training,
    add_foreground_training,
    add_snapshot,
    get_candidates,
    guarantee_nonempty,
    grow_labels,
    add_neighborhood,
)

from neurolight.gunpowder.nodes.clahe import scipyCLAHE

import logging

def foreground_pipeline(setup_config, get_data_sources=None):
    input_shape = gp.Coordinate(setup_config["INPUT_SHAPE"])
    output_shape = gp.Coordinate(setup_config["OUTPUT_SHAPE"])
    voxel_size = gp.Coordinate(setup_config["VOXEL_SIZE"])

    input_size = input_shape * voxel_size
    output_size = output_shape * voxel_size

    pipeline, snapshot_datasets, raw, labels, matched = get_training_inputs(
        setup_config, get_data_sources
    )

    pipeline, gt_fg, loss_weights = add_label_processing(pipeline, setup_config, labels)

    snapshot_datasets += [
        # gt label data
        (gt_fg, output_size, "volumes/gt_fg"),
        (loss_weights, output_size, "volumes/loss_weights"),
    ]

    pipeline = add_data_augmentation(pipeline, raw)
    if setup_config["CLAHE"]:
        pipeline = pipeline + scipyCLAHE(
            [raw], kernel_size=gp.Coordinate([20, 64, 64]) * voxel_size, clip_limit=setup_config["CLIP_LIMIT"]
        )

    if setup_config["NUM_WORKERS"] > 1:
        pipeline = add_caching(pipeline, setup_config)

    if setup_config["TRAIN_FOREGROUND"]:

        pipeline, fg_pred, fg_pred_gradient = add_foreground_training(
            pipeline, setup_config, raw, gt_fg, loss_weights
        )

        snapshot_datasets += [
            # output data
            (fg_pred, output_size, "volumes/fg_pred")
        ]

        snapshot_datasets += [
            # gradient debugging
            (fg_pred_gradient, output_size, "volumes/fg_pred_gradient")
        ]

        inputs = [gt_fg, loss_weights, fg_pred_gradient]

        outputs = (raw, fg_pred)

    else:
        pipeline, fg_pred = add_foreground_prediction(pipeline, setup_config, raw)

        snapshot_datasets += [
            # output data
            (fg_pred, output_size, "volumes/fg_pred")
        ]
        inputs = []

        outputs = (raw, fg_pred)

    if setup_config["PROFILE_EVERY"] > 0:
        pipeline += gp.PrintProfilingStats(every=int(setup_config["PROFILE_EVERY"]))

    if setup_config["SNAPSHOT_EVERY"] > 0:

        pipeline = add_snapshot(pipeline, setup_config, snapshot_datasets)

    return (pipeline,) + outputs + (inputs,)


def embedding_pipeline(setup_config, get_data_sources=None):
    output_shape = gp.Coordinate(setup_config["OUTPUT_SHAPE"])
    voxel_size = gp.Coordinate(setup_config["VOXEL_SIZE"])
    output_size = output_shape * voxel_size

    pipeline, snapshot_datasets, raw, labels, matched = get_training_inputs(
        setup_config, get_data_sources
    )

    if setup_config["AUX_TASK"]:
        pipeline, neighborhood_gt, neighborhood_mask = add_neighborhood(
            pipeline, setup_config, matched
        )

    pipeline = grow_labels(pipeline, setup_config, labels)

    pipeline = add_data_augmentation(pipeline, raw)

    if setup_config["NUM_WORKERS"] > 1:
        pipeline = add_caching(pipeline, setup_config)

    pipeline, fg_pred = add_foreground_prediction(pipeline, setup_config, raw)

    pipeline, maxima = get_candidates(pipeline, setup_config, fg_pred)

    snapshot_datasets += [
        (fg_pred, output_size, "volumes/fg_pred", logging.INFO),
        (maxima, output_size, "volumes/fg_maxima", logging.INFO),
    ]

    if setup_config["TRAIN_EMBEDDING"]:

        if setup_config["AUX_TASK"]:
            pipeline, embedding, embedding_gradient, neighborhood, neighborhood_gradient = add_embedding_training(
                pipeline,
                setup_config,
                raw,
                labels,
                maxima,
                neighborhood_gt,
                neighborhood_mask,
            )
            inputs = (labels, maxima, neighborhood_mask, neighborhood_gt)

            snapshot_datasets += [
                # output data
                (embedding, output_size, "volumes/embedding"),
                (neighborhood, output_size, "volumes/neighborhood"),
                (neighborhood_gt, output_size, "volumes/neighborhood_gt"),
                (neighborhood_mask, output_size, "volumes/neighborhood_mask"),
            ]

            snapshot_datasets += [
                # gradient debugging
                (embedding_gradient, output_size, "volumes/embedding_gradient"),
                (neighborhood_gradient, output_size, "volumes/neighborhood_gradient"),
            ]

            outputs = (raw, embedding)
        else:
            pipeline, embedding, embedding_gradient = add_embedding_training(
                pipeline, setup_config, raw, labels, maxima
            )
            inputs = (labels, maxima)

            snapshot_datasets += [
                # output data
                (embedding, output_size, "volumes/embedding")
            ]

            snapshot_datasets += [
                # gradient debugging
                (embedding_gradient, output_size, "volumes/embedding_gradient")
            ]

            outputs = (raw, embedding)

    else:
        pipeline, embedding = add_embedding_prediction(pipeline, setup_config, raw)
        inputs = tuple()

        snapshot_datasets += [
            # output data
            (embedding, output_size, "volumes/embedding")
        ]

        outputs = (raw, embedding)

    if setup_config["SNAPSHOT_EVERY"] > 0:
        pipeline = add_snapshot(pipeline, setup_config, snapshot_datasets)

    if setup_config["PROFILE_EVERY"] > 0:
        pipeline += gp.PrintProfilingStats(every=int(setup_config["PROFILE_EVERY"]))

    return (pipeline,) + outputs + (inputs,)
