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

from omegaconf.dictconfig import DictConfig

import logging


def foreground_pipeline(setup_config: DictConfig, get_data_sources=None):
    assert isinstance(setup_config, DictConfig), "Not using an OmegaConf for configs!"
    input_shape = gp.Coordinate(setup_config.data.input_shape)
    output_shape = gp.Coordinate(setup_config.data.output_shape)
    voxel_size = gp.Coordinate(setup_config.data.voxel_size)

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
    if setup_config.clahe.enabled:
        pipeline = pipeline + scipyCLAHE(
            [raw],
            kernel_size=gp.Coordinate(setup_config.clahe.kernel_size) * voxel_size,
            clip_limit=setup_config.clahe.clip_limit,
        )

    if setup_config.precache.num_workers > 1:
        pipeline = add_caching(pipeline, setup_config)

    if setup_config.pipeline.train_foreground:

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

        requests = (
            (raw, input_size),
            (fg_pred, output_size),
            (gt_fg, output_size),
            (loss_weights, output_size),
            (fg_pred_gradient, output_size),
        )

    else:
        pipeline, fg_pred = add_foreground_prediction(pipeline, setup_config, raw)

        snapshot_datasets += [
            # output data
            (fg_pred, output_size, "volumes/fg_pred")
        ]

        requests = ((raw, input_size), (fg_pred, output_size))

    if setup_config.training.profile_every > 0:
        pipeline += gp.PrintProfilingStats(
            every=int(setup_config.training.profile_every)
        )

    if setup_config.snapshot.every > 0:

        pipeline = add_snapshot(pipeline, setup_config, snapshot_datasets)

    return (pipeline, requests)


def embedding_pipeline(setup_config: DictConfig, get_data_sources=None):
    assert isinstance(setup_config, DictConfig), "Not using an OmegaConf for configs!"
    output_shape = gp.Coordinate(setup_config.data.output_shape)
    input_shape = gp.Coordinate(setup_config.data.input_shape)
    voxel_size = gp.Coordinate(setup_config.data.voxel_size)
    output_size = output_shape * voxel_size
    input_size = input_shape * voxel_size

    pipeline, snapshot_datasets, raw, labels, matched = get_training_inputs(
        setup_config, get_data_sources
    )

    if (
        setup_config.emb_model.aux_task.enabled
        and setup_config.pipeline.train_embedding
    ):
        pipeline, neighborhood_gt, neighborhood_mask = add_neighborhood(
            pipeline, setup_config, matched
        )

    pipeline = grow_labels(pipeline, setup_config, labels)

    pipeline = add_data_augmentation(pipeline, raw)

    if setup_config.precache.num_workers > 1:
        pipeline = add_caching(pipeline, setup_config)

    pipeline, fg_pred = add_foreground_prediction(pipeline, setup_config, raw)

    pipeline, maxima = get_candidates(pipeline, setup_config, fg_pred)

    snapshot_datasets += [
        (fg_pred, output_size, "volumes/fg_pred", logging.INFO),
        (maxima, output_size, "volumes/fg_maxima", logging.INFO),
    ]

    if setup_config.pipeline.train_embedding:

        if setup_config.emb_model.aux_task.enabled:
            pipeline, embedding, embedding_gradient, neighborhood, neighborhood_gradient = add_embedding_training(
                pipeline,
                setup_config,
                raw,
                labels,
                maxima,
                neighborhood_gt,
                neighborhood_mask,
            )

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

            requests = (
                (raw, input_size),
                (embedding, output_size),
                (labels, output_size),
                (maxima, output_size),
                (neighborhood_mask, output_size),
                (neighborhood_gt, output_size),
            )
        else:
            pipeline, embedding, embedding_gradient = add_embedding_training(
                pipeline, setup_config, raw, labels, maxima
            )

            snapshot_datasets += [
                # output data
                (embedding, output_size, "volumes/embedding")
            ]

            snapshot_datasets += [
                # gradient debugging
                (embedding_gradient, output_size, "volumes/embedding_gradient")
            ]

            requests = (
                (raw, input_size),
                (labels, output_size),
                (maxima, output_size),
                (embedding, output_size),
            )

    else:
        if setup_config.emb_model.aux_task.enabled:
            pipeline, embedding, neighborhood = add_embedding_prediction(
                pipeline, setup_config, raw
            )

            snapshot_datasets += [
                # output data
                (embedding, output_size, "volumes/embedding"),
                (neighborhood, output_size, "volumes/neighborhood"),
            ]

            requests = (
                (raw, input_size),
                (embedding, output_size),
                (neighborhood, output_size),
            )

        else:
            pipeline, embedding = add_embedding_prediction(pipeline, setup_config, raw)

            snapshot_datasets += [
                # output data
                (embedding, output_size, "volumes/embedding")
            ]

            requests = ((raw, input_size), (embedding, output_size))

    if setup_config.snapshot.every > 0:
        pipeline = add_snapshot(pipeline, setup_config, snapshot_datasets)

    if setup_config.training.profile_every > 0:
        pipeline += gp.PrintProfilingStats(
            every=int(setup_config.training.profile_every)
        )

    return (pipeline, requests)

