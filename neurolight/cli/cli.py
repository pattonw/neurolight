import click
from omegaconf import OmegaConf
import contextlib

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def working_directory(path):
    """Changes working directory and returns to previous on exit."""
    prev_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)


@click.group()
def neurolight():
    pass


@neurolight.command()
@click.argument("setup", type=click.Path(exists=True, file_okay=False, writable=True))
@click.argument("secrets", type=click.Path(exists=True, dir_okay=False))
@click.option("-n", "--num-iterations", "num_iterations", type=int, default=1)
def train_fg(setup, secrets, num_iterations):
    import gunpowder as gp
    from neurolight.pipelines import foreground_pipeline, DEFAULT_CONFIG

    setup_config = DEFAULT_CONFIG
    setup_config = OmegaConf.merge(setup_config, OmegaConf.load(open(secrets)))
    setup_config = OmegaConf.merge(
        setup_config, OmegaConf.load(open(f"{setup}/config.yaml"))
    )

    setup_config.training.num_iterations = num_iterations

    # Change working directory to setup directory.
    # Pipe logs to "setup/training.log"
    with working_directory(setup):
        logging.basicConfig(level=logging.INFO, filename="training.log")

        pipeline, requests = foreground_pipeline(setup_config)
        request = gp.BatchRequest()

        click.echo(f"Training foreground {num_iterations} iterations")
        for key, shape in requests:
            request.add(key, shape)
        with gp.build(pipeline):
            for i in range(setup_config["NUM_ITERATIONS"]):
                pipeline.request_batch(request)


@neurolight.command()
@click.argument("setup", type=click.Path(exists=True, file_okay=False, writable=True))
@click.option("--checkpoint-range", type=(int, int), default=(-1, -1))
@click.option("--test", type=bool, default=False)
def grid_search_fg(setup, checkpoint_range, test):
    from neurolight.pipelines.validation_pipeline import fg_validation_pipeline
    from neurolight.pipelines import DEFAULT_CONFIG, DEFAULT_EVAL_CONFIG

    from gunpowder import build, BatchRequest, ArraySpec

    import json
    from pathlib import Path
    import sys
    import pickle
    import numpy as np
    import itertools

    with working_directory(setup):
        logging.basicConfig(level=logging.INFO, filename="grid_search.log")

        all_checkpoints = [
            int(x.name.split("_")[-1])
            for x in Path(__file__).resolve().parent.iterdir()
            if x.name.startswith("fg_net_checkpoint")
        ]
        latest_checkpoint = max(all_checkpoints)

        checkpoints = [int(x) for x in sys.argv[1:]]
        if len(checkpoints) == 0:
            checkpoints = [latest_checkpoint]
        if len(checkpoints) == 1 and checkpoints[0] == -1:
            checkpoints = all_checkpoints
        setup = str(Path(__file__).resolve().parent.name)

        snapshot_file = (
            "/groups/mousebrainmicro/home/pattonw/Code/Scripts"
            + "/neurolight_experiments/mouselight/07_evaluations/validations.zarr"
        )
        raw_path = "volumes/{block}/raw"
        raw_clahe_path = "volumes/{block}/raw_clahe"
        gt_path = "points/{block}/ground_truth"

        grid_search_parameters = [("eval.component_threshold_fg", [10000, 30000])]

        for checkpoint in checkpoints:
            grid_search_keys = [k for k, v in grid_search_parameters]
            for grid_search_values in itertools.product(
                *[v for k, v in grid_search_parameters]
            ):
                config = {}
                config.update(DEFAULT_CONFIG)
                config.update(DEFAULT_EVAL_CONFIG)
                config.update(json.load(open("config.json", "r")))

                config.fg_model.setup = setup
                config.fg_model.checkpoint = checkpoint

                config.data.input_shape = tuple(
                    a + b
                    for a, b in zip(
                        config.data.input_shape, [6 * x for x in [4, 12, 12]]
                    )
                )
                config.data.output_shape = tuple(
                    a + b
                    for a, b in zip(
                        config.data.output_shape, [6 * x for x in [4, 12, 12]]
                    )
                )

                config.eval.snapshot.every = int(
                    checkpoint == latest_checkpoint or len(checkpoints) == 1
                )
                prefix = "test_" if test else ""
                output_dir = (
                    f"eval_grid_search/{prefix}"
                    f"{'-'.join([k.split('.')[-1].lower() for k, _ in grid_search_parameters])}"
                )
                config.eval.snapshot.directory = f"{output_dir}/snapshots"
                if not Path(config.eval.snapshot.directory).exists():
                    Path(config.eval.snapshot.directory).mkdir(parents=True)

                grid_iter_name = f"{'-'.join([str(x) for x in grid_search_values])}"
                config.eval.snapshot.file_name = (
                    f"{{checkpoint}}_{{block}}_{grid_iter_name}.zarr"
                )

                # to add clahe to the raw data:
                config.eval.clahe.enabled = False
                if config.clahe.enabled:
                    raw_path = raw_clahe_path

                scores_file = f"{{checkpoint}}_{grid_iter_name}.obj"
                if Path(output_dir, scores_file.format(checkpoint=checkpoint)).exists():
                    logger.info(f"skipping checkpoint: {checkpoint}")
                    continue
                logger.info(f"evaluating checkpoint: {checkpoint}")

                config.eval.blocks = list(range(1, 26))
                if test:
                    config.eval.blocks = [8]
                # config["COORDINATE_SCALE"] = 1

                for k, v in zip(grid_search_keys, grid_search_values):
                    config[k] = v

                pipeline, score_key = fg_validation_pipeline(
                    config, snapshot_file, raw_path, gt_path
                )
                request = BatchRequest()
                request[score_key] = ArraySpec(nonspatial=True)
                with build(pipeline):
                    batch = pipeline.request_batch(request)
                    scores = batch[score_key].data

                    distances = scores[0, 0, :]
                    optimal_threshold = np.argmin(distances)
                    optimal_score = distances[optimal_threshold]
                    np.set_printoptions(precision=3)
                    np.set_printoptions(suppress=True)
                    if config.eval.snapshot.every:
                        logger.info(
                            f"Optimal score: {optimal_score} at {scores[0,1,optimal_threshold]}"
                        )
                        logger.info(scores)

                pickle.dump(
                    scores,
                    Path(output_dir, scores_file.format(checkpoint=checkpoint)).open(
                        "wb"
                    ),
                )


@neurolight.command()
@click.argument("setup", type=click.Path(exists=True, file_okay=False, writable=True))
@click.argument("secrets", type=click.Path(exists=True, dir_okay=False))
@click.option("-n", "--num-iterations", "num_iterations", type=int, default=1)
def train_emb(setup, secrets, num_iterations):
    import gunpowder as gp

    from neurolight.pipelines import embedding_pipeline, DEFAULT_CONFIG

    setup_config = DEFAULT_CONFIG
    setup_config = OmegaConf.merge(setup_config, OmegaConf.load(open(secrets)))
    setup_config = OmegaConf.merge(
        setup_config, OmegaConf.load(open(f"{setup}/config.yaml"))
    )

    setup_config.training.num_iterations = num_iterations

    # Change working directory to setup directory.
    # Pipe logs to "setup/training.log"
    with working_directory(setup):
        logging.basicConfig(level=logging.INFO, filename="training.log")

        pipeline, requests = embedding_pipeline(setup_config)
        request = gp.BatchRequest()
        for key, shape in requests:
            request.add(key, shape)
        with gp.build(pipeline):
            for i in range(setup_config["NUM_ITERATIONS"]):
                pipeline.request_batch(request)


@neurolight.command()
@click.argument("setup", type=click.Path(exists=True, file_okay=False, writable=True))
@click.option("--checkpoint-range", type=(int, int), default=(-1, -1))
@click.option("--test", type=bool, default=False)
def grid_search_emb(setup, checkpoint_range, test):
    from neurolight.pipelines.validation_pipeline import emb_validation_pipeline
    from neurolight.pipelines import DEFAULT_CONFIG

    from gunpowder import build, BatchRequest, ArraySpec

    from pathlib import Path
    import pickle
    import itertools
    import copy

    import numpy as np

    with working_directory(setup):
        logging.basicConfig(level=logging.INFO, filename="grid_search.log")

        all_checkpoints = sorted(
            [
                int(x.name.split("_")[-1])
                for x in Path.cwd().iterdir()
                if x.name.startswith("emb_net_checkpoint")
            ]
        )
        latest_checkpoint = max(all_checkpoints)

        if checkpoint_range == (-1, -1):
            checkpoints = [latest_checkpoint]
        elif checkpoint_range == (0, -1):
            checkpoints = all_checkpoints
        else:
            checkpoints = [
                checkpoint
                for checkpoint in checkpoints
                if checkpoint_range[0] <= checkpoint <= checkpoint_range[1]
            ]

        snapshot_file = (
            "/groups/mousebrainmicro/home/pattonw/Code/Scripts"
            + "/neurolight_experiments/mouselight/07_evaluations/validations.zarr"
        )
        candidates_path = "predicted_volumes/candidates/unet/setup_02/301000/{block}"
        fg_pred_path = "predicted_volumes/fg/unet/setup_02/301000/{block}"
        raw_path = "volumes/{block}/raw"
        raw_clahe_path = "volumes/{block}/raw_clahe"
        gt_path = "points/{block}/ground_truth"
        mst_path = "predicted_volumes/mst/unet/setup_02/301000/{block}"
        dense_mst_path = "predicted_volumes/mst_dense/unet/setup_02/301000/{block}"
        grid_search_parameters = [
            ("candidates.spacing", [5000, 10000, 20000]),
            ("um_loss.coordinate_scale", [0.01, 0.05, 10]),
        ]
        # grid_search_parameters = [("eval.clahe_mode", ["blockwise", "global"])]

        for checkpoint in checkpoints:
            grid_search_keys = [k for k, v in grid_search_parameters]
            for grid_search_values in itertools.product(
                *[v for k, v in grid_search_parameters]
            ):

                config = copy.deepcopy(DEFAULT_CONFIG)
                config = OmegaConf.merge(config, OmegaConf.load(open(f"config.yaml")))

                config.emb_model.setup = Path.cwd().name
                config.emb_model.checkpoint = checkpoint

                config.data.input_shape = tuple(
                    a + b
                    for a, b in zip(
                        config.data.input_shape, [6 * x for x in [4, 12, 12]]
                    )
                )
                config.data.output_shape = tuple(
                    a + b
                    for a, b in zip(
                        config.data.output_shape, [6 * x for x in [4, 12, 12]]
                    )
                )

                config.eval.snapshot.every = int(
                    checkpoint == latest_checkpoint or len(checkpoints) == 1
                )
                prefix = "test_" if test else ""
                output_dir = (
                    f"eval_grid_search/{prefix}"
                    f"{'-'.join([k.split('.')[-1].lower() for k, _ in grid_search_parameters])}"
                )
                config.eval.snapshot.directory = f"{output_dir}/snapshots"
                if not Path(config.eval.snapshot.directory).exists():
                    Path(config.eval.snapshot.directory).mkdir(parents=True)

                grid_iter_name = f"{'-'.join([str(x) for x in grid_search_values])}"
                if test:
                    grid_iter_name += "_test"
                config.eval.snapshot.file_name = (
                    f"{{checkpoint}}_{{block}}_{grid_iter_name}.zarr"
                )

                # to add clahe to the raw data:
                config.eval.clahe.enabled = False
                if config.clahe.enabled:
                    raw_path = raw_clahe_path

                # if candidates_mst_path and candidates_mst_path_dense not supplied, you
                # can calculate the mst based on the embedding gradients. (This doesn't work well)
                # config["EVAL_MINIMAX_EMBEDDING_DIST"] = False

                scores_file = f"{{checkpoint}}_{grid_iter_name}.obj"

                if Path(output_dir, scores_file.format(checkpoint=checkpoint)).exists():
                    logger.info(f"skipping checkpoint: {checkpoint}")
                    if test:
                        pass
                    else:
                        continue

                logger.info(f"evaluating checkpoint: {checkpoint}")

                config.eval.blocks = list(range(1, 26))
                if test:
                    config.eval.blocks = [8]

                for k, v in zip(grid_search_keys, grid_search_values):
                    OmegaConf.update(config, k, v)

                pipeline, score_key = emb_validation_pipeline(
                    config,
                    snapshot_file,
                    raw_path,
                    gt_path,
                    # candidates_mst_path=mst_path,
                    # candidates_mst_dense_path=dense_mst_path,
                    # candidates_path=candidates_path,
                    fg_pred_path=fg_pred_path,
                )
                request = BatchRequest()
                request[score_key] = ArraySpec(nonspatial=True)
                with build(pipeline):
                    batch = pipeline.request_batch(request)
                    scores = batch[score_key].data

                    distances = scores[0, 0, :]
                    optimal_threshold = np.argmin(distances)
                    optimal_score = distances[optimal_threshold]
                    np.set_printoptions(precision=3)
                    np.set_printoptions(suppress=True)
                    if config.eval.snapshot.every:
                        logger.info(
                            f"Optimal score: {optimal_score} at {scores[0,1,optimal_threshold]}"
                        )
                        logger.info(scores)

                pickle.dump(
                    scores,
                    Path(output_dir, scores_file.format(checkpoint=checkpoint)).open(
                        "wb"
                    ),
                )

