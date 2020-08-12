import click

import logging

from omegaconf import OmegaConf


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

    import json
    import logging

    logging.basicConfig(level=logging.INFO, filename="training.log")

    setup_config = DEFAULT_CONFIG
    setup_config.update(OmegaConf.load(open(secrets)))
    setup_config.update(OmegaConf.load(open(f"{setup}/config.json")))

    setup_config.training.num_iterations = num_iterations

    pipeline, requests = foreground_pipeline(setup_config)
    request = gp.BatchRequest()

    click.echo(f"Training foreground {num_iterations} iterations")
    for key, shape in requests:
        request.add(key, shape)
    with gp.build(pipeline):
        for i in range(setup_config["NUM_ITERATIONS"]):
            pipeline.request_batch(request)


@neurolight.command()
def grid_search_fg():
    print("grid_search_fg")


@neurolight.command()
def train_emb():
    print("train_emb")


@neurolight.command()
def grid_search_emb():
    print("grid_search_emb")
