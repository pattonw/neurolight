from neurolight.transforms.swc_to_graph import swc_to_pickle, parse_consensus
import numpy as np
from tqdm import tqdm

from pathlib import Path

samples = (
    # "2017-09-25",
    # "2017-11-17",
    # "2018-03-09",
    # "2018-04-25",
    # "2018-05-23",
    # "2018-06-14",
    "2018-07-02",
    "2018-08-01",
    "2018-10-01",
    "2018-12-01",
)


tracing_source = Path("/groups/mousebrainmicro/mousebrainmicro/tracing_complete")
axon_source_template = "{node_id}/consensus.swc"
dendrite_source_template = "{node_id}/dendrite.swc"
transform_source_template = "/nrs/mouselight/SAMPLES/{sample}/transform.txt"


def load_transform(transform_path: Path):
    text = transform_path.open("r").read()
    lines = text.split("\n")
    constants = {}
    for line in lines:
        if len(line) > 0:
            variable, value = line.split(":")
            constants[variable] = float(value)
    spacing = (
        np.array([constants["sx"], constants["sy"], constants["sz"]])
        / 2 ** (constants["nl"] - 1)
        / 1000
    )
    origin = spacing * (
        (np.array([constants["ox"], constants["oy"], constants["oz"]]) // spacing)
        / 1000
    )
    return origin, spacing

def sample_cable_length(consensus_dir, transform):

    cable_length = 0

    for consensus_neuron in tqdm(consensus_dir.iterdir(), "Consensus neurons: "):
        if consensus_neuron.name.startswith("G"):
            pass
        else:
            continue

        consensus_graph = parse_consensus(
            consensus_neuron / "consensus.swc",
            consensus_neuron / "dendrite.swc",
            transform,
            offset=np.array([0, 0, 0]),
            resolution=np.array([300, 300, 1000]),
            transpose=[2, 1, 0],
        )

        for u, v in consensus_graph.edges:
            cable_length += np.linalg.norm(
                consensus_graph.nodes[u]["location"]
                - consensus_graph.nodes[v]["location"]
            )
    return cable_length


for sample in tqdm(samples, "Samples: "):
    sample_tracings = tracing_source / sample
    assert sample_tracings.is_dir(), f"{sample_tracings} is not a directory!"

    transform = Path(transform_source_template.format(sample=sample))
    assert transform.exists(), f"Sample {sample} has no transform.txt at {transform}!"

    # cable_length = sample_cable_length(sample_tracings, transform)
    offset, spacing = load_transform(transform)
    print(f"sample {sample} has offset {offset} and spacing {spacing}")

    #print(f"sample {sample} has cable length {cable_length / 1000} microns")