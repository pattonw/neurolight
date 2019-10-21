from tqdm import tqdm

from neurolight.transforms.txt_to_graph import txt_to_pickle
from neurolight.transforms.swc_to_graph import swc_to_pickle

from pathlib import Path
import shutil


# data sources
tracing_source = Path("/groups/mousebrainmicro/mousebrainmicro/tracing_complete")
axon_source_template = "{node_id}/consensus.swc"
dendrite_source_template = "{node_id}/dendrite.swc"
skeletonization_source_template = "/groups/mousebrainmicro/mousebrainmicro/cluster/Reconstructions/{sample}/skeleton-graph.txt"
transform_source_template = "/nrs/mouselight/SAMPLES/{sample}/transform.txt"

# data targets
target = Path("/nrs/funke/mouselight-v2")
consensus_target_template = (
    "/nrs/funke/mouselight-v2/{sample}/consensus_tracings/{node_id}.obj"
)

# samples of interest
samples = (
    # "2017-09-25",
    # "2017-11-17",
    # "2018-03-09",
    # "2018-04-25",
    # "2018-05-23",
    # "2018-06-14",
    "2018-07-02",
    "2018-08-01",
    # "2018-10-01",
)

for sample in tqdm(samples, "Samples: "):
    sample_tracings = tracing_source / sample
    assert sample_tracings.is_dir(), f"{sample_tracings} is not a directory!"

    transform = Path(transform_source_template.format(sample=sample))
    assert transform.exists(), f"Sample {sample} has no transform.txt at {transform}!"

    neurons = [
        sub_file.name for sub_file in sample_tracings.iterdir() if sub_file.is_dir()
    ]

    sample_target = target / sample
    if not sample_target.exists():
        sample_target.mkdir()

    # read consensus / dendrite swc's into objects
    for neuron in tqdm(neurons, "Neurons: "):
        consensus = sample_tracings / neuron / "consensus.swc"
        assert (
            consensus.exists()
        ), f"Neuron {neuron} in Sample {sample} has no consensus.swc at {consensus}!"
        dendrite = sample_tracings / neuron / "dendrite.swc"
        assert (
            dendrite.exists()
        ), f"Neuron {neuron} in Sample {sample} has no dendrite.swc at {dendrite}!"

        consensus_target_dir = sample_target / "consensus_tracings"
        if not consensus_target_dir.exists():
            consensus_target_dir.mkdir()

        consensus_target = consensus_target_dir / f"{neuron}.obj"
        if not consensus_target.exists():
            swc_to_pickle(consensus, dendrite, transform, consensus_target)

    # read skeletonization and move it
    skeletonization = Path(skeletonization_source_template.format(sample=sample))
    assert (
        skeletonization.exists()
    ), f"Sample {sample} has no skeleton-graph.txt at {skeletonization}"
    txt_to_pickle(skeletonization, transform, sample_target / "skeletonization.obj")
