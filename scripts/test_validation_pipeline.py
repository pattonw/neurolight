from neurolight.pipelines.validation_pipeline import validation_pipeline
from neurolight.pipelines import DEFAULT_CONFIG

import gunpowder as gp

if __name__ == "__main__":

    DEFAULT_CONFIG["INPUT_SHAPE"] = tuple(
        a + b for a, b in zip(DEFAULT_CONFIG["INPUT_SHAPE"], [12 * x for x in [4, 12, 12]])
    )
    DEFAULT_CONFIG["OUTPUT_SHAPE"] = tuple(
        a + b for a, b in zip(DEFAULT_CONFIG["OUTPUT_SHAPE"], [12 * x for x in [4, 12, 12]])
    )
    pipeline, score_key = validation_pipeline(DEFAULT_CONFIG)
    request = gp.BatchRequest()
    request[score_key] = gp.ArraySpec(nonspatial=True)
    with gp.build(pipeline):
        batch = pipeline.request_batch(request)
        print(batch[score_key])
        print(batch[score_key].data)
