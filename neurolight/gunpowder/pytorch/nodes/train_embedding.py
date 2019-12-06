import logging

from gunpowder.torch.nodes.train import Train
from gunpowder.array import ArrayKey, Array, ArraySpec
from gunpowder.ext import torch

from typing import Dict

logger = logging.getLogger(__name__)


class EmbeddingTrainer(Train):
    """An extension of the torch Train node that creates a foreground unet specific
    training step. This is necessary to log application specific values to
    tensorboard since the generic train node simply logs loss, and I don't see
    an easy way of extending the logging functionality.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss,
        optimizer,
        inputs: Dict[str, ArrayKey],
        output: ArrayKey,
        target: ArrayKey,
        gradients=None,
        array_specs: Dict[str, ArraySpec] = None,
        checkpoint_basename: str = "model",
        save_every: int = 2000,
        log_dir: str = None,
        log_every: int = 1,
    ):

        super(EmbeddingTrainer, self).__init__(
            model,
            loss,
            optimizer,
            inputs,
            output,
            target,
            gradients,
            array_specs,
            checkpoint_basename,
            save_every,
            log_dir,
            log_every,
        )

    def train_step(self, batch, request):

        inputs = self._Train__collect_provided_inputs(batch)
        targets = self._Train__collect_provided_targets(batch)
        requested_outputs = self._Train__collect_requested_outputs(request)

        device_inputs = {
            k: torch.as_tensor(v, device=self.device) for k, v in inputs.items()
        }

        device_targets = {
            k: torch.as_tensor(v, device=self.device) for k, v in targets.items()
        }

        self.optimizer.zero_grad()
        outputs = {"output": self.model(**device_inputs)}

        logger.debug("model output: %s", outputs["output"])
        logger.debug("expected output: %s", device_targets["output"])
        loss = self.loss(outputs["output"], device_targets["output"])
        loss.backward()
        self.optimizer.step()

        for array_key, array_name in requested_outputs.items():
            spec = self.spec[array_key].copy()
            spec.roi = request[array_key].roi
            batch.arrays[array_key] = Array(
                outputs[array_name].cpu().detach().numpy(), spec
            )

        batch.loss = loss.cpu().detach().numpy()
        self.iteration += 1
        batch.iteration = self.iteration

        if batch.iteration % self.save_every == 0:

            checkpoint_name = self._checkpoint_name(
                self.checkpoint_basename, batch.iteration
            )

            logger.info("Creating checkpoint %s", checkpoint_name)

            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                },
                checkpoint_name,
            )

        if self.summary_writer and batch.iteration % self.log_every == 0:
            self.summary_writer.add_scalar("loss", batch.loss, batch.iteration)
