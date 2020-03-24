import logging

from gunpowder.batch_request import BatchRequest
from gunpowder.torch.nodes.train import Train
from gunpowder.array import ArrayKey, Array
from gunpowder.array_spec import ArraySpec
from gunpowder.coordinate import Coordinate
from gunpowder.ext import torch
import numpy as np

from typing import Dict, Union

logger = logging.getLogger(__name__)


class TrainForeground(Train):
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
        outputs: Dict[Union[int, str], ArrayKey],
        target: ArrayKey,
        weights: ArrayKey,
        input_size: Coordinate,
        output_size: Coordinate,
        gradients=None,
        array_specs: Dict[str, ArraySpec] = None,
        checkpoint_basename: str = "model",
        save_every: int = 2000,
        log_dir: str = None,
        log_every: int = 1,
    ):

        self.input_size = input_size
        self.output_size = output_size

        self.weights = weights

        super(TrainForeground, self).__init__(
            model,
            loss,
            optimizer,
            inputs,
            outputs,
            target,
            gradients,
            array_specs,
            checkpoint_basename,
            save_every,
            log_dir,
            log_every,
        )
        self.outputs = outputs

        self.intermediate_layers = {}
        self.register_hooks()

    def prepare(self, request):
        """
        TODO: There is no prepare method for the train nodes.
        This is a pain because it means that whatever is in the pipeline
        when it passes this node will be used as the inputs/targets etc.

        If you request ground truth labels of size "input_size", your loss
        function will probably throw an error due to it comparing the output
        of your network with size "output_size" to your labels which have size
        "input_size".
        """
        deps = BatchRequest()
        # Get the roi for the outputs
        output_requests = BatchRequest()
        for array_key in self.outputs.values():
            if array_key in request:
                output_requests[array_key] = request[array_key].copy()
        output_total_roi = output_requests.get_total_roi()
        diff = self.output_size - output_total_roi.get_shape()
        assert Coordinate([x % 2 for x in diff]) == Coordinate(
            [0] * len(self.output_size)
        )

        output_roi = output_total_roi.grow(diff // 2, diff // 2)
        assert output_roi.get_shape() == self.output_size

        # Grow the output roi to fit the appropriate input roi
        diff = self.input_size - output_roi.get_shape()
        assert Coordinate([x % 2 for x in diff]) == Coordinate(
            [0] * len(self.output_size)
        )
        input_roi = output_roi.grow(diff // 2, diff // 2)

        # Request inputs:
        for array_key in self.inputs.values():
            deps[array_key] = ArraySpec(roi=input_roi)

        # Request targets:
        for array_key in self.targets.values():
            deps[array_key] = ArraySpec(roi=output_roi)
        deps[self.weights] = ArraySpec(roi=output_roi)

        return deps

    def register_hooks(self):
        for key in self.outputs:
            if isinstance(key, str):
                layer = getattr(self.model, key)
                layer.register_forward_hook(self.create_hook(key))

    def create_hook(self, key):
        def save_layer(module, input, output):
            self.intermediate_layers[key] = output

        return save_layer

    def train_step(self, batch, request):
        self.intermediate_layers = {}

        inputs = self._Train__collect_provided_inputs(batch)
        targets = self._Train__collect_provided_targets(batch)
        weights = batch[self.weights].data
        if weights.dtype == np.uint32:
            weights = weights.astype(np.int32)
        weights = torch.as_tensor(weights, device=self.device)
        requested_outputs = self._Train__collect_requested_outputs(request)

        device_inputs = {
            k: torch.as_tensor(v, device=self.device).unsqueeze(0)
            for k, v in inputs.items()
        }

        device_targets = {
            k: torch.as_tensor(v, device=self.device).unsqueeze(0)
            for k, v in targets.items()
        }
        weights = weights.unsqueeze(0)

        self.optimizer.zero_grad()
        model_outputs = self.model(**device_inputs)
        if isinstance(model_outputs, tuple):
            outputs = {i: model_outputs[i] for i in range(len(model_outputs))}
        elif isinstance(model_outputs, torch.Tensor):
            outputs = {0: model_outputs}
        else:
            raise RuntimeError(
                "Torch train node only supports return types of tuple",
                f"and torch.Tensor from model.forward(). not {type(model_outputs)}",
            )

        for array_name, array_key in self.gradients.items():
            if array_key not in request:
                continue
            if isinstance(array_name, int):
                tensor = outputs[array_name]
            elif isinstance(array_name, str):
                tensor = getattr(self.model, array_name)
            else:
                raise RuntimeError(
                    "only ints and strings are supported as gradients keys"
                )
            tensor.retain_grad()

        logger.debug("model output: %s", outputs[0])
        logger.debug("expected output: %s", device_targets["output"])
        loss = self.loss(
            input=outputs[list(self.outputs.keys())[0]], target=device_targets["output"], weights=weights
        )
        loss.backward()
        self.optimizer.step()

        for array_name, array_key in self.gradients.items():
            if array_key not in request:
                continue
            if isinstance(array_name, int):
                tensor = outputs[array_name]
            elif isinstance(array_name, str):
                tensor = getattr(self.model, array_name)
            else:
                raise RuntimeError(
                    "only ints and strings are supported as gradients keys"
                )

            spec = self.spec[array_key].copy()
            spec.roi = request[array_key].roi

            gradient = tensor.grad
            if gradient is None:
                batch.arrays[array_key] = Array(np.zeros(tensor.shape, dtype=spec.dtype), spec)
            else:
                batch.arrays[array_key] = Array(
                    torch.squeeze(tensor.grad, 0).cpu().numpy(), spec
                )

        for array_key, array_name in requested_outputs.items():
            if isinstance(array_name, int):
                tensor = outputs[array_name]
            elif isinstance(array_name, str):
                tensor = getattr(self.model, array_name)
            else:
                raise RuntimeError(
                    "only ints and strings are supported as gradients keys"
                )
            spec = self.spec[array_key].copy()
            spec.roi = request[array_key].roi
            batch.arrays[array_key] = Array(
                torch.squeeze(tensor, 0).cpu().detach().numpy(), spec
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
            self.summary_writer.add_scalar("mean", outputs[0].mean(), batch.iteration)
            self.summary_writer.add_scalar("std", outputs[0].std(), batch.iteration)
