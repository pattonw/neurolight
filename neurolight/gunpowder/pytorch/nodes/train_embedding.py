from gunpowder import Array

from gunpowder.torch.nodes.train import Train
from gunpowder.ext import torch
from gunpowder.profiling import Timing

import numpy as np

import logging
from pathlib import Path


logger = logging.getLogger(__name__)


class TrainEmbedding(Train):
    """An extension of the torch Train node that creates a foreground unet specific
    training step. This is necessary to log application specific values to
    tensorboard since the generic train node simply logs loss, and I don't see
    an easy way of extending the logging functionality.
    """

    def __init__(
        self,
        *args,
        synchronize_timing=False,
        cudnn_enabled=True,
        cudnn_deterministic=False,
        cudnn_benchmark=True,
        **kwargs,
    ):

        self.synchronize_timing = synchronize_timing

        super(TrainEmbedding, self).__init__(*args, **kwargs)

        torch.backends.cudnn.enabled = cudnn_enabled
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = cudnn_benchmark

    def write_csv(self, data):
        log_file = Path("data_log.csv")
        if not log_file.exists():
            log_file.touch()
        with log_file.open("a+") as f:
            f.write(", ".join([f"{x}" for x in data]) + "\n")

    def train_step(self, batch, request):
        timing_setup = Timing(self, "setup")
        timing_setup.start()

        inputs = self._Train__collect_provided_inputs(batch)
        requested_outputs = self._Train__collect_requested_outputs(request)

        # keys are argument names of model forward pass
        device_inputs = {
            k: torch.as_tensor(v, device=self.device) for k, v in inputs.items()
        }

        logger.debug(f"Model inputs: {device_inputs}")

        # get outputs. Keys are tuple indices or model attr names as in self.outputs
        self.optimizer.zero_grad()

        timing_setup.stop()

        timing_forward = Timing(self, "forward")
        timing_forward.start()

        model_outputs = self.model(**device_inputs)

        torch.cuda.synchronize()
        timing_forward.stop()

        if isinstance(model_outputs, tuple):
            outputs = {i: model_outputs[i] for i in range(len(model_outputs))}
        elif isinstance(model_outputs, torch.Tensor):
            outputs = {0: model_outputs}
        else:
            raise RuntimeError(
                "Torch train node only supports return types of tuple",
                f"and torch.Tensor from model.forward(). not {type(model_outputs)}",
            )
        outputs.update(self.intermediate_layers)

        # Some inputs to the loss should come from the batch, not the model
        provided_loss_inputs = self._Train__collect_provided_loss_inputs(batch)

        device_loss_inputs = {
            k: torch.as_tensor(v, device=self.device)
            for k, v in provided_loss_inputs.items()
        }

        # Some inputs to the loss function should come from the outputs of the model
        # Update device loss inputs with tensors from outputs if available
        flipped_outputs = {v: outputs[k] for k, v in self.outputs.items()}
        device_loss_inputs = {
            k: flipped_outputs.get(v, device_loss_inputs.get(k))
            for k, v in self.loss_inputs.items()
        }

        device_loss_args = []
        for i in range(len(device_loss_inputs)):
            if i in device_loss_inputs:
                device_loss_args.append(device_loss_inputs.pop(i))
            else:
                break
        device_loss_kwargs = {}
        for k, v in list(device_loss_inputs.items()):
            if isinstance(k, str):
                device_loss_kwargs[k] = device_loss_inputs.pop(k)
        assert (
            len(device_loss_inputs) == 0
        ), f"Not all loss inputs could be interpreted. Failed keys: {device_loss_inputs.keys()}"

        self.retain_gradients(request, outputs)

        logger.debug("model outputs: %s", outputs)
        logger.debug(f"loss_inputs: {device_loss_args}, {device_loss_kwargs}")

        timing_loss = Timing(self, "loss")
        timing_loss.start()

        if self.loss.aux_task:
            loss, emst, edges_u, edges_v, dist, ratio_pos, ratio_neg, neighborhood_loss = self.loss(
                *device_loss_args, **device_loss_kwargs
            )

        else:
            loss, emst, edges_u, edges_v, dist, ratio_pos, ratio_neg = self.loss(
                *device_loss_args, **device_loss_kwargs
            )

        if self.synchronize_timing:
            torch.cuda.synchronize()
        timing_loss.stop()

        timing_backward = Timing(self, "backward")
        timing_backward.start()

        loss.backward()

        if self.synchronize_timing:
            torch.cuda.synchronize()
        timing_backward.stop()

        timing_optimizer_step = Timing(self, "optimizer")
        timing_optimizer_step.start()

        self.optimizer.step()

        if self.synchronize_timing:
            torch.cuda.synchronize()
        timing_optimizer_step.stop()

        timing_requested_outputs = Timing(self, "requested_outs")
        timing_requested_outputs.start()

        # add requested model outputs to batch
        for array_key, array_name in requested_outputs.items():
            spec = self.spec[array_key].copy()
            spec.roi = request[array_key].roi
            batch.arrays[array_key] = Array(
                outputs[array_name].cpu().detach().numpy(), spec
            )

        if self.synchronize_timing:
            torch.cuda.synchronize()
        timing_requested_outputs.stop()

        timing_gradients = Timing(self, "gradients")
        timing_gradients.start()

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
            batch.arrays[array_key] = Array(tensor.grad.cpu().detach().numpy(), spec)

        if self.synchronize_timing:
            torch.cuda.synchronize()
        timing_gradients.stop()

        timing_checkpoint = Timing(self, "gradients")
        timing_checkpoint.start()

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

        if self.synchronize_timing:
            torch.cuda.synchronize()
        timing_checkpoint.stop()

        timing_logging = Timing(self, "logging")
        timing_logging.start()

        # calculate stats:
        dist = dist.detach().numpy()
        ratio_pos = ratio_pos.numpy()
        ratio_neg = ratio_neg.numpy()

        # make sure ratio_pos, ratio_neg and dist are not empty
        ratio_pos = np.concatenate([ratio_pos, np.array([0])], axis=0)
        ratio_neg = np.concatenate([ratio_neg, np.array([0])], axis=0)
        dist = np.concatenate([dist, np.array([0])], axis=0)

        # get pos and neg aspects of loss:
        pos_loss = sum(ratio_pos * dist)
        neg_loss = batch.loss - pos_loss

        # Reorder edges to process mst in the proper order
        # In the case of a constrained um_loss, edges won't be in ascending order
        order = np.argsort(dist, axis=-1)
        ratio_pos = np.take(ratio_pos, order)
        ratio_neg = np.take(ratio_neg, order)
        dist = np.take(dist, order)

        # Calculate a score:
        false_pos = np.cumsum(ratio_neg)
        false_neg = 1 - np.cumsum(ratio_pos)
        scores = false_pos + false_neg
        best_score = scores.min()
        best_score_index = np.argmin(scores)
        # best score is a range of values
        best_alpha_min = np.take(dist, best_score_index)
        try:
            best_alpha_max = np.take(dist, best_score_index + 1)
        except IndexError:
            best_alpha_max = np.nan

        if self.summary_writer and batch.iteration % self.log_every == 0:

            self.summary_writer.add_scalar("loss", batch.loss, batch.iteration)
            self.summary_writer.add_scalar("pos_loss", pos_loss, batch.iteration)
            self.summary_writer.add_scalar("neg_loss", neg_loss, batch.iteration)
            # The alpha to use for this iteration that would have provided the best score
            self.summary_writer.add_scalar(
                "optimal_threshold_min", best_alpha_min, batch.iteration
            )
            # The next highest alpha. There should be a large gap between each process
            self.summary_writer.add_scalar(
                "optimal_threshold_max", best_alpha_max, batch.iteration
            )
            # The size of optimal alpha range.
            self.summary_writer.add_scalar(
                "optimal_threshold_range",
                best_alpha_max - best_alpha_min,
                batch.iteration,
            )
            # The score given a perfectly chosen alpha
            self.summary_writer.add_scalar("best_score", best_score, batch.iteration)
            # The average intra object distance
            self.summary_writer.add_scalar(
                "intra_object_mean_dist", np.mean(ratio_pos * dist), batch.iteration
            )
            # The average inter object distance
            self.summary_writer.add_scalar(
                "inter_object_mean_dist", np.mean(ratio_neg * dist), batch.iteration
            )
            # The number of non-background objects
            self.summary_writer.add_scalar(
                "num_obj",
                len(device_loss_kwargs["target"].unique()) - 1,
                batch.iteration,
            )

            if self.loss.aux_task:
                self.summary_writer.add_scalar(
                    "neighborhood_loss", neighborhood_loss.detach().numpy(), batch.iteration
                )
                print(f"neighborhood_loss: {neighborhood_loss.detach().numpy()}")

            self.write_csv(
                [
                    batch.iteration,
                    batch.loss,
                    pos_loss,
                    neg_loss,
                    best_alpha_min,
                    best_alpha_min,
                    best_alpha_max - best_alpha_min,
                    best_score,
                    np.mean(ratio_pos * dist),
                    np.mean(ratio_neg * dist),
                ]
            )

        if self.synchronize_timing:
            torch.cuda.synchronize()
        timing_logging.stop()

        batch.profiling_stats.add(timing_setup)
        batch.profiling_stats.add(timing_requested_outputs)
        batch.profiling_stats.add(timing_gradients)
        batch.profiling_stats.add(timing_checkpoint)

        # batch.profiling_stats.add(timing_logging)
        batch.profiling_stats.add(timing_forward)
        batch.profiling_stats.add(timing_loss)
        batch.profiling_stats.add(timing_backward)
        batch.profiling_stats.add(timing_optimizer_step)
