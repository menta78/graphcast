# Copyright 2023 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Loss functions (and terms for use in loss functions) used for weather."""

from typing import Mapping

from graphcast import xarray_tree
from jax import numpy as jnp
from typing_extensions import Protocol
import xarray


LossAndDiagnostics = tuple[xarray.DataArray, xarray.Dataset]


class LossFunction(Protocol):
  """A loss function.

  This is a protocol so it's fine to use a plain function which 'quacks like'
  this. This is just to document the interface.
  """

  def __call__(self,
               predictions: xarray.Dataset,
               targets: xarray.Dataset,
               **optional_kwargs) -> LossAndDiagnostics:
    """Computes a loss function.

    Args:
      predictions: Dataset of predictions.
      targets: Dataset of targets.
      **optional_kwargs: Implementations may support extra optional kwargs.

    Returns:
      loss: A DataArray with dimensions ('batch',) containing losses for each
        element of the batch. These will be averaged to give the final
        loss, locally and across replicas.
      diagnostics: Mapping of additional quantities to log by name alongside the
        loss. These will will typically correspond to terms in the loss. They
        should also have dimensions ('batch',) and will be averaged over the
        batch before logging.
    """


def weighted_mse(
    predictions: xarray.Dataset,
    targets: xarray.Dataset,
    per_variable_weights: Mapping[str, float],
) -> LossAndDiagnostics:

  loss = (predictions.elev - targets.elev)**2
  loss = loss.mean(["time", "node"])

 #def debug_callback(predictions, targets, loss):
 #    import pdb; pdb.set_trace()
 #import jax
 #jax.debug.callback(debug_callback, predictions, targets, loss)

  return loss, dict(elev=loss)
