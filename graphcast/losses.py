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
import numpy as np
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
  """Latitude- and pressure-level-weighted MSE loss."""
  def loss(prediction, target):
    prediction = prediction.fillna(0)
    target = target.fillna(0)
    loss = (prediction - target)**2
    return _mean_preserving_batch(loss)

  losses = xarray_tree.map_structure(loss, predictions, targets)
  return sum_per_variable_losses(losses, per_variable_weights)


def _mean_preserving_batch(x: xarray.DataArray) -> xarray.DataArray:
  return x.mean([d for d in x.dims if d != 'batch'], skipna=False)


def sum_per_variable_losses(
    per_variable_losses: Mapping[str, xarray.DataArray],
    weights: Mapping[str, float],
) -> LossAndDiagnostics:
  """Weighted sum of per-variable losses."""
  if not set(weights.keys()).issubset(set(per_variable_losses.keys())):
    raise ValueError(
        'Passing a weight that does not correspond to any variable '
        f'{set(weights.keys())-set(per_variable_losses.keys())}')

  weighted_per_variable_losses = {
      name: loss * weights.get(name, 1)
      for name, loss in per_variable_losses.items()
  }
  total = xarray.concat(
      weighted_per_variable_losses.values(), dim='variable', join='exact').sum(
          'variable', skipna=False)
  return total, per_variable_losses  # pytype: disable=bad-return-type


def _check_uniform_spacing_and_get_delta(vector):
  diff = np.diff(vector)
  if not np.all(np.isclose(diff[0], diff)):
    raise ValueError(f'Vector {diff} is not uniformly spaced.')
  return diff[0]
