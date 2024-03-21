import os, time
import pickle as pkl
import dataclasses
import datetime
import functools
import math
import re
import gc
from typing import Optional

import cartopy.crs as ccrs
from graphcast import autoregressive
from graphcast import casting
from graphcast import checkpoint
from graphcast import data_utils
from graphcast import graphcast
from graphcast import normalization
from graphcast import rollout
from graphcast import xarray_jax
from graphcast import xarray_tree
from IPython.display import HTML
import ipywidgets as widgets
import haiku as hk
import jax, optax
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import xarray

# should ensure that memory is deallocated when the buffers are released in the GPU
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


def parse_file_parts(file_name):
  return dict(part.split("-", 1) for part in file_name.split("_"))


# @title Authenticate with Google Cloud Storage



print("creating/loading model_config and task_config")

# very coarse model to test on my laptop
# PUT BACK LATENT SIZE TO 512, I PUT IT TO 256 FOR MEMORY CONSTRAINT
model_config = graphcast.ModelConfig(resolution=1.0, mesh_size=4, latent_size=256, gnn_msg_steps=16, hidden_layers=1, 
        radius_query_fraction_edge_length=0.6, mesh2grid_edge_normalization_factor=0.6180338738074472)

task_config = graphcast.TaskConfig(
                input_variables=['u10', 'v10', 'hs'], 
                target_variables=['hs'], 
              forcing_variables=('u10', 'v10'),
              pressure_levels=[10.], 
              input_duration='6h')

params = None # params must be initialized
# fl = open("params.pkl", "rb")
# params = pkl.load(fl)

state = {}



print("loading the training data")

datadir = "./data/graphcastWavesExample/"
os.system(f"mkdir {datadir}")

# the variables of this file have 4 or 5 dimensions: (batch, time, lat, lon) or (batch, time, level, lat, lon)
#exampleBatchFileName = "waves200912_1batch.nc" # works also with this
exampleBatchFileName = "waves200912_1batch_fillna.nc"
exampleBatchFilePath = os.path.join(datadir, exampleBatchFileName)
example_batch = xarray.open_dataset(exampleBatchFilePath)

diffStddevByLevelFileName = "diffs_stddev_by_level.nc"
diffStddevByLevelFilePath = os.path.join(datadir, diffStddevByLevelFileName)
diffs_stddev_by_level = xarray.open_dataset(diffStddevByLevelFilePath)

meanByLevelFileName = "mean_by_level.nc"
meanByLevelFilePath = os.path.join(datadir, meanByLevelFileName)
mean_by_level = xarray.open_dataset(meanByLevelFilePath)

stddevByLevelFileName = 'stddev_by_level.nc'
stddevByLevelFilePath = os.path.join(datadir, stddevByLevelFileName)
stddev_by_level = xarray.open_dataset(stddevByLevelFilePath)



print("extracting training and testing sets")

train_inputs, train_targets, train_forcings = data_utils.extract_inputs_targets_forcings(
    example_batch, target_lead_times=slice("6h", "60h"),
    **dataclasses.asdict(task_config))


eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
    example_batch, target_lead_times=slice("60h", "72h"),
    **dataclasses.asdict(task_config))




# @title Build jitted functions, and possibly initialize random weights

def construct_wrapped_graphcast(
    model_config: graphcast.ModelConfig,
    task_config: graphcast.TaskConfig):
  """Constructs and wraps the GraphCast Predictor."""
  # Deeper one-step predictor.
  predictor = graphcast.GraphCast(model_config, task_config)

  # Modify inputs/outputs to `graphcast.GraphCast` to handle conversion to
  # from/to float32 to/from BFloat16.
  predictor = casting.Bfloat16Cast(predictor)

  # Modify inputs/outputs to `casting.Bfloat16Cast` so the casting to/from
  # BFloat16 happens after applying normalization to the inputs/targets.
  predictor = normalization.InputsAndResiduals(
      predictor,
      diffs_stddev_by_level=diffs_stddev_by_level,
      mean_by_level=mean_by_level,
      stddev_by_level=stddev_by_level)

  # Wraps everything so the one-step model can produce trajectories.
  predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True)
  return predictor


@hk.transform_with_state
def run_forward(model_config, task_config, inputs, targets_template, forcings):
  predictor = construct_wrapped_graphcast(model_config, task_config)
  return predictor(inputs, targets_template=targets_template, forcings=forcings)


@hk.transform_with_state
def loss_fn(model_config, task_config, inputs, targets, forcings):
  predictor = construct_wrapped_graphcast(model_config, task_config)
  loss, diagnostics = predictor.loss(inputs, targets, forcings)
  return xarray_tree.map_structure(
      lambda x: xarray_jax.unwrap_data(x.mean(), require_jax=True),
      (loss, diagnostics))

#def grads_fn(params, state, model_config, task_config, inputs, targets, forcings):
#  def _aux(params, state, i, t, f):
#    (loss, diagnostics), next_state = loss_fn.apply(
#        params, state, jax.random.PRNGKey(0), model_config, task_config,
#        i, t, f)
#    return loss, (diagnostics, next_state)
#  (loss, (diagnostics, next_state)), grads = jax.value_and_grad(
#      _aux, has_aux=True)(params, state, inputs, targets, forcings)
#  return loss, diagnostics, next_state, grads

def grads_fn(params, state, inputs, targets, forcings, model_config, task_config):
    def _aux(params, state, i, t, f):
        (loss, diagnostics), next_state = loss_fn.apply(
                params, state, jax.random.PRNGKey(0), model_config, task_config, 
                i, t, f)
        return loss, (diagnostics, next_state)
    (loss, (diagnostics, next_state)), grads = jax.value_and_grad(
            _aux, has_aux=True)(params, state, inputs, targets, forcings)
    return loss, diagnostics, next_state, grads

# Jax doesn't seem to like passing configs as args through the jit. Passing it
# in via partial (instead of capture by closure) forces jax to invalidate the
# jit cache if you change configs.
def with_configs(fn):
  return functools.partial(
      fn, model_config=model_config, task_config=task_config)

# Always pass params and state, so the usage below are simpler
def with_params(fn):
  return functools.partial(fn, params=params, state=state)

# Our models aren't stateful, so the state is always empty, so just return the
# predictions. This is requiredy by our rollout code, and generally simpler.
def drop_state(fn):
  return lambda **kw: fn(**kw)[0]

init_jitted = jax.jit(with_configs(run_forward.init))
if params is None:
  params, state = init_jitted(
      rng=jax.random.PRNGKey(0),
      inputs=train_inputs,
      targets_template=train_targets,
      forcings=train_forcings)




print("training")

print("  getting the functions for computing the loss, the backpropagation, the forward step")
loss_fn_jitted = drop_state(with_params(jax.jit(with_configs(loss_fn.apply))))
grads_fn_jitted = jax.jit(with_configs(grads_fn))
run_forward_jitted = drop_state(with_params(jax.jit(with_configs(
    run_forward.apply))))

print("  creating the optimizer")
lr = 1e-3
optimiser = optax.adam(lr, b1=0.9, b2=0.999, eps=1e-8)
opt_state = optimiser.init(params)

# training loop
nepochs = 40
#nepochs = 2
jitted = True
for iepoch in range(nepochs):
    print(f"epoch {iepoch}")
    ## @title Gradient computation (backprop through time)
    if jitted:
        loss, diagnostics, next_state, grads = grads_fn_jitted(params, state, train_inputs, train_targets, train_forcings)
    else:
        loss, diagnostics, next_state, grads = grads_fn(params, state, train_inputs, train_targets, train_forcings, model_config, task_config)
    
    # optimizer step
    updates, opt_state = optimiser.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    
    mean_grad = np.mean(jax.tree_util.tree_flatten(jax.tree_util.tree_map(lambda x: np.abs(x).mean(), grads))[0])

    print(f"Loss: {loss:.4f}, Mean |grad|: {mean_grad:.6f}")
    
print("  autoregressive rollout (forward step)")
# @title Autoregressive rollout (keep the loop in JAX)
predictions = run_forward_jitted(
    rng=jax.random.PRNGKey(0),
    inputs=train_inputs,
    targets_template=train_targets * np.nan,
    forcings=train_forcings)

predictions
    
print("Inputs:  ", train_inputs.dims.mapping)
print("Targets: ", train_targets.dims.mapping)
print("Forcings:", train_forcings.dims.mapping)



# setup optimiser


