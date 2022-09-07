# Copyright 2022 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Transformer-based language model.
Reusing decoder only model from examples/wmt.
"""

# pylint: disable=attribute-defined-outside-init
# See issue #620.
# pytype: disable=wrong-arg-count
# pytype: disable=wrong-keyword-args
# pytype: disable=attribute-error

from functools import partial
from typing import Sequence

import optax
from flax import linen as nn
from flax import struct, serialization
import jax
import jax.numpy as jnp
import numpy as onp
from jax.random import PRNGKey, split
from jax.experimental.ode import odeint
from jax import lax, jit, vmap, random

@struct.dataclass
class TransformerConfig:
  """Global hyperparameters used to minimize obnoxious kwarg plumbing."""
  num_heads: int = 4
  num_enc_layers: int = 2
  num_dec_layers: int = 2
  dropout_rate: float = 0.1
  deterministic: bool = False
  d_model: int = 40
  max_len: int = 3000
  obs_emb_hidden_sizes: Sequence[int] = (100,)
  num_mixtures: int = 4
  num_latents: int = 2
  covariance_eps: float = 1e-5
  checkpoint: str = ''
  default_device: int = 0

@jit
def spring_dy_dt(y, t, mass, k):
    """
    params:
        y: vector with velocity in the first position and position in the second
        t: time (not used)
        mass: scalar
        k: scalar, spring stiffness
        
        It implements the system followed by a spring in 1d
        
        :math:`\dot{v} = -\frac{k}{m}x`
        :math:`\dot{x} = v`
        
    return:
        dy_dt: 2 dimensional vector with the derivative of the velocity and position in the first and second position respectively
    """
    x_v = jnp.flip(y)
    theta = jnp.stack([-k/mass, 1])
    dy_dt = x_v * theta
    return dy_dt

# odeint_jit = odeint, static_argnums=(0,)
vect_odeint = vmap(odeint,in_axes=(None,0,None,0,0))

def add_proportional_noise(array, std_noise_ratio, subkey, clip=True):
  std = std_noise_ratio * lax.abs(array) + 1e-6
  array = jax.random.normal(subkey, array.shape) * std + array
  if clip:
    array = jnp.clip(array, a_min=1e-5)
  return array

@partial(jit, static_argnums=[1,2])
def generative_model(key, batch_size, num_times, noise_std_ratio = 0.05):
    key, *subkeys = random.split(key, 10)
    batch_y0 = random.uniform(subkeys[0], (batch_size,), minval=-10.0, maxval=10.0)
    # batch_y0 = jnp.stack([jnp.zeros((batch_size,)),batch_y0],axis=1)
    batch_y0 = jnp.stack([jnp.zeros((batch_size,)),jnp.ones((batch_size,))*1.0],axis=1)
    # batch_c = random.choice(subkeys[0], jnp.array([10.0,1.0]), (batch_size,))
    batch_mass = random.uniform(subkeys[1], (batch_size,), minval=0.1, maxval=1.0)
    # batch_mass = random.choice(subkeys[1], jnp.array([0.1,1.]), (batch_size,))
    # batch_k = batch_c * batch_mass
    batch_k = random.uniform(subkeys[2], (batch_size,), minval=1.0, maxval=10.0)
    batch_k_noise = add_proportional_noise(batch_k, noise_std_ratio, subkeys[4])
    batch_mass_noise = add_proportional_noise(batch_mass, noise_std_ratio, subkeys[5])
    all_y = vect_odeint(spring_dy_dt, batch_y0, jnp.linspace(0,10,num_times), batch_mass_noise, batch_k_noise)
    batch_positions = all_y[:,:,1]
    batch_positions = add_proportional_noise(batch_positions, noise_std_ratio,subkeys[3], clip=False)
    #expand positions since the MLP should learn to embed per observation.
    return batch_positions[:,:,None], \
           jnp.stack([batch_mass, batch_k],axis=1)[:,None]

class ObsEmbed(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, x):
        for feat in self.config.obs_emb_hidden_sizes:
            x = nn.Dense(feat)(x)
            x = nn.relu(x)
        
        x = nn.Dense(self.config.d_model)(x)
        
        return x

class PositionalEncoder(nn.Module):
    config: TransformerConfig
    #todo: shall we add dropout? there's documentation to read though.
    
    @staticmethod
    def init_pe(d_model: int, max_length: int):
        positions = jnp.arange(max_length)[:, None]
        div_term = jnp.exp(jnp.arange(0, d_model) * (-jnp.log(10000.0)/d_model))
        
        temp = positions * div_term
        even_mask = positions % 2 == 0
        
        pe = jnp.where(even_mask, jnp.sin(temp), jnp.cos(temp))
        
        return pe[None,:,:]
    
    
    @nn.compact
    def __call__(self, x):
        cfg = self.config
        pe = self.variable('consts', 'pe', PositionalEncoder.init_pe, cfg.d_model, cfg.max_len)
        # batch_apply_pe = nn.vmap(lambda x, pe: x + pe[:x.shape[0]], in_axes=(0,None))
        return x + pe.value[:,:x.shape[1]]

class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block.
    Args:
      config: TransformerConfig dataclass containing hyperparameters.
    """
    config: TransformerConfig

    @nn.compact
    def __call__(self, inputs):
        """Applies Transformer MlpBlock module."""
        cfg = self.config
        x = nn.Dense(
            cfg.d_model * 2)(inputs)
        x = nn.relu(x)
        x = nn.Dropout(rate=cfg.dropout_rate)(
            x, deterministic=cfg.deterministic)
        output = nn.Dense(cfg.d_model)(x)
        output = nn.Dropout(rate=cfg.dropout_rate)(
            output, deterministic=cfg.deterministic)
        return output


class EncoderLayer(nn.Module):
  """Transformer encoder layer.
  Args:
    config: TransformerConfig dataclass containing hyperparameters.
  """
  config: TransformerConfig

  @nn.compact
  def __call__(self,
               inputs):
    """Applies EncoderBlock module.
    Args:
      inputs: input data for decoder
    Returns:
      output after transformer encoder block.
    """
    cfg = self.config

    # Encoder block.
    assert inputs.ndim == 3
    x = nn.LayerNorm()(inputs)
    x = nn.SelfAttention(
        num_heads=cfg.num_heads,
        use_bias=False, #should we use bias? I guess it doesn't matter
        broadcast_dropout=False,
        dropout_rate=cfg.dropout_rate,
        deterministic=cfg.deterministic,
        decode=False)(x)
    x = nn.Dropout(rate=cfg.dropout_rate)(
        x, deterministic=cfg.deterministic)
    x = x + inputs

    # MLP block.
    z = nn.LayerNorm()(x)
    z = MlpBlock(config=cfg)(z)

    return x + z

class DecoderLayer(nn.Module):
  """Transformer encoder-decoder layer.
  Args:
    config: TransformerConfig dataclass containing hyperparameters.
  """
  config: TransformerConfig

  @nn.compact
  def __call__(self, output_emb, encoded_input):
    """Applies EncoderBlock module.
    Args:
      inputs: input data for decoder
    Returns:
      output after transformer encoder block.
    """
     
    cfg = self.config

    # Decoder block.
    assert encoded_input.ndim == 3 and output_emb.ndim == 3
    x = nn.LayerNorm()(output_emb)
    x = nn.SelfAttention(
        num_heads=cfg.num_heads,
        use_bias=False,
        broadcast_dropout=False,
        dropout_rate=cfg.dropout_rate,
        deterministic=cfg.deterministic,
        decode=False)(x)
    x = nn.Dropout(rate=cfg.dropout_rate)(
        x, deterministic=cfg.deterministic)
    x = x + output_emb

    z = nn.LayerNorm()(x)

    x = nn.MultiHeadDotProductAttention(num_heads=cfg.num_heads,
        use_bias=False,
        broadcast_dropout=False,
        dropout_rate=cfg.dropout_rate,
        deterministic=cfg.deterministic,
        decode = False)(z, encoded_input)

    x = x + z

    # MLP block.
    z = nn.LayerNorm()(x)

    z = MlpBlock(config=cfg)(z)

    return x + z

class TransformerStack(nn.Module):
  config: TransformerConfig

  @nn.compact
  def __call__(self, q):
    cfg = self.config
    x = ObsEmbed(cfg)(q)
    enc_input = PositionalEncoder(cfg)(x)

    # for _ in range(cfg.num_enc_layers):
    enc_input = nn.Sequential([EncoderLayer(cfg) for _ in range(cfg.num_enc_layers)])(enc_input)

    start = self.param('start', nn.initializers.uniform(),(1,1,cfg.d_model))
    so_far_dec = start.repeat(q.shape[0],axis=0)
    

    seq_decoded = []
    for _ in range(cfg.num_mixtures):
      dec = so_far_dec
      # dec = decoder(dec, enc_input)
      for _ in range(cfg.num_dec_layers):
        dec = DecoderLayer(cfg)(dec,enc_input)
      so_far_dec = jnp.concatenate([so_far_dec, lax.stop_gradient(dec[:,-1:])],axis=1)
      seq_decoded.append(dec[:,-1:,:])
    seq_decoded = jnp.concatenate(seq_decoded, axis=1)


    num_means = cfg.num_latents
    num_covariance_terms = int(num_means * (num_means + 1) / 2)
    num_mixture_params = 1 + num_means + num_covariance_terms
    dist_params = nn.Sequential([nn.Dense(cfg.d_model*2),
                                  nn.relu,
                                  nn.Dense(num_mixture_params)])(seq_decoded)

    return dist_params

class TransformerGaussianMixturePosterior(nn.Module):
  config: TransformerConfig

  @nn.compact
  def __call__(self, q):
    cfg = self.config
    num_means = cfg.num_latents
    num_covariance_terms = int(num_means * (num_means + 1) / 2)

    mixture_params = TransformerStack(cfg)(q)
    mix_p = jnp.exp(mixture_params[:,:,0])
    means = mixture_params[:,:, 1: 1+num_means]
    covariance_terms = mixture_params[:,:,-num_covariance_terms:]
    covariance_matrices = self.get_cov_matrices_from_vectors(covariance_terms, num_means, cfg.covariance_eps)

    return dict(mix_p=mix_p/mix_p.sum(axis=1,keepdims=True), means=means, covariance_matrices=covariance_matrices)

  @staticmethod
  def get_cov_matrices_from_vectors(covariance_terms, num_means, eps):
    x = covariance_terms
    output_shape = (*x.shape[:-1], num_means, num_means)
    x = jnp.concatenate([x, x[:,:,num_means:][:,:,::-1]], axis=-1)
    x = x.reshape(output_shape)
    x = jnp.triu(x)

    eps = jnp.eye(num_means)[None,None] * eps
    cov_matrices = jnp.matmul(x, x.swapaxes(-2,-1)) + eps
    return cov_matrices

def gaussian_mixture_logpdf(latents, dist_params):
    mix_p, means, covs = dist_params['mix_p'], dist_params['means'], dist_params['covariance_matrices']
    
    normals_log_prob = jax.scipy.stats.multivariate_normal.logpdf(latents, means, covs)
    category_log_prob = jax.lax.log(mix_p)
    
    return jax.nn.logsumexp(normals_log_prob + category_log_prob, axis=1)

def gaussian_mixture_sample(dist_params):
    mix_p, means, covs = dist_params['mix_p'], dist_params['means'], dist_params['covariance_matrices']



def update_step(apply_fn, q, latents, opt_state, params, state, dropout_key):
    def loss(params):
        d_params = apply_fn({'params':params, **state},
                                    q, rngs={'dropout': dropout_key})
        l = -gaussian_mixture_logpdf(latents, d_params).mean()
        return l
    
    l, grads = jax.value_and_grad(loss, has_aux=False)(params)
    updates, opt_state = tx.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return opt_state, params, l

if __name__ == "__main__":
  key, *subkeys = split(PRNGKey(11234), 4)
  cfg = TransformerConfig(checkpoint='checkpoint-1826pm', default_device=0)
  # jax.default_device = jax.devices()[cfg.default_device]
  m = TransformerGaussianMixturePosterior(cfg)
  batch_size = 4000
  obs_lenght= 200
  init_rngs = {'params': subkeys[0], 'dropout':subkeys[1]}
  variables = m.init(init_rngs, jnp.ones((batch_size, obs_lenght, 1)))
  state, params = variables.pop('params')
  del variables
  tx = optax.adam(learning_rate=0.001)
  opt_state = tx.init(params)

  total_data_size = int(3e6)
  num_epochs = 50*12*6
  finish = False
  params_checkpoints = {}
  save_params = 50
  loaded_chkpnt = False
  # def train(key, params, state, tx, opt_state, num_epochs, total_data_size):
  for e in range(num_epochs):
      if cfg.checkpoint and not loaded_chkpnt:
        loaded_chkpnt = True
        with open(cfg.checkpoint, 'rb') as f:
          (key, params, opt_state) = serialization.from_bytes((key, params, opt_state), f.read())

      key, subkey = split(key)
      q, latents = jax.tree_map(onp.array, generative_model(subkey, total_data_size, obs_lenght))
      for i in range(total_data_size//batch_size):
          if (i+1) % save_params == 0:
              params_checkpoints[f'opt_state_{e}_{i}'] = jax.tree_map(onp.array,opt_state)
              params_checkpoints[f'params_{e}_{i}'] = jax.tree_map(onp.array,params)
              params_checkpoints[f'key_{e}_{i}'] = jax.tree_map(onp.array,key)
          key, subkey = split(key)
          opt_state, params, l = jit(update_step, static_argnums=(0,))(
          # opt_state, params, l = update_step(
              m.apply, 
              jnp.array(q[i*batch_size: (i+1)*batch_size]), 
              jnp.array(latents[i*batch_size: (i+1)*batch_size]),
              opt_state, params, state, subkey)
          if i % 10 == 0:      
              print(e,i,l)
          if jnp.isnan(l) or l == 0:
              print('failed',e,i,l)
              finish = True
              break
      if finish:
          break

