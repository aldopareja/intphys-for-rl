from functools import partial
from typing import Sequence

import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint
from jax import lax, jit, vmap, random

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

@partial(jit, static_argnums=[1,2,3])
def generate_data_batch(key, batch_size, num_times, noise_std_ratio = 0.05):
    key, *subkeys = random.split(key, 10)
    # batch_y0 = random.uniform(subkeys[0], (batch_size,2), minval=-10.0, maxval=10.0)
    # batch_y0 = jnp.stack([jnp.zeros((batch_size,)),batch_y0],axis=1)
    batch_y0 = jnp.stack([jnp.zeros((batch_size,)),jnp.ones((batch_size,))*1.0],axis=1)
    # batch_c = random.choice(subkeys[0], jnp.array([10.0,1.0]), (batch_size,))
    batch_mass = random.uniform(subkeys[1], (batch_size,), minval=0.1, maxval=1.0)
    # batch_mass = random.choice(subkeys[1], jnp.array([0.1,1.]), (batch_size,))
    # batch_k = batch_c * batch_mass
    batch_k = random.uniform(subkeys[2], (batch_size,), minval=1.0, maxval=10.0)
    batch_k_noise = add_proportional_noise(batch_k, noise_std_ratio, subkeys[4])
    batch_mass_noise = add_proportional_noise(batch_mass, noise_std_ratio, subkeys[5])
    batch_positions = simulate(batch_y0, num_times, batch_mass_noise, batch_k_noise)
    batch_positions = add_proportional_noise(batch_positions, noise_std_ratio,subkeys[3], clip=False)
    #expand positions since the MLP should learn to embed per observation.
    return batch_positions, \
           jnp.stack([batch_mass, batch_k],axis=1)[:,None]
           
def simulate(batch_y0, num_times, batch_mass, batch_k):
  return vect_odeint(spring_dy_dt, batch_y0, jnp.linspace(0,10,num_times), batch_mass, batch_k)[:,:,1:]