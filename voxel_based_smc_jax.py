import jax.numpy as jnp
from jax import ShapedArray, random, jit
from functools import partial

def get_occupancy_grid(object_poses, 
                       num_side_bins,
                       false_positive_noise,
                       false_negative_noise):
  bin_size = 1/num_side_bins
  bin_idx = object_poses / 

@partial(jit, static_argnames=('num_objects'))
def multi_object_model(T: int, num_objects: int, num_side_bins: int):
  num_dims = 2

  k = random.PRNGKey(0)
  k, sk = random.split(k)
  x = random.uniform(sk, shape=(num_dims, num_objects))

  xs = x


  return xs



