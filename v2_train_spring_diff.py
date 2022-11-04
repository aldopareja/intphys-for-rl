import pickle

import numpy as onp

import jax
from jax.tree_util import tree_map
from jax import vmap
from jax.random import PRNGKey, split
from jax import numpy as jnp

import optax
from flax import linen as nn

from RealNVP_flow import RealNVPConfig, RealNVP_trunc, ObstoQ
from spring_model import generate_data_batch, simulate

class Model(nn.Module):
  config : RealNVPConfig
  
  def setup(self):
    self.obs_to_q = ObstoQ(self.config)
    self.flow = RealNVP_trunc(self.config)
    
  def __call__(self, obs, latent, key):
    mu, cov = self.obs_to_q(obs)
    return self.flow.log_probability(latent, mu, cov), self.rsample(mu, cov, key)
  
  def rsample(self, mu, cov, key):
    key, _ = split(key)
    return self.flow.rsample(mu,cov,key)
  
  def rsample_test(self, obs, key):
    mu, cov = self.obs_to_q(obs)
    key, _ = split(key)
    return self.flow.rsample(mu,cov,key)
  
  def log_prob_test(self, latent, mu, cov):
    return self.flow.log_probability(latent, mu, cov)
  
  def get_q(self, obs):
    return self.obs_to_q(obs)
  
def sum_gaussian_logpdf(x, mean, cov):
    """
    x, mean and cov should be of shape B,1, and it computes a vectorized gaussian logpdf and returns the sum
    """
    assert mean.shape == x.shape
    logpdf = vmap(jax.scipy.stats.norm.logpdf)(
        x, mean, cov + 1e-6
    )
    return logpdf.sum()
  
def update_step(
    apply_func,
    simulate,
    q,
    latents,
    opt_state,
    params,
    key,
    tx,
    proportional_noise,
    beta,
):
    assert beta >= 0 and beta <= 1

    def loss(params):
        batch_size = q.shape[0]
        k1, _ = split(key)
        latent_log_prob, z_sample = apply_func(
            {"params": params}, q, latents, k1
        )
        
        z_mass, z_k = z_sample[:, 0], z_sample[:, 1]
        x_hat = simulate(
            batch_y0=jnp.stack(
                [jnp.zeros((batch_size,)), jnp.ones((batch_size,)) * 1.0], axis=1
            ),
            num_times=q.shape[1],
            batch_mass=z_mass,
            batch_k=z_k,
        )[:,:,0]
        
        
        # reconstruction_loss = -vmap(sum_gaussian_logpdf)(q, x_hat, jnp.abs(x_hat) * proportional_noise).mean()
        reconstruction_loss = ((q-x_hat)**2).mean()

        l = -latent_log_prob.mean() * (1 - beta) + beta * reconstruction_loss
        return l, jax.lax.stop_gradient(-latent_log_prob.mean())

    (l, rc), grads = jax.value_and_grad(loss, has_aux=True)(params)
    grads_nan = jnp.any(jnp.array(tree_map(lambda x: jnp.any(jnp.isnan(x)), jax.tree_leaves(grads))))
    grads = jax.lax.cond( grads_nan, 
                         lambda: tree_map(lambda x: jnp.zeros_like(x), grads),
                         lambda: grads)
    updates, opt_state = tx.update(tree_map(lambda x: jnp.clip(x,-1.0,1.0),grads), opt_state)
    params = optax.apply_updates(params, updates)
    return opt_state, params, l, rc, grads_nan

def initialize_model_and_state(
    key: PRNGKey, obs_length, lr, load_idx = None, chkpt_folder=None
):
    key, *sks = split(key, 10)

    cfg = RealNVPConfig(
        f_hidden_size=20,
        f_num_layers=3,
        num_latent_vars=2,
        num_flow_layers=5,
        q_mlp_hidden_size=200,
        q_mlp_num_layers=5,
        out_min=0.01,
        out_max=10.1,
    )

    m = Model(cfg)
    

    variables = m.init(
        {"params": sks[0]}, jnp.ones((10,obs_length)), jnp.ones((10,cfg.num_latent_vars)), PRNGKey(0)
    )
    state, params = variables.pop("params")
    del variables

    tx = optax.adam(learning_rate=lr)
    opt_state = tx.init(params)
    
    if load_idx is not None:
        with open(f'{chkpt_folder}params_{load_idx}', 'rb') as f:
            params = pickle.load(f)
        with open(f'{chkpt_folder}opt_state_{load_idx}', 'rb') as f:
            opt_state = pickle.load(f)    
        
    return m, tx, opt_state, params
  
if __name__ == "__main__":
    key = PRNGKey(67458)

    ## data params
    generation_size = int(1e4)
    obs_length = 100
    num_epochs = 9999999
    batch_size = 200

    ## reconstruction params
    proportional_noise = 0.05
    beta = 0.5

    ## checkpoint params
    save_params = 1
    print_every = 10
    chkpt_folder = "chkpts_flow_beta_05/"

    m, tx, opt_state, params = initialize_model_and_state(
        PRNGKey(62389456),
        obs_length=obs_length,
        lr=1e-4,
    )
    load_idx = None
    
    while True:
      finished = False
      # save_idx = load_idx
      save_idx = 0 if load_idx is None else load_idx+1
      if load_idx is not None:
        with open(f'{chkpt_folder}params_{load_idx}', 'rb') as f:
            params = pickle.load(f)
        with open(f'{chkpt_folder}opt_state_{load_idx}', 'rb') as f:
            opt_state = pickle.load(f)    
        with open(f'{chkpt_folder}key_{load_idx}', 'rb') as f:
            loaded_key = pickle.load(f)  
      else:
        loaded_key = key

      for e in range(num_epochs):
          old_key = key
          key = loaded_key if e == 0 else key
          key, subkey = split(key)
          q, latents = tree_map(
              onp.array,
              generate_data_batch(subkey, generation_size, num_times=obs_length),
          )
          q = q[:,:,0]
          latents = latents[:,0,:]
          for i in range(generation_size // batch_size - 1):
              key, subkey = split(key)
              # opt_state, params, l, rc, gn = update_step(
              opt_state, params, l, rc, gn = jax.jit(update_step, static_argnums=(0,1,7,9))(
                  m.apply,
                  simulate,
                  q=jnp.array(q[i * batch_size : (i + 1) * batch_size]),
                  latents=jnp.array(latents[i * batch_size : (i + 1) * batch_size]),
                  opt_state=opt_state,
                  params=params,
                  key=subkey,
                  tx=tx,
                  proportional_noise=proportional_noise,
                  beta=beta,
              ) 
              if i % print_every == 0:
                  print(e, i, l, rc)
              if gn:
                print('######gradients fucked#######')
              if jnp.isnan(l) or l == 0 or jnp.isinf(l):
                  print("failed", e, i, l, rc)
                  finished = True
                  break
              if (i + 1) % save_params == 0:
                  with open(f"{chkpt_folder}params_{save_idx}", "wb") as f:
                      pickle.dump(params, f)
                  with open(f"{chkpt_folder}opt_state_{save_idx}", "wb") as f:
                      pickle.dump(opt_state, f)
                  with open(f"{chkpt_folder}key_{save_idx}", "wb") as f:
                      pickle.dump(old_key, f)

          if finished:
              print("current idx ---->", load_idx)
              break
      load_idx += 1

