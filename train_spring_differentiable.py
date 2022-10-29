import numpy as onp
import pickle

import jax
from jax import jit, vmap
from jax.tree_util import tree_map
import jax.numpy as jnp
from jax.random import PRNGKey, split

import optax

from spring_model import generate_data_batch, simulate
from flax_transformer_v2 import (
    v_sigmoid_trunc_gaussian_mixture_sample,
    sigmoid_trunc_gaussian_mixture_logpdf,
    TransformerConfig,
    GaussianMixturePosteriorConfig,
    IndependentGaussianMixtureConfig,
    IndependentGaussianMixtures,
)


def initialize_model_and_state(
    key: PRNGKey, obs_length, num_input_vars, lr, deterministic, load_idx=None
):
    key, *sks = split(key, 10)

    t_cfg = TransformerConfig(
        obs_emb_hidden_sizes=(100,),
        d_model=40,
        add_positional_encoding=True,
        deterministic=deterministic,
    )
    g_cfg = GaussianMixturePosteriorConfig(
        emb_size=50, covariance_eps=1e-3, use_tril=True
    )
    ig_cfg = IndependentGaussianMixtureConfig(
        group_variables=(2,), num_mixtures_per_group=(5,)
    )

    m = IndependentGaussianMixtures(ig_cfg, g_cfg, t_cfg)

    variables = m.init(
        {"params": sks[0], "dropout": sks[1]}, jnp.ones((2, obs_length, num_input_vars))
    )
    state, params = variables.pop("params")
    del variables

    tx = optax.adam(learning_rate=lr)
    opt_state = tx.init(params)
    if load_idx is not None:
        with open(f"params_{load_idx}", "rb") as f:
            params = pickle.load(f)
        with open(f"opt_state_{load_idx}", "rb") as f:
            opt_state = pickle.load(f)
    return m, tx, opt_state, params, state


def sum_gaussian_logpdf(x, mean, cov):
    """
    x, mean and cov should be of shape B,1, and it computes a vectorized gaussian logpdf and returns the sum
    """
    assert mean.shape[-1] == 1 and len(mean.shape) == 2
    logpdf = jax.vmap(jax.scipy.stats.norm.logpdf)(
        x[:, 0], mean[:, 0], cov[:, 0] + 1e-6
    )
    return logpdf.sum()


def update_step(
    encoder_apply,
    simulate,
    q,
    latents,
    opt_state,
    params,
    state,
    key,
    tx,
    proportional_noise,
    beta,
):
    assert beta >= 0 and beta <= 1

    def loss(params):
        batch_size = q.shape[0]
        dropout_key, other_key = split(key)
        d_params = encoder_apply(
            {"params": params, **state}, q, rngs={"dropout": dropout_key}
        )

        mix_p, means, covs = (
            d_params[0]["mix_p"],
            d_params[0]["means"],
            d_params[0]["covariance_matrices"],
        )
        z_sample = v_sigmoid_trunc_gaussian_mixture_sample(
            split(other_key, mix_p.shape[0]), mix_p, means, covs, 0.1, 10.0
        )
        z_mass, z_k = z_sample[:, 0], z_sample[:, 1]
        x_hat = simulate(
            batch_y0=jnp.stack(
                [jnp.zeros((batch_size,)), jnp.ones((batch_size,)) * 1.0], axis=1
            ),
            num_times=q.shape[1],
            batch_mass=z_mass,
            batch_k=z_k,
        )
        
        latent_loss = -sigmoid_trunc_gaussian_mixture_logpdf(
                latents, mix_p, means, covs, 0.1, 10.0
            ).mean()
        
        reconstruction_loss = -vmap(sum_gaussian_logpdf)(q, x_hat, jnp.abs(x_hat) * proportional_noise).mean()

        l = latent_loss * (1 - beta) + beta * reconstruction_loss
        return l, reconstruction_loss

    (l, rc), grads = jax.value_and_grad(loss, has_aux=True)(params)
    grads_nan = jnp.any(jnp.array(jax.tree_map(lambda x: jnp.any(jnp.isnan(x)), jax.tree_leaves(grads))))
    grads = jax.lax.cond( grads_nan, 
                         lambda: jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), grads),
                         lambda: grads)
    updates, opt_state = tx.update(jax.tree_util.tree_map(lambda x: jnp.clip(x,-1.0,1.0),grads), opt_state)
    params = optax.apply_updates(params, updates)
    return opt_state, params, l, rc, grads_nan


if __name__ == "__main__":
    key = PRNGKey(67458)

    ## data params
    generation_size = int(1e4)
    obs_length = 100
    num_epochs = 9999999
    batch_size = 200

    ## reconstruction params
    proportional_noise = 0.05
    beta = 1.0

    ## checkpoint params
    save_params = 1
    print_every = 10
    chkpt_folder = "chkpts_beta/"

    m, tx, opt_state, params, state = initialize_model_and_state(
        PRNGKey(62389456),
        obs_length=obs_length,
        num_input_vars=1,
        lr=1e-4,
        deterministic=False,
    )
    load_idx = None
    
    while True:
      finished = False
      
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
          for i in range(generation_size // batch_size - 1):
              key, subkey = split(key)
              # opt_state, params, l, rc, gn = update_step(
              opt_state, params, l, rc, gn = jax.jit(update_step, static_argnums=(0,1,8,10))(
                  m.apply,
                  simulate,
                  q=jnp.array(q[i * batch_size : (i + 1) * batch_size]),
                  latents=jnp.array(latents[i * batch_size : (i + 1) * batch_size]),
                  opt_state=opt_state,
                  params=params,
                  state=state,
                  key=subkey,
                  tx=tx,
                  proportional_noise=proportional_noise,
                  beta=beta,
              )
              if i % print_every == 0:
                  print(e, i, l, rc)
              if jnp.isnan(l) or l == 0 or jnp.isinf(l) or gn:
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
