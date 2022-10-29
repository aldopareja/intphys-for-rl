import pickle
import numpy as onp

import jax
from jax.random import split, PRNGKey
from jax import numpy as jnp
from jax import vmap, jit, pmap
from jax.tree_util import tree_map

from flax import linen as nn

import optax

from RealNVP_flow import RealNVP_trunc, RealNVPConfig
from flax_transformer_v2 import TransformerConfig, TransformerDiagGaussian

from box_model import MAX_LATENT, generative_model, simulate


class Model(nn.Module):
    flow_config: RealNVPConfig
    transformer_cfg: TransformerConfig

    def setup(self):
        self.obs_to_q = TransformerDiagGaussian(self.transformer_cfg)
        self.flow = RealNVP_trunc(self.flow_config)

    def __call__(self, obs, latent):
        mu, cov = self.obs_to_q(obs)
        key = self.make_rng("rsample_key")
        return self.flow.log_probability(latent, mu, cov), self.rsample(mu, cov, key)

    def rsample(self, mu, cov, key):
        key, _ = split(key)
        return self.flow.rsample(mu, cov, key)


def load_chkpt(load_idx, chkpt_folder):
    with open(f"{chkpt_folder}params_{load_idx}", "rb") as f:
        params = pickle.load(f)
    with open(f"{chkpt_folder}opt_state_{load_idx}", "rb") as f:
        opt_state = pickle.load(f)
    with open(f"{chkpt_folder}key_{load_idx}", "rb") as f:
        loaded_key = pickle.load(f)
    return params, opt_state, loaded_key


def save_chkpt(save_idx, chkpt_folder, params, opt_state, old_key):
    with open(f"{chkpt_folder}params_{save_idx}", "wb") as f:
        pickle.dump(params, f)
    with open(f"{chkpt_folder}opt_state_{save_idx}", "wb") as f:
        pickle.dump(opt_state, f)
    with open(f"{chkpt_folder}key_{save_idx}", "wb") as f:
        pickle.dump(old_key, f)


def initialize_model_and_state(
    key: PRNGKey, obs_length, lr, num_input_vars, load_idx=None, chkpt_folder=None, deterministic=False
):
    key, *sks = split(key, 10)

    f_cfg = RealNVPConfig(
        f_mlp_hidden_size=40,
        f_mlp_num_layers=2,
        num_latent_vars=10,
        num_flow_layers=16,
        # q_mlp_hidden_size=200,
        # q_mlp_num_layers=5,
        out_min=-MAX_LATENT,
        out_max=MAX_LATENT,
    )

    t_cfg = TransformerConfig(
        num_enc_layers=1,
        num_dec_layers=1,
        dropout_rate=0.1,
        deterministic=deterministic,
        d_model=100,
        add_positional_encoding=False,
        obs_emb_hidden_sizes=(100,),
        num_latents=10,
    )

    m = Model(flow_config=f_cfg, transformer_cfg=t_cfg)

    variables = m.init(
        {"params": sks[0], "dropout": sks[1], "rsample_key": sks[2]},
        jnp.ones((2, obs_length, num_input_vars)),
        jnp.ones((2, 10)),
    )
    state, params = variables.pop("params")
    del variables

    tx = optax.adam(learning_rate=lr)
    opt_state = tx.init(params)

    if load_idx is not None:
        params, opt_state, _ = load_chkpt(load_idx, chkpt_folder)

    return m, tx, opt_state, params, state


def update_step(
    apply_func,
    point_cloud,
    latents,
    opt_state,
    params,
    state,
    key,
):
    assert beta >= 0 and beta <= 1

    def loss(params):
        batch_size = point_cloud.shape[0]
        flow_key, dropout_key, sim_k = split(key, 3)
        latent_log_prob, z_sample = apply_func(
            {"params": params, **state},
            point_cloud,
            latents,
            rngs={"rsample_key": flow_key, "dropout": dropout_key},
        )

        pc_hat, _ = vmap(simulate, in_axes=(0, 0, None, None))(
            split(sim_k, batch_size),
            z_sample,
            0.01,#OBSERVATION_NOISE, for the resimulation better to add very little noise, it's enough on the observation
            NUM_POINTS,
        )

        # reconstruction_loss = -vmap(sum_gaussian_logpdf)(q, x_hat, jnp.abs(x_hat) * proportional_noise).mean()
        reconstruction_loss = ((point_cloud - pc_hat) ** 2).mean()

        l = -latent_log_prob.mean() * (1 - beta) + beta * reconstruction_loss
        return l, jax.lax.stop_gradient(reconstruction_loss)

    (l, rc), grads = jax.value_and_grad(loss, has_aux=True)(params)
    grads_nan = jnp.any(
        jnp.array(tree_map(lambda x: jnp.any(jnp.isnan(x)), jax.tree_leaves(grads)))
    )
    grads = jax.lax.cond(
        grads_nan, lambda: tree_map(lambda x: jnp.zeros_like(x), grads), lambda: grads
    )
    updates, opt_state = tx.update(
        tree_map(lambda x: jnp.clip(x, -1.0, 1.0), grads), opt_state
    )
    params = optax.apply_updates(params, updates)
    return opt_state, params, l, rc, grads_nan


if __name__ == "__main__":
    key = PRNGKey(639265)
    # data generation params
    OBSERVATION_NOISE = 0.5
    NUM_POINTS = 100
    num_epochs = 9999999
    batch_size = 200
    generation_size = int(12e3)
    gen_ks = jit(
        lambda k: split(k, generation_size).reshape(2, generation_size // 2, 2)
    )

    ## reconstruction params
    beta = 0.0

    ## checkpoint params
    save_params = 1
    print_every = 20
    chkpt_folder = "box_chkpts_flow_beta_00/"
    load_idx = 0

    m, tx, opt_state, params, state = initialize_model_and_state(
        key=PRNGKey(1574),
        obs_length=NUM_POINTS,
        lr=1e-4,
        num_input_vars=3,
    )

    while True:
        finished = False

        save_idx = 0 if load_idx is None else load_idx + 1
        if load_idx:
            params, opt_state, loaded_key = load_chkpt(load_idx, chkpt_folder)
        else:
            loaded_key = key

        for e in range(num_epochs):
            old_key = key
            key = loaded_key if e == 0 else key
            key, subkey = split(key)
            point_cloud, latents, _ = tree_map(
                onp.array,
                pmap(
                    jit(
                        vmap(generative_model, in_axes=(0, None, None)),
                        static_argnums=(1, 2),    
                    ),
                    in_axes=(0, None, None),
                    static_broadcasted_argnums=(1, 2),
                )(gen_ks(key), OBSERVATION_NOISE, NUM_POINTS),
            )
            
            point_cloud, latents = tree_map(
                lambda x: onp.concatenate([x[0], x[1]]), [point_cloud, latents]
            )

            for i in range(generation_size // batch_size - 1):
                key, subkey = split(key)
                # opt_state, params, l, rc, gn = update_step(
                opt_state, params, l, rc, gn = jit(update_step, static_argnums=(0,))(
                    m.apply,
                    jnp.array(point_cloud[i * batch_size : (i + 1) * batch_size]),
                    jnp.array(latents[i * batch_size : (i + 1) * batch_size]),
                    opt_state,
                    params,
                    state,
                    key,
                )
                if i % print_every == 0:
                    print(e, i, l, rc)
                if gn:
                    print("######gradients fucked#######")
                if jnp.isnan(l) or l == 0 or jnp.isinf(l):
                    print("failed", e, i, l, rc)
                    finished = True
                    break
                if (i + 1) % save_params == 0:
                    save_chkpt(save_idx, chkpt_folder, params, opt_state, old_key)
            if finished:
                print("current idx ---->", load_idx)
                break
        load_idx += 1
