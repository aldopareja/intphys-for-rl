from dataclasses import dataclass
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


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


@dataclass
class OptimCfg:
    max_lr: float = 1e-3
    num_steps: int = 10000
    pct_start: float = 0.01
    div_factor: float = 1e1
    final_div_factor: float = 1e1
    weight_decay: float = 0.00005
    gradient_clipping: float = 5.0


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


def save_chkpt(save_idx, chkpt_folder, params, opt_state, old_key, extra=""):
    with open(f"{chkpt_folder}params_{save_idx}_{extra}", "wb") as f:
        pickle.dump(params, f)
    with open(f"{chkpt_folder}opt_state_{save_idx}_{extra}", "wb") as f:
        pickle.dump(opt_state, f)
    with open(f"{chkpt_folder}key_{save_idx}_{extra}", "wb") as f:
        pickle.dump(old_key, f)


def initialize_model_and_state(
    key: PRNGKey,
    obs_length,
    num_input_vars,
    optim_cfg: OptimCfg,
    load_idx=None,
    chkpt_folder=None,
    deterministic=False,
):
    key, *sks = split(key, 10)

    f_cfg = RealNVPConfig(
        f_hidden_size=80,
        f_num_layers=3,
        num_latent_vars=10,
        num_flow_layers=24,
        out_min=-MAX_LATENT,
        out_max=MAX_LATENT,
        stabilization_factor=2.0,
    )

    t_cfg = TransformerConfig(
        num_enc_layers=1,
        num_dec_layers=3,
        dropout_rate=0.1,
        deterministic=deterministic,
        d_model=200,
        add_positional_encoding=False,
        obs_emb_hidden_sizes=(200,),
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

    schedule = optax.cosine_onecycle_schedule(
        optim_cfg.num_steps,
        optim_cfg.max_lr,
        optim_cfg.pct_start,
        optim_cfg.div_factor,
        optim_cfg.final_div_factor,
    )
    tx = optax.chain(
        optax.clip_by_global_norm(optim_cfg.gradient_clipping),
        optax.adamw(learning_rate=schedule, weight_decay=optim_cfg.weight_decay),
    )
    # tx = optax.sgd(learning_rate=1e-9)
    opt_state = tx.init(params)

    if load_idx is not None:
        params, _, _ = load_chkpt(load_idx, chkpt_folder)
        opt_state = tx.init(params)

    return m, tx, opt_state, params, state


def check_for_nans(x):
    return jnp.any(
        jnp.array(tree_map(lambda x: jnp.any(jnp.isnan(x)), jax.tree_leaves(x)))
    )


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

    def loss(params, point_cloud_, latents_):
        batch_size = point_cloud_.shape[0]
        flow_key, dropout_key, sim_k = split(key, 3)
        latent_log_prob, z_sample = apply_func(
            {"params": params, **state},
            point_cloud_,
            latents_,
            rngs={"rsample_key": flow_key, "dropout": dropout_key},
        )

        # pc_hat, _ = vmap(simulate, in_axes=(0, 0, None, None))(
        #     split(sim_k, batch_size),
        #     z_sample,
        #     0.01,#OBSERVATION_NOISE, for the resimulation better to add very little noise, it's enough on the observation
        #     NUM_POINTS,
        # )

        # try to stablize training
        # latent_log_prob = jnp.where(latent_log_prob>-1e2, latent_log_prob, 0.0)
        # reconstruction_loss = -vmap(sum_gaussian_logpdf)(q, x_hat, jnp.abs(x_hat) * proportional_noise).mean()
        # reconstruction_loss = ((point_cloud - pc_hat) ** 2).mean(axis=(1,2))

        l = -latent_log_prob.mean() * (
            1 - beta
        )  # - beta * (latent_log_prob/jax.lax.stop_gradient(reconstruction_loss)).mean()
        return l, jnp.array(0.0)  # jax.lax.stop_gradient(reconstruction_loss.mean())

    # l_grad = jax.jit(jax.value_and_grad(loss,argnums=(0,)))
    # gs = []
    # ls = []
    # for i in range(point_cloud.shape[0]):
    #     l, g_i = l_grad(params,i,point_cloud[i:i+1],latents[i:i+1])
    #     gs.append(g_i)
    #     ls.append(l)
    #     if check_for_nans(g_i):
    #         loss(params,i, point_cloud[i:i+1],latents[i:i+1])

    (l, rc), grads = jax.value_and_grad(loss, has_aux=True)(
        params, point_cloud, latents
    )
    grads_nan = jnp.any(
        jnp.array(tree_map(lambda x: jnp.any(jnp.isnan(x)), jax.tree_leaves(grads)))
    )
    grads = jax.lax.cond(
        grads_nan, lambda: tree_map(lambda x: jnp.zeros_like(x), grads), lambda: grads
    )
    updates, opt_state = tx.update(
        # tree_map(lambda x: jnp.clip(x, -1.0, 1.0), grads),
        grads,
        opt_state,
        params,
    )
    params = optax.apply_updates(params, updates)
    norm = jnp.linalg.norm(jax.flatten_util.ravel_pytree(params)[0])
    return opt_state, params, l, rc, grads_nan, norm


no_grad_gen_model = lambda k, o, n: jax.tree_map(
    jax.lax.stop_gradient, generative_model(k, o, n)
)


if __name__ == "__main__":
    import wandb

    wandb.login()
    wandb.init(project="box_training")
    cfg = AttrDict()

    cfg.key = PRNGKey(639265)
    # data generation params
    cfg.OBSERVATION_NOISE = 0.25
    cfg.NUM_POINTS = 100
    cfg.num_epochs = 9999999
    cfg.batch_size = 6000
    cfg.generation_size = int(12e3)
    gen_ks = jit(
        lambda k: split(k, cfg.generation_size).reshape(2, cfg.generation_size // 2, 2)
    )

    ## reconstruction params
    beta = 0.0

    ## checkpoint params
    cfg.save_params = 2
    cfg.print_every = 1
    cfg.chkpt_folder = "box_chkpts_high_cap_3/"
    cfg.load_idx = 0
    wandb.config.update(cfg)

    ## optim config
    cfg.optim_cfg = OptimCfg(
        max_lr=5e-5,
        num_steps=int(10000),
        pct_start=0.01,
        div_factor=1e1,
        final_div_factor=1e0,
        weight_decay=0.0005,
        gradient_clipping=5.0,
    )
    wandb.config.update({'optim_cfg':cfg.optim_cfg.__dict__})

    m, tx, opt_state, params, state = initialize_model_and_state(
        key=PRNGKey(1574),
        obs_length=cfg.NUM_POINTS,
        optim_cfg=cfg.optim_cfg,
        num_input_vars=3,
        chkpt_folder=cfg.chkpt_folder,
        load_idx=cfg.load_idx,
    )

    wandb.config.update(
        {
            "flow_cfg": m.flow_config.__dict__,
            "transformer_cfg": m.transformer_cfg.__dict__,
        }
    )

    while True:
        finished = False

        save_idx = 0 if cfg.load_idx is None else cfg.load_idx + 1

        if cfg.load_idx:
            _, _, loaded_key = load_chkpt(cfg.load_idx, cfg.chkpt_folder)
        else:
            loaded_key = cfg.key

        losses = []
        rcs = []
        for e in range(cfg.num_epochs):
            old_key = cfg.key
            cfg.key = loaded_key if e == 0 else cfg.key
            cfg.key, subkey = split(cfg.key)
            point_cloud, latents, _ = tree_map(
                onp.array,
                pmap(
                    jit(
                        vmap(no_grad_gen_model, in_axes=(0, None, None)),
                        static_argnums=(1, 2),
                    ),
                    in_axes=(0, None, None),
                    static_broadcasted_argnums=(1, 2),
                )(gen_ks(cfg.key), cfg.OBSERVATION_NOISE, cfg.NUM_POINTS),
            )

            point_cloud, latents = tree_map(
                lambda x: onp.concatenate([x[0], x[1]]), [point_cloud, latents]
            )

            for i in range(cfg.generation_size // cfg.batch_size):
                cfg.key, subkey = split(cfg.key)
                # opt_state, params, l, rc, gn, norm = update_step(
                opt_state, params, l, rc, gn, norm = jit(
                    update_step, static_argnums=(0,)
                )(
                    m.apply,
                    jnp.array(
                        point_cloud[i * cfg.batch_size : (i + 1) * cfg.batch_size]
                    ),
                    jnp.array(latents[i * cfg.batch_size : (i + 1) * cfg.batch_size]),
                    opt_state,
                    params,
                    state,
                    cfg.key,
                )
                losses.append(jax.lax.stop_gradient(l))
                rcs.append(rc)
                if i % cfg.print_every == 0:
                    print(e, i, l)
                    wandb.log({"epoch": e, "i": i, "l": l, "norm": norm, "gn": gn})
                if jnp.isnan(l) or l == 0 or jnp.isinf(l):
                    print("failed", e, i, l, rc)
                    finished = True
                    break
                if gn:
                    print("######gradients fucked#######")
                    continue
                if (i + 1) % cfg.save_params == 0:
                    save_chkpt(save_idx, cfg.chkpt_folder, params, opt_state, old_key)
            if finished:
                print("current idx ---->", cfg.load_idx)
                break
            if e % 100 == 0:
                save_chkpt(
                    save_idx,
                    "box_chkpts_epochs/",
                    params,
                    opt_state,
                    old_key,
                    extra=f"ep_{e}_l_{l}",
                )
        cfg.load_idx += 1
