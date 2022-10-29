import jax
from jax import numpy as jnp

from flax import linen as nn
from flax import struct

from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors


@struct.dataclass
class RealNVPConfig:
    f_mlp_hidden_size: int = 20
    f_mlp_num_layers: int = 3
    num_latent_vars: int = 2
    num_flow_layers: int = 5
    q_mlp_hidden_size: int = 200
    q_mlp_num_layers: int = 5
    out_min: float = 0.01
    out_max: float = 10.1


def create_mlp(hidden_size, output_size, num_layers, use_layer_norm: bool = False):
    l = []
    for _ in range(num_layers - 1):
        l.append(nn.Dense(hidden_size))
        l.append(nn.leaky_relu)
        l.append(nn.LayerNorm()) if use_layer_norm else None

    l.append(nn.Dense(output_size))
    return nn.Sequential(l)


class RealNVPLayer(nn.Module):
    config: RealNVPConfig
    even: bool

    @staticmethod
    def init_mask(num_vars, even):
        d = int(num_vars / 2)
        mask = jax.lax.cond(
            even,
            lambda: jnp.concatenate([jnp.ones((d,)), jnp.zeros((num_vars - d,))]),
            lambda: jnp.concatenate([jnp.zeros((d,)), jnp.ones((num_vars - d,))]),
        )
        return mask

    def setup(self):
        cfg = self.config

        self.s_func, self.t_func = [
            create_mlp(cfg.f_mlp_hidden_size, cfg.num_latent_vars, cfg.f_mlp_num_layers)
            for _ in range(2)
        ]
        self.scale = self.param(
            "scale",
            nn.initializers.uniform(),
            (1, cfg.num_latent_vars),
        )
        self.mask = self.init_mask(cfg.num_latent_vars, self.even)

    def forward(self, x):
        assert len(x.shape) == 2 and x.shape[-1] == self.config.num_latent_vars
        x_mask = x * self.mask
        s = self.s_func(x_mask) * self.scale
        t = self.t_func(x_mask)

        y = x_mask + (1 - self.mask) * (x * jnp.exp(s) + t)

        log_det_jac = ((1 - self.mask) * s).sum(axis=1)
        return y, log_det_jac

    def inverse(self, y):
        y_mask = y * self.mask
        s = self.s_func(y_mask) * self.scale
        t = self.t_func(y_mask)

        x = y_mask + (1 - self.mask) * (y - t) * jnp.exp(-s)

        inv_log_det_jac = ((1 - self.mask) * -s).sum(axis=1)

        return x, inv_log_det_jac


class RealNVP_trunc(nn.Module):
    config: RealNVPConfig

    def setup(self):
        cfg = self.config

        self.layers = [
            RealNVPLayer(cfg, i % 2 == 0) for i in range(cfg.num_flow_layers)
        ]
        self.trunc = tfb.Chain(
            [
                tfb.Shift(shift=cfg.out_min),
                tfb.Scale(scale=cfg.out_max - cfg.out_min),
                tfb.Sigmoid(),
                tfb.Scale(scale=1.0),  # added to make the truncation smoother
            ]
        )

    def log_probability(self, x, mu, cov):
        assert x.shape == mu.shape and cov.shape == x.shape, f'{x},{mu}'

        log_prob = self.trunc.inverse_log_det_jacobian(x).sum(axis=1)
        x = self.trunc.inverse(x)
        for l in reversed(self.layers):
            x, inv_log_det_jac = l.inverse(x)
            log_prob += inv_log_det_jac

        log_prob += tfd.MultivariateNormalDiag(loc=mu, scale_diag=cov).log_prob(x)

        return log_prob

    def rsample(self, mu, cov, key):
        x = tfd.MultivariateNormalDiag(loc=mu, scale_diag=cov).sample(seed=key)

        for l in self.layers:
            x, _ = l.forward(x)

        return self.trunc.forward(x)


class ObstoQ(nn.Module):
    config: RealNVPConfig

    @nn.compact
    def __call__(self, obs):
        c = self.config
        mlp = create_mlp(
            c.q_mlp_hidden_size, c.num_latent_vars*2, c.q_mlp_num_layers, use_layer_norm=True
        )
        q = mlp(obs)
        mu = q[:,:c.num_latent_vars]
        cov = jnp.exp(q[:,c.num_latent_vars:])
        return mu, cov
