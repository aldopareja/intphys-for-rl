import jax
from jax import numpy as jnp

from flax import linen as nn
from flax import struct

from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors


@struct.dataclass
class RealNVPConfig:
    f_hidden_size: int = 20
    f_num_layers: int = 3
    num_latent_vars: int = 10
    num_flow_layers: int = 5
    q_mlp_hidden_size: int = 200
    q_mlp_num_layers: int = 5
    out_min: float = 0.01
    out_max: float = 10.1
    stabilization_factor: float = 10.0


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
        return mask[None]

    def setup(self):
        cfg = self.config

        # self.s_func, self.t_func = [
        #     create_mlp(cfg.f_mlp_hidden_size, cfg.num_latent_vars, cfg.f_mlp_num_layers)
        #     for _ in range(2)
        # ]
        self.nn = GatedDenseNet(cfg)
        self.mask = self.init_mask(cfg.num_latent_vars, self.even)
        self.scaling_factor = self.param("scaling_factor",
                                         nn.initializers.constant(cfg.stabilization_factor),
                                         (1,cfg.num_latent_vars))
        
    def __call__(self,z,inverse=True):
        if inverse:
            return self.inverse(z)
        else:
            return self.forward(z)
        

    def forward(self, x):
        assert len(x.shape) == 2 and x.shape[-1] == self.config.num_latent_vars
        assert len(self.mask.shape) == 2 and self.mask.shape[-1] == x.shape[-1]
        x_mask = x * self.mask
        # s = self.s_func(x_mask)
        # t = self.t_func(x_mask)
        s,t = self.nn(x_mask).split(2,axis=-1)
        
        s_fac = jnp.exp(self.scaling_factor)
        s = nn.tanh(s / s_fac) * s_fac
        
        s = s * (1 - self.mask)
        t = t * (1 - self.mask)

        x = (x+t) * jnp.exp(s)

        log_det_jac = s.sum(axis=1)
        return x, log_det_jac

    def inverse(self, y):
        assert len(y.shape) == 2 and y.shape[-1] == self.config.num_latent_vars
        assert len(self.mask.shape) == 2 and self.mask.shape[-1] == y.shape[-1]
        y_mask = y * self.mask
        # s = self.s_func(y_mask)
        # t = self.t_func(y_mask)
        s,t = self.nn(y_mask).split(2,axis=-1)
        
        s_fac = jnp.exp(self.scaling_factor)
        s = nn.tanh(s / s_fac) * s_fac
        
        s = s * (1 - self.mask)
        t = t * (1 - self.mask)

        y = y * jnp.exp(-s)  - t

        inv_log_det_jac = -s.sum(axis=1)

        return y, inv_log_det_jac
        
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
        # self.scale_stabilizer = tfb.Scale(scale=cfg.stabilization_factor)

    def log_probability(self, x, mu, cov):
        assert x.shape == mu.shape and cov.shape == x.shape, f'{x},{mu}'

        log_prob = self.trunc.inverse_log_det_jacobian(x).sum(axis=1)
        # log_prob = jnp.zeros_like(x.shape[0])
        x = self.trunc.inverse(x)
        for l in reversed(self.layers):
            x, inv_log_det_jac = l(x,inverse=True)
            log_prob += inv_log_det_jac
            
            #use scaling to stabilize the output
            # log_prob += self.scale_stabilizer.inverse_log_det_jacobian(x).reshape(1,)
            # x = self.scale_stabilizer.inverse(x)

        log_prob += tfd.MultivariateNormalDiag(loc=mu, scale_diag=cov).log_prob(x)

        return log_prob

    def rsample(self, mu, cov, key):
        x = tfd.MultivariateNormalDiag(loc=mu, scale_diag=cov).sample(seed=key)

        for l in self.layers:
            x, _ = l(x,inverse=False)
            # x = self.scale_stabilizer.forward(x)

        # return x
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
    
class ConcatELU(nn.Module):
    def __call__(self, x):
        return jnp.concatenate([nn.elu(x), nn.elu(-x)], axis=-1)


class GatedDenseLayer(nn.Module):
    cfg: RealNVPConfig
    
    @nn.compact
    def __call__(self, x):
        out = nn.Sequential([
            ConcatELU(),
            nn.Dense(self.cfg.f_hidden_size),
            ConcatELU(),
            nn.Dense(2*self.cfg.f_hidden_size)
        ])(x)
        val, gate = out.split(2, axis=-1)
        return x + val*nn.sigmoid(gate)

class GatedDenseNet(nn.Module):
    cfg: RealNVPConfig
    
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.cfg.f_hidden_size)(x)
        for _ in range(self.cfg.f_num_layers - 1):
            x = GatedDenseLayer(self.cfg)(x)
            x = nn.LayerNorm()(x)
        x = ConcatELU()(x)
        x = nn.Dense(2*self.cfg.num_latent_vars)(x)
        return x
    
    
if __name__ == "__main__":
    m = GatedDenseLayer(RealNVPConfig())
    m.init(jax.random.PRNGKey(0),jnp.arange(20,dtype=jnp.float32).reshape(2,10))