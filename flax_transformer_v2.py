from typing import Sequence
from dataclasses import field

import numpy as onp

import jax
from jax import numpy as jnp
from jax.random import split, PRNGKey

from flax import struct
from flax import linen as nn


@struct.dataclass
class TransformerConfig:
    """Global hyperparameters used to minimize obnoxious kwarg plumbing."""

    num_heads: int = 4
    num_enc_layers: int = 1
    num_dec_layers: int = 1
    dropout_rate: float = 0.1
    deterministic: bool = False
    d_model: int = 100
    add_positional_encoding = False
    max_len: int = 3000  # positional encoding
    obs_emb_hidden_sizes: Sequence[int] = (200,)
    num_mixtures: int = 2


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
    # todo: shall we add dropout? there's documentation to read though.

    @staticmethod
    def init_pe(d_model: int, max_length: int):
        positions = jnp.arange(max_length)[:, None]
        div_term = jnp.exp(jnp.arange(0, d_model) * (-jnp.log(10000.0) / d_model))

        temp = positions * div_term
        even_mask = positions % 2 == 0

        pe = jnp.where(even_mask, jnp.sin(temp), jnp.cos(temp))

        return pe[None, :, :]

    @nn.compact
    def __call__(self, x):
        cfg = self.config
        pe = self.variable(
            "consts", "pe", PositionalEncoder.init_pe, cfg.d_model, cfg.max_len
        )
        # batch_apply_pe = nn.vmap(lambda x, pe: x + pe[:x.shape[0]], in_axes=(0,None))
        return x + pe.value[:, : x.shape[1]]


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
        x = nn.Dense(cfg.d_model * 2)(inputs)
        x = nn.relu(x)
        x = nn.Dropout(rate=cfg.dropout_rate)(x, deterministic=cfg.deterministic)
        output = nn.Dense(cfg.d_model)(x)
        output = nn.Dropout(rate=cfg.dropout_rate)(
            output, deterministic=cfg.deterministic
        )
        return output


class EncoderLayer(nn.Module):
    """Transformer encoder layer.
    Args:
      config: TransformerConfig dataclass containing hyperparameters.
    """

    config: TransformerConfig

    @nn.compact
    def __call__(self, inputs):
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
            use_bias=False,  # should we use bias? I guess it doesn't matter
            broadcast_dropout=False,
            dropout_rate=cfg.dropout_rate,
            deterministic=cfg.deterministic,
            decode=False,
        )(x)
        x = nn.Dropout(rate=cfg.dropout_rate)(x, deterministic=cfg.deterministic)
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
            decode=False,
        )(x)
        x = nn.Dropout(rate=cfg.dropout_rate)(x, deterministic=cfg.deterministic)
        x = x + output_emb

        z = nn.LayerNorm()(x)

        x = nn.MultiHeadDotProductAttention(
            num_heads=cfg.num_heads,
            use_bias=False,
            broadcast_dropout=False,
            dropout_rate=cfg.dropout_rate,
            deterministic=cfg.deterministic,
            decode=False,
        )(z, encoded_input)

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
        enc_input = PositionalEncoder(cfg)(x) if cfg.add_positional_encoding else x

        # for _ in range(cfg.num_enc_layers):
        enc_input = nn.Sequential(
            [EncoderLayer(cfg) for _ in range(cfg.num_enc_layers)]
        )(enc_input)

        start = self.param("start", nn.initializers.uniform(), (1, 1, cfg.d_model))
        so_far_dec = start.repeat(q.shape[0], axis=0)

        seq_decoded = []
        for _ in range(cfg.num_mixtures):
            dec = so_far_dec
            # dec = decoder(dec, enc_input)
            for _ in range(cfg.num_dec_layers):
                dec = DecoderLayer(cfg)(dec, enc_input)
            so_far_dec = jnp.concatenate(
                [so_far_dec, jax.lax.stop_gradient(dec[:, -1:])], axis=1
            )
            seq_decoded.append(dec[:, -1:, :])
        seq_decoded = jnp.concatenate(seq_decoded, axis=1)

        return seq_decoded


@struct.dataclass
class GaussianMixturePosteriorConfig:
    num_latents: int = 2
    emb_size: int = 100
    covariance_eps: float = 1e-6


class TransformerGaussianMixturePosterior(nn.Module):
    config: GaussianMixturePosteriorConfig

    @nn.compact
    def __call__(self, seq_decoded):
        """takes the output of a transformer decoder and outputs the parameters of a multivariate gaussian mixture"""
        cfg = self.config
        num_means = cfg.num_latents
        num_covariance_terms = int(num_means * (num_means + 1) / 2)
        num_mixture_params = 1 + num_means + num_covariance_terms

        dist_params = nn.Sequential(
            [nn.Dense(cfg.emb_size), nn.relu, nn.Dense(num_mixture_params)]
        )(seq_decoded)

        mix_p = jax.nn.softmax(dist_params[:, :, 0])
        means = dist_params[:, :, 1 : 1 + num_means]
        covariance_terms = dist_params[:, :, -num_covariance_terms:]
        covariance_matrices = self.get_cov_matrices_from_vectors(
            covariance_terms, num_means, cfg.covariance_eps
        )

        return dict(
            mix_p=mix_p,
            means=means,
            covariance_matrices=covariance_matrices,
        )

    @staticmethod
    def get_cov_matrices_from_vectors(covariance_terms, num_means, eps):
        x = covariance_terms
        output_shape = (*x.shape[:-1], num_means, num_means)
        x = jnp.concatenate([x, x[:, :, num_means:][:, :, ::-1]], axis=-1)
        x = x.reshape(output_shape)
        x = jnp.triu(x)

        eps = jnp.eye(num_means)[None, None] * eps
        cov_matrices = jnp.matmul(x, x.swapaxes(-2, -1)) + eps
        return cov_matrices


def gaussian_mixture_logpdf(latents, dist_params):
    """
    Parameters:
    --------------
    latents: jax.array with shape Batch x 1 x num_latents
      the single dimension is needed so that the same sample gets broadcasted through all mixtures
    dist_params: Dict(
            mix_p: jax.array with shape Batch x num_mixtures encoding the probability of each mixture
            means: jax.array with shape Batch x num_mixtures x num_means with the multivariate mean of each mixture
            cov_matrices: jax.array with shape Batch x num_mixtures x num_means x num_means
            )
    """
    mix_p, means, covs = (
        dist_params["mix_p"],
        dist_params["means"],
        dist_params["covariance_matrices"],
    )

    normals_log_prob = jax.scipy.stats.multivariate_normal.logpdf(latents, means, covs)
    category_log_prob = jax.lax.log(mix_p)

    return jax.nn.logsumexp(normals_log_prob + category_log_prob, axis=1)

def select_mixture_sample(all_mixtures, index):
    '''takes a single sample from a mixture density num_mixtures x num_variables and returns the value given by the :index:
    mixture num_variables
    '''
    return all_mixtures[index,:]

'''by vectorizing over the batch size, we can pick from each batch, a corresponding index.
essentially a fast version of:

    onp.concatenate([batched_all_mixtures[i,idx] for i,idx in enumerate(batched_index)])
'''
v_select_mixture_sample = jax.vmap(select_mixture_sample, in_axes=(0, 0))

def gaussian_mixture_sample(key, dist_params):
    mix_p, means, covs = (
        dist_params["mix_p"],
        dist_params["means"],
        dist_params["covariance_matrices"],
    )
    key, subkey = split(key)
    n_sample = jax.random.multivariate_normal(key, means, covs)
    cat_sample = jax.random.categorical(subkey, jax.lax.log(mix_p), axis=-1)
    
    chosen_sample = v_select_mixture_sample(n_sample, cat_sample)
    return chosen_sample


@struct.dataclass
class IndependentGaussianMixtureConfig:
    # sequence with the number of variables on each group (each group modeled with a gaussian mixture)
    group_variables: Sequence[int] = (3, 4)
    # number of mixtures used on each group
    num_mixtures_per_group: Sequence[int] = (2, 8)


class IndependentGaussianMixtures(nn.Module):
    config: IndependentGaussianMixtureConfig
    mixtures_cfg: GaussianMixturePosteriorConfig
    transformer_cfg: TransformerConfig

    @nn.compact
    def __call__(self, q):
        """
        Parameters:
          q: array with shape B x num_obs x seq_length
            contains a batch of sequences of observable variables. In the case of a point cloud
            this contains Batch size x 3 x num_points (since each point is x,y,z).
        returns:
          list_dist_params: list of dicts, len(list_dist_params) = len(group_varibles)
            list of dicts where each dict contains batched parameters of a multivariate gaussian mixture.
            see :meth:`gaussian_mixture_logpdf`
        """
        cfg = self.config
        transformer_cfg = self.transformer_cfg.replace(
            num_mixtures=sum(cfg.num_mixtures_per_group)
        )
        mixtures_cfg = self.mixtures_cfg

        seq_decoded = TransformerStack(transformer_cfg)(q)

        v = onp.cumsum(onp.array((0, *cfg.num_mixtures_per_group)))
        seq_dec_list = [seq_decoded[:, v[i] : v[i + 1]] for i in range(len(v) - 1)]
        
        dist_params_list = []
        for seq_dec, n_vars in zip(seq_dec_list, cfg.group_variables):
            d_pars = TransformerGaussianMixturePosterior(
                mixtures_cfg.replace(num_latents=n_vars)
            )(seq_dec)
            dist_params_list.append(d_pars)

        return dist_params_list


if __name__ == "__main__":
    m = IndependentGaussianMixtures(
        IndependentGaussianMixtureConfig(),
        GaussianMixturePosteriorConfig(),
        TransformerConfig(),
    )
    key, *subkeys = split(PRNGKey(11234), 4)
    init_rngs = {"params": subkeys[0], "dropout": subkeys[1]}
    variables = m.init(init_rngs, jnp.ones((batch_size, obs_lenght, 1)))
    m.init(init_rngs, jnp.ones((100, 6, 999)))
