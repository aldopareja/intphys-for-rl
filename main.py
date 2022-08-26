'''
tools used to compute the posterior over the parameters of a state-space model given some trajectory
'''
import math

import torch
from torch import nn
from torchdiffeq import odeint

import utils as u
from utils import MLP, NormalMixture1D


def dy_dt(y, theta):
    dy_dt = torch.einsum('bkj,bj->bk', theta,y)
    return dy_dt


def solve_ode_sample(batch_size, n_vars, y_init_range, theta_range, noise_std_ratio, ts, device='cpu'):
    y_init = torch.distributions.Uniform(*torch.tensor(y_init_range, device=device)).rsample((batch_size, n_vars))
    theta_dist = torch.distributions.Uniform(*torch.tensor(theta_range, device=device))
    theta = theta_dist.rsample((batch_size, n_vars, n_vars))
    ts = ts.to(device)

    y_mean = odeint(lambda t, y: dy_dt(y, theta=theta), y_init, ts, rtol=1e-6, atol=1e-5)
    y_obs = torch.distributions.Normal(loc=y_mean, scale=noise_std_ratio * torch.abs(y_mean) + 1e-5).rsample()
    y_obs = y_obs.swapaxes(0, 1)

    return theta, y_obs


class BatchedNormalMixture1D():
    def __init__(self, mixture_probs, means, stds):
        self.num_mixtures = mixture_probs.shape[-1]
        self.normals = torch.distributions.Normal(loc=means, scale=stds)
        self.mixture_probs = mixture_probs

    def log_prob(self, batched_obs):
        batched_obs = batched_obs.reshape(-1, 1).expand(-1, self.num_mixtures)
        normal_log_prob = self.normals.log_prob(batched_obs)
        category_log_prob = torch.log(self.mixture_probs)
        return torch.logsumexp(normal_log_prob + category_log_prob, dim=1)


class BatchedMultivariateNormalMixture():
    def __init__(self, mixture_probs, means, covariance_matrices):
        '''
        :param mixture_probs: Bxnum_mixtures should sum up to one on the num_mixtures direction
        :param means: B x num_mixtures x num_variables
        :param covariance_matrices: B x num_mixtures x num_variables x num_variables, each should be positive definite

        builds several multivariate normals and uses mixture probs to mix them.
        '''
        self.multivariatenormals = torch.distributions.MultivariateNormal(loc=means,
                                                                          covariance_matrix=covariance_matrices)
        self.mixture_probs = mixture_probs

    def log_prob(self, batched_x):
        '''
        :param batched_obs: B x 1 x num_vars or 1 x 1 x num_vars (which would broadcast over the batch)
        :return: B x log_prob log probability of the given x for each of the batched multivariate gaussian mixtures.
        '''
        normal_log_prob = self.multivariatenormals.log_prob(batched_x)
        category_log_prob = torch.log(self.mixture_probs)

        return torch.logsumexp(normal_log_prob + category_log_prob, dim=1)

    def sample(self, num_samples):
        sample = self.multivariatenormals.sample((num_samples,))
        cat = torch.distributions.Categorical(self.mixture_probs)
        cat_sample = cat.sample((num_samples,)).squeeze()
        return sample[torch.arange(sample.shape[0]),
                      torch.arange(sample.shape[1]),
                      cat_sample]


class SeqGaussMixPosterior(nn.Module):
    def __init__(self, num_mixtures):
        super().__init__()
        self.embed_obs = MLP(2, 10, output_size=10, unsqueeze=False)
        self.decode_mu = MLP(1, 12, input_size=10,
                             last_activation=lambda y: torch.tanh(y) * 10,
                             unsqueeze=False)
        self.decode_sigma = MLP(1, 12, input_size=10,
                                last_activation=torch.exp,
                                unsqueeze=False)
        self.decode_mixture_prob = MLP(1, 12, input_size=10,
                                       last_activation=torch.exp,
                                       unsqueeze=False)
        self.trans_dec = nn.TransformerDecoderLayer(d_model=10, nhead=2, dim_feedforward=20,
                                                    batch_first=True, dropout=0.0)

        self.start = torch.nn.Parameter(torch.rand(1, 1, 10))
        self.num_mixtures = num_mixtures

    def get_q_x_given_obs(self, obs):
        obs_embed = self.embed_obs(obs)
        so_far_decoded = self.start.expand(obs_embed.shape)
        all_mu, all_sigma, all_mix_p = [], [], []
        for _ in range(self.num_mixtures):
            obs_dec = self.trans_dec(so_far_decoded, obs_embed)
            so_far_decoded = torch.cat([so_far_decoded, obs_dec[:, -1:, :]], dim=-2).detach()
            all_mu.append(self.decode_mu(obs_dec[:, -1:, :]))
            all_sigma.append(self.decode_sigma(obs_dec[:, -1:, :]))
            all_mix_p.append(self.decode_mixture_prob(obs_dec[:, -1:, :]))
        mixture_probs = torch.cat(all_mix_p, dim=1).reshape(len(obs), self.num_mixtures)
        mu = torch.cat(all_mu, dim=1).reshape(len(obs), self.num_mixtures)
        sigma = torch.cat(all_sigma, dim=1).reshape(len(obs), self.num_mixtures)
        return BatchedNormalMixture1D(mixture_probs / mixture_probs.sum(dim=1).unsqueeze(-1), mu, sigma)

    def forward(self, theta, obs):
        q_x_given_obs = self.get_q_x_given_obs(obs)
        log_prob = q_x_given_obs.log_prob(theta)
        return -torch.mean(log_prob)


class SeqGaussMixPosteriorV2(nn.Module):
    def __init__(self, num_mixtures, d_model, num_obs_vars, num_trans_layers):
        super().__init__()
        self.embed_obs = MLP(input_size=num_obs_vars, output_size=d_model, num_hidden_units=d_model * 2,
                             num_hidden_layers=2, unsqueeze=False)
        self.pos_enc = u.PositionalEncoding
        self.trans_enc = nn.Sequential(*[nn.TransformerEncoderLayer(d_model=d_model, nhead=d_model // 10,
                                                                    dim_feedforward=d_model * 2,
                                                                    dropout=0.0, batch_first=True)
                                         for _ in range(num_trans_layers)])
        self.trans_dec = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=d_model, nhead=d_model // 10,
                                                                          dim_feedforward=d_model * 2,
                                                                          dropout=0.0, batch_first=True),
                                               num_layers=num_trans_layers)

        self.decode_mu = MLP(input_size=d_model, num_hidden_layers=1, num_hidden_units=d_model * 2,
                             unsqueeze=False)
        self.decode_sigma = MLP(input_size=d_model, num_hidden_layers=1, num_hidden_units=d_model * 2,
                                last_activation=torch.exp,
                                unsqueeze=False)
        self.decode_mixture_prob = MLP(input_size=d_model, num_hidden_layers=1, num_hidden_units=d_model * 2,
                                       last_activation=torch.exp,
                                       unsqueeze=False)

        self.start = torch.nn.Parameter(torch.rand(1, 1, d_model))
        self.num_mixtures = num_mixtures
        self.d_model = d_model

    def get_q_x_given_obs(self, obs):
        '''

        :param obs: shape: Batch, Seq, 1 (single feature)
        :return:
        '''
        memory = self.embed_obs(obs)
        memory = self.trans_enc(memory)
        so_far_decoded = self.start.expand(obs.shape[0], 1, self.d_model)
        all_mu, all_sigma, all_mix_p = [], [], []
        for _ in range(self.num_mixtures):
            dec = self.trans_dec(so_far_decoded, memory)
            so_far_decoded = torch.cat([so_far_decoded, dec[:, -1:, :]], dim=-2).detach()
            all_mu.append(self.decode_mu(dec[:, -1:, :]))
            all_sigma.append(self.decode_sigma(dec[:, -1:, :]))
            all_mix_p.append(self.decode_mixture_prob(dec[:, -1:, :]))
        mixture_probs = torch.cat(all_mix_p, dim=1).reshape(len(obs), self.num_mixtures)
        mu = torch.cat(all_mu, dim=1).reshape(len(obs), self.num_mixtures)
        sigma = torch.cat(all_sigma, dim=1).reshape(len(obs), self.num_mixtures)
        return BatchedNormalMixture1D(mixture_probs / mixture_probs.sum(dim=1).unsqueeze(-1), mu, sigma)

    def forward(self, theta, obs):
        q_x_given_obs = self.get_q_x_given_obs(obs)
        log_prob = q_x_given_obs.log_prob(theta)
        return -torch.mean(log_prob)


class SeqPosteriorTransformer(nn.Module):
    '''
    Implements a transformer that takes a batch of observed sequences and return a list of batch x output_seq_length x d_model embeddings
    to do so a usual transformer architecture is used.
    '''

    def __init__(self, obs_embedder: nn.Module, d_model: int, num_transformer_layers: int, positional_encoding=True,
                 decode_length=2):
        '''
        :param obs_embedder: must be able to process an observation batch B x input_length x num_features into B x input_length x d_model
        :param d_model: d_model  is the width of the transformer as explained in the "attention is all you need" paper
        :param num_transformer_layers: number of layers for both the encoder and the decoder
        :param positional_encoding: if true, adds positional encoding to the encoder to account for 'time' in the input space
                                    not supporting positional encoding at the output since the main usage of this module is a permutation
                                    invariant gaussian mixture decoded from the decoded sequence
        :param decode_length: number of decoding steps done, in our case this is equal to the number of mixtures.
        '''
        super().__init__()
        self.embed_obs = obs_embedder

        self.pos_enc = u.PositionalEncoding(d_model) if positional_encoding else nn.Identity()

        self.start = torch.nn.Parameter(torch.rand(1, 1, d_model))
        self.d_model = d_model

        self.trans_enc = nn.Sequential(*[nn.TransformerEncoderLayer(d_model=d_model, nhead=d_model // 10,
                                                                    dim_feedforward=d_model * 2,
                                                                    dropout=0.1, batch_first=True)
                                         for _ in range(num_transformer_layers)])
        self.trans_dec = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=d_model, nhead=d_model // 10,
                                                                          dim_feedforward=d_model * 2,
                                                                          dropout=0.1, batch_first=True),
                                               num_layers=num_transformer_layers)

        self.decode_length = decode_length

    def forward(self, obs):
        '''
        :param obs: batchsize x sequence_length x input_features
        :return: batchsize x decode_length x d_model
        '''
        memory = self.embed_obs(obs)
        memory = self.pos_enc(memory)
        memory = self.trans_enc(memory)
        so_far_decoded = self.start.expand(obs.shape[0], 1, self.d_model)
        seq_decoded = []
        for _ in range(self.decode_length):
            dec = self.trans_dec(so_far_decoded, memory)
            so_far_decoded = torch.cat([so_far_decoded, dec[:, -1:, :]], dim=-2).detach()
            seq_decoded.append(dec[:, -1:, :])
        seq_decoded = torch.cat(seq_decoded, dim=-2)
        return seq_decoded


class GaussianMixturePosterior(nn.Module):
    def __init__(self, obs_to_embed: SeqPosteriorTransformer, num_latents):
        super().__init__()
        self.obs_to_embed = obs_to_embed
        self.num_means = int(num_latents)
        self.num_covariance_terms = int(num_latents * (num_latents + 1) / 2)
        output_size = int(1 + self.num_means + self.num_covariance_terms)  # need a dimension for the mixture

        self.dist_decoder = MLP(input_size=obs_to_embed.d_model,
                                output_size=output_size,
                                num_hidden_units=obs_to_embed.d_model * 2,
                                num_hidden_layers=2,
                                unsqueeze=False)
        self.num_latents = int(num_latents)

    def get_q_x_given_seqembed(self, seqembed):
        '''
        :param seqembed: B x num_mixtures x d_model sequential generation of mixture embeddings.
        :return:
        '''
        output = self.dist_decoder(seqembed)
        mix_p = torch.exp(output[:, :, 0])
        means = output[:, :, 1:1 + self.num_means]
        covariance_terms = output[:, :, 1 + self.num_means:]
        covariance_matrices = self.get_covariance_matrices_from_vectors(covariance_terms,
                                                                        device=seqembed.device)

        return BatchedMultivariateNormalMixture(mixture_probs=mix_p / mix_p.sum(dim=1, keepdim=True),
                                                means=means,
                                                covariance_matrices=covariance_matrices)
    #
    @staticmethod
    def get_covariance_matrices_from_vectors(covariance_terms, device, eps=0.000025):
        '''
        :param covariance_terms: B x num_mixtures x n(n+1)/2 terms on the lower triangular matrices used to compute the covariance matrices
        :param device:
        :param eps:
        :return:
        '''
        batch_size = covariance_terms.shape[0]
        num_mix = covariance_terms.shape[1]
        mat_size = [int(math.sqrt(covariance_terms.shape[-1] * 2))] * 2
        cov_matrices = torch.zeros(batch_size, num_mix, *mat_size, device=device)
        idx1, idx2 = torch.tril_indices(*mat_size)
        cov_matrices[:, :, idx1, idx2] = covariance_terms
        eps = torch.eye(mat_size[0], device=device).expand(cov_matrices.shape) * eps

        # positive definite covariance matrices L*L' + eps * I
        cov_matrices = cov_matrices.matmul(cov_matrices.transpose(-1, -2)) + eps
        return cov_matrices

    def forward(self, theta, obs):
        '''
        :param theta: B x num_vars or 1 x num_vars (which would broadcast), theta will be reshaped into -1, num_latents
        :param obs: B x seq_length x features (or num_vars for observed variables) obs_to_embed must take this same shape
        :return: negative log likelihood of theta given the inference network's posteriors approximation. Keeps gradients
                 to make this trainable.
        '''
        batch_size = obs.shape[0]
        seqembed = self.obs_to_embed(obs)
        q_x_given_seqembed = self.get_q_x_given_seqembed(seqembed)
        log_prob = q_x_given_seqembed.log_prob(theta.reshape(batch_size, 1, self.num_latents))
        return -torch.mean(log_prob)


def amortize_inference(inference_network, gen_traces_fn, optimizer, num_iterations):
    loss_history = []
    for i in range(num_iterations):
        x, obs = gen_traces_fn()
        loss = inference_network(x, obs)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_history.append(loss.item())

    return loss_history
