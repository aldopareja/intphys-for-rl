import torch
from torch import nn
import math

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 10000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[0, :x.size(1)]
        return self.dropout(x)



class MLP(nn.Module):
    def __init__(self, num_hidden_layers, num_hidden_units, input_size=1,
                 output_size=1, activation=nn.ReLU(), last_activation=nn.Identity(), unsqueeze = True):

        super().__init__()
        layers = [nn.Sequential(nn.Linear(input_size,num_hidden_units),activation)]
        layers += [nn.Sequential(nn.Linear(num_hidden_units, num_hidden_units), activation)
                   for _ in range(num_hidden_layers - 1)]
        layers += [nn.Sequential(nn.Linear(num_hidden_units, output_size))]
        self.network = nn.Sequential(*layers)
        self.last_activation = last_activation
        self.unsqueeze = unsqueeze

    def forward(self,obs):
        if self.unsqueeze:
            emb = self.network(obs.unsqueeze(-1)).squeeze()
        else:
            emb = self.network(obs)
        return self.last_activation(emb)

class NormalMixture1D(torch.distributions.Distribution):
    def __init__(self, mixture_probs, means, stds):
        self.num_mixtures = len(mixture_probs)
        self.mixture_probs = torch.tensor(mixture_probs, dtype=torch.float32) if isinstance(mixture_probs, list) else mixture_probs
        self.normals = torch.distributions.Normal(
            loc=torch.tensor(means, dtype=torch.float32) if isinstance(means, list) else means,
            scale=torch.tensor(stds, dtype=torch.float32) if isinstance(stds, list) else stds,
        )

    def sample(self, n) -> torch.Tensor:
        mix_idx = torch.multinomial(self.mixture_probs, n, True).unsqueeze(-1)
        normals = self.normals.sample((n,))
        return torch.gather(normals, 1, mix_idx).squeeze()

    def log_prob(self, value):
        value_expanded = value.unsqueeze(-1).expand(-1, self.num_mixtures)
        normal_log_prob = self.normals.log_prob(value_expanded)
        category_log_prob = torch.log(self.mixture_probs)
        return torch.logsumexp(normal_log_prob + category_log_prob, dim=1)

def print_grads(model):
    for p in model.parameters():
        print('grads')
        print(p.grad.max(), p.grad.min(), p.grad.mean())
        print('abs')
        print(p.max(), p.min(), p.mean())
