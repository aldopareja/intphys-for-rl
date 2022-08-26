import torch
torch.manual_seed(0)
import  utils as u
import main as m

ts = torch.tensor([0.0,1.0,2,3,4,5,8,10,15,20,30,40,60,70,80])
n_vars = 2
batch_size = 1000
d_model = 60
noise_std_ratio = 0.1
num_trasformer_layers = 2

obs_to_embed = u.MLP(input_size=n_vars,output_size=d_model,num_hidden_layers=1,num_hidden_units=d_model*2, unsqueeze=False)


obs_to_embed = m.SeqPosteriorTransformer(obs_embedder=obs_to_embed,
                                         d_model=d_model,
                                         num_transformer_layers=num_trasformer_layers,
                                         positional_encoding=True,
                                         decode_length=2)

inference_network = m.GaussianMixturePosterior(obs_to_embed= obs_to_embed,
                                               num_latents= n_vars**2).cuda()

optimizer = torch.optim.Adam(params=inference_network.parameters(), lr=0.00001)

gen_traces_fn = lambda: m.solve_ode_sample(batch_size, n_vars, [-10.0, 10.0], [-0.1, 0.1], noise_std_ratio, ts,
                                          device='cuda')

losses = m.amortize_inference(inference_network, gen_traces_fn, optimizer, 100)

theta, sample = m.solve_ode_sample(2, n_vars, [-10.0, 10.0], [-0.1, 0.1], noise_std_ratio, ts,
                                       device='cuda')

posterior = inference_network.get_q_x_given_seqembed(inference_network.obs_to_embed(sample[:1]))
posterior.sample(10)