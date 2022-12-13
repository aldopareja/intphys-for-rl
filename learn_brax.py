from json import load
import pickle
from copy import deepcopy
from stringprep import in_table_a1
from typing import Dict

import numpy as onp

import jax
from jax.random import split, PRNGKey, uniform
from jax import numpy as jnp
from jax.config import config

import optax

import brax
from brax.io.image import _scene
from brax.math import relative_quat

from pytinyrenderer import TinyRenderCamera as Camera
from pytinyrenderer import TinyRenderLight as Light
from google.protobuf import text_format

from istar_map_tqdm_patch import array_apply

from flax_transformer_v2 import (
    IndependentGaussianMixtures,
    IndependentGaussianMixtureConfig,
    GaussianMixturePosteriorConfig,
    TransformerConfig,
    gaussian_mixture_logpdf,
    gaussian_mixture_sample
)

_DEFAULT_CAMERA = {
    "viewWidth": 640,
    "viewHeight": 480,
    "position": [1.5, -3.0, 1.625],
    "target": [0.0, 0.0, 0],
    "up": [0, 0, 1],
    "hfov": 58.0,
    "vfov": 43.5,
}
_DEFAULT_LIGHT = {
    "direction": [0.57735, -0.57735, 0.57735],
    "ambient": 0.8,
    "diffuse": 0.8,
    "specular": 0.6,
    "shadowmap_center": [0.0, 0.0, 0],
}
_DEFAULT_CONFIG = """bodies {
  name: "box"
  colliders {
    box {
      halfsize {
        x: 0.75
        y: 0.25
        z: 0.125
      }
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
}
dt: 0.05000000074505806
substeps: 20
dynamics_mode: "pbd"
"""
default_env = text_format.Parse(_DEFAULT_CONFIG, brax.Config())


def get_default_camera(width, height):
    cam = deepcopy(_DEFAULT_CAMERA)
    cam["vfov"] = cam["hfov"] * height / width
    cam["viewWidth"] = width * 1
    cam["viewHeight"] = height * 1
    cam["position"] = [c * 2 for c in cam["position"]]
    return Camera(**cam)


def get_default_light():
    return Light(**_DEFAULT_LIGHT)


def render_depth_array(env_cfg, qp_dict, height=320, width=320, *_):
    sys = brax.System(env_cfg)
    my_qp = brax.QP(**qp_dict)
    scene, instances = _scene(sys, my_qp)
    camera = get_default_camera(width=width, height=height)
    x = scene.get_camera_image(instances, get_default_light(), camera)
    x = onp.array(x.depthbuffer)
    x = onp.reshape(x, (camera.view_height, camera.view_width))
    return x


def get_point_cloud_from_depth(depth_map: onp.ndarray, sample_size, normalize: Dict[float,float]={'x':320,'y':320}):
    """gets a point cloud of `sample_size` numbers from `depth_map`
    It first ignores the far plane, then computes x,y,z as the x,y pixel coordinates and
    sets z as the depth value in `depth_array`.

    Parameters:
    -------------
        depth_map: onp.ndarray
            Height x Width array with the depth values of each pixel
        sample_size: number of samples from the given point cloud. This get sampled from the 
            foreground depth map with replacement.
        normalize:

    Returns:
    -------------
        point_cloud: onp.ndarray
            `sample_size` x 3 array with each row having the x,y,z coordinates of each
            point. x and y are the pixel values, whereas z is the distance from the camera.
    """
    non_far_mask = depth_map > -1e4
    x, y = non_far_mask.nonzero()
    z = depth_map[x, y]
    if normalize['x'] != 1.0 or normalize['y'] != 1.0:
        z = 1/(-1-z)
    point_clouds = onp.stack([x/normalize['x'], y/normalize['y'], z], axis=1)
    idx = onp.random.randint(0, point_clouds.shape[0], sample_size)
    return point_clouds[idx]


def render_point_cloud(env_cfg, qp_dict, height=320, width=320, sample_size=1000):
    depth = render_depth_array(env_cfg, qp_dict, height, width)
    point_cloud = get_point_cloud_from_depth(depth, sample_size)
    return point_cloud


def sample_box_rotation(rng: PRNGKey, angle=None, img=None):
    # sample imaginary part of quaternion (axis of rotation)
    key, *subkeys = split(rng, 4)
    img = (
        uniform(subkeys[0], minval=-1.0, maxval=1.0, shape=(3,)) + 1e-6
        if img is None
        else img
    )
    img = img / jnp.linalg.norm(img)

    # sample angle (how much to rotate)
    angle = (
        uniform(subkeys[1], minval=-2 * jnp.pi, maxval=2 * jnp.pi)
        if angle is None
        else angle
    )
    w = jnp.cos(angle / 2)[None]
    img = jnp.sin(angle / 2) * img
    rot = jnp.concatenate([w, img])
    return rot


def sample_qp(key: PRNGKey):
    key, subkey = split(key)
    # position of each body in 3d (z is up, right-hand coordinates)
    pos = uniform(key, minval=-2.0, maxval=2.0, shape=(3,))[None]
    # velocity of each body in 3d
    vel = jnp.array([[0.0, 0.0, 0.0]])
    # rotation about center of body, as a quaternion (w, x, y, z)
    rot = sample_box_rotation(subkey)[None]
    # angular velocity about center of body in 3d
    ang = jnp.array([[0.0, 0.0, 0.0]])
    return dict(pos=pos, rot=rot, vel=vel, ang=ang)


vj_sample_qp = jax.jit(jax.vmap(sample_qp))


@jax.jit
def compute_diffs(batch_qp):
    init = jax.tree_map(lambda x: x[0 : len(x) : 2], batch_qp)
    final = jax.tree_map(lambda x: x[1 : len(x) : 2], batch_qp)
    pos_diff = (final["pos"] - init["pos"])[:, 0]  # remove unnecessary object dimension
    rot_diff = jax.vmap(relative_quat, in_axes=(0, 0))(
        init["rot"][:, 0], final["rot"][:, 0]
    )
    return dict(pos_diff=pos_diff[:,None], rot_diff=rot_diff[:,None])

def build_render_args(batch_qp, height, width, num_points):
    args = [
        (default_env, jax.tree_map(lambda x: x[i], batch_qp), height, width, num_points)
        for i in range(batch_qp['pos'].shape[0])
    ]
    return args


def generate_data_batch(
    key: PRNGKey, batch_size, height=320, width=320, num_points=999
):
    batch_qp = vj_sample_qp(split(key, batch_size * 2))
    diffs = compute_diffs(batch_qp)
    batch_qp, diffs = jax.tree_map(lambda x: onp.array(x), [batch_qp, diffs])

    args = build_render_args(batch_qp, height, width, num_points)
    point_clouds = array_apply(render_point_cloud, args, True, chunksize=1000)
    point_clouds = onp.stack(point_clouds)

    # concatenate in pairs since we want to predict differences
    point_clouds = onp.concatenate(
        [point_clouds[0 : batch_size * 2 : 2], point_clouds[1 : batch_size * 2 : 2]],
        axis=-1,
    )
    return diffs, point_clouds


def generate_single_example(key, height=320, width=320, num_points=999):
    batch_qp = vj_sample_qp(split(key))
    diffs = compute_diffs(batch_qp)
    batch_qp, diffs = jax.tree_map(lambda x: onp.array(x), [batch_qp, diffs])
    args = build_render_args(batch_qp, height, width, num_points)
    depths = array_apply(render_depth_array, args, False)
    point_clouds = array_apply(render_point_cloud, args, False)
    point_clouds = onp.stack(point_clouds)

    # concatenate in pairs since we want to predict differences
    point_clouds = onp.concatenate(
        [point_clouds[0 : 2 : 2], point_clouds[1 : 2 : 2]],
        axis=-1,
    )
    return batch_qp, diffs, depths, point_clouds

def sample_diffs_from_posterior(key, dist_params_list, num_samples):
    ks = split(key, num_samples*2)
    pos_dif = jax.vmap(gaussian_mixture_sample, in_axes=(0,None))(ks[:num_samples], dist_params_list[0])
    rot_dif = jax.vmap(gaussian_mixture_sample, in_axes=(0,None))(ks[num_samples:], dist_params_list[1])
    return dict(pos_diff=pos_dif, rot_diff=rot_dif)

def add_diffs_to_qp(qp, diffs):
    pos = qp['pos'] + diffs['pos_diff']
    rot = jax.vmap(jax.vmap(brax.math.quat_mul),in_axes=(0,None))(diffs['rot_diff'], qp['rot'])
    return dict(pos=pos, rot=rot, vel=jnp.zeros(pos.shape), ang=jnp.zeros(pos.shape))
    


def update_step(apply_fn, p_clouds, latents_list, opt_state, params, state, dropout_key, tx:optax.GradientTransformation):
    def loss(params):
        d_params_list = apply_fn(
            {"params": params, **state}, p_clouds, rngs={"dropout": dropout_key}
        )
        l = [
                -gaussian_mixture_logpdf(l, d).mean()
                for l, d in zip(latents_list, d_params_list)
            ]
        return sum(l)

    l, grads = jax.value_and_grad(loss, has_aux=False)(params)
    updates, opt_state = tx.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return opt_state, params, l

def procedure(load_save):
    key, *sks = split(PRNGKey(12345), 10)
    generation_size = 200000
    batch_size = 300
    obs_length = 999
    num_epochs = 999999
    load_idx = load_save
    save_idx = load_save+1

    m = IndependentGaussianMixtures(
        IndependentGaussianMixtureConfig(
            group_variables=(3, 4), num_mixtures_per_group=(2, 8)
        ),
        GaussianMixturePosteriorConfig(),
        TransformerConfig(),
    )

    variables = m.init(
        {"params": sks[0], "dropout": sks[1]}, jnp.ones((batch_size, obs_length, 6))
    )
    state, params = variables.pop('params')
    del variables
    
    # params_checkpoints = {}
    save_params = 10

    tx = optax.adam(learning_rate=0.0001)
    opt_state = tx.init(params)

    with open(f'params_{load_idx}', 'rb') as f:
        params = pickle.load(f)
    with open(f'opt_state_{load_idx}', 'rb') as f:
        opt_state = pickle.load(f)    
    with open(f'key_{load_idx}', 'rb') as f:
        loaded_key = pickle.load(f)    
    

    finished = False
    print_every = 20
    for e in range(num_epochs):
        old_key = key
        key = loaded_key if e == 0 else key
        key, subkey = split(key)
        diffs, p_clouds = generate_data_batch(subkey, generation_size, num_points=obs_length)
        for i in range(generation_size//batch_size - 1):
            # if (i+1) % save_params == 0:
            #     params_checkpoints[f'opt_state_{e}_{i}'] = jax.tree_map(onp.array,opt_state)
            #     params_checkpoints[f'params_{e}_{i}'] = jax.tree_map(onp.array,params)
            #     params_checkpoints[f'key_{e}_{i}'] = jax.tree_map(onp.array,key)
            key, subkey = split(key)
            # opt_state, params, l = update_step(
            opt_state, params, l = jax.jit(update_step, static_argnums=(0,7))(
                m.apply,
                p_clouds = jnp.array(p_clouds[i*batch_size: (i+1)*batch_size]),
                latents_list = [jnp.array(diffs['pos_diff'][i*batch_size: (i+1)*batch_size]), 
                                jnp.array(diffs['rot_diff'][i*batch_size: (i+1)*batch_size])],
                opt_state=opt_state,
                params = params,
                state = state,
                dropout_key=subkey,
                tx=tx,
            )
            if i%print_every == 0:
                print(e,i,l)
                # print(e,i,0)
            if jnp.isnan(l) or l == 0:
            # if  (e==89 and i == 499):
                print('failed',e,i,l)
                finished = True
                break
            if (i+1) % save_params == 0:
                with open(f'params_{save_idx}','wb') as f:
                        pickle.dump(params, f)
                with open(f'opt_state_{save_idx}','wb') as f:
                        pickle.dump(opt_state, f)
                with open(f'key_{save_idx}','wb') as f:
                        pickle.dump(old_key, f)

        if finished:
            break



if __name__ == "__main__":
    load_save = 17
    while True:
        procedure(load_save)
        load_save += 1
    # key, *sks = split(PRNGKey(12345), 10)
    # generation_size = 20000
    # batch_size = 150
    # obs_length = 999
    # num_epochs = 999999
    # load_idx = 4
    # save_idx = 5

    # m = IndependentGaussianMixtures(
    #     IndependentGaussianMixtureConfig(
    #         group_variables=(3, 4), num_mixtures_per_group=(2, 8)
    #     ),
    #     GaussianMixturePosteriorConfig(),
    #     TransformerConfig(),
    # )

    # variables = m.init(
    #     {"params": sks[0], "dropout": sks[1]}, jnp.ones((batch_size, obs_length, 6))
    # )
    # state, params = variables.pop('params')
    # del variables
    
    # params_checkpoints = {}
    # save_params = 10

    # tx = optax.adam(learning_rate=0.0005)
    # opt_state = tx.init(params)

    # with open(f'params_{load_idx}', 'rb') as f:
    #     params = pickle.load(f)
    # with open(f'opt_state_{load_idx}', 'rb') as f:
    #     opt_state = pickle.load(f)    
    # with open(f'key_{load_idx}', 'rb') as f:
    #     loaded_key = pickle.load(f)    
    

    # finished = False
    # print_every = 10
    # for e in range(num_epochs):
    #     old_key = key
    #     key = loaded_key if e == 0 else key
    #     key, subkey = split(key)
    #     diffs, p_clouds = generate_data_batch(subkey, generation_size, num_points=obs_length)
    #     for i in range(generation_size//batch_size - 1):
    #         if (i+1) % save_params == 0:
    #             params_checkpoints[f'opt_state_{e}_{i}'] = jax.tree_map(onp.array,opt_state)
    #             params_checkpoints[f'params_{e}_{i}'] = jax.tree_map(onp.array,params)
    #             params_checkpoints[f'key_{e}_{i}'] = jax.tree_map(onp.array,key)
    #         key, subkey = split(key)
    #         # opt_state, params, l = update_step(
    #         opt_state, params, l = jax.jit(update_step, static_argnums=(0,7))(
    #             m.apply,
    #             p_clouds = jnp.array(p_clouds[i*batch_size: (i+1)*batch_size]),
    #             latents_list = [jnp.array(diffs['pos_diff'][i*batch_size: (i+1)*batch_size]), 
    #                             jnp.array(diffs['rot_diff'][i*batch_size: (i+1)*batch_size])],
    #             opt_state=opt_state,
    #             params = params,
    #             state = state,
    #             dropout_key=subkey,
    #             tx=tx,
    #         )
    #         if i%print_every == 0:
    #             print(e,i,l)
    #             # print(e,i,0)
    #         if jnp.isnan(l) or l == 0:
    #         # if  (e==89 and i == 499):
    #             print('failed',e,i,l)
    #             finished = True
    #             break
    #         with open(f'params_{save_idx}','wb') as f:
    #                 pickle.dump(params, f)
    #         with open(f'opt_state_{save_idx}','wb') as f:
    #                 pickle.dump(opt_state, f)
    #         with open(f'key_{save_idx}','wb') as f:
    #                 pickle.dump(old_key, f)

    #     if finished:
    #         break
