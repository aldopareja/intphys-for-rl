from functools import partial

import jax
from jax import vmap, jit
from jax import numpy as jnp
from jax.random import uniform, split, PRNGKey, normal, bernoulli
import jax.tree_util as jtu

from matplotlib import pyplot as plt


Quaternion = jnp.ndarray
Vector3 = jnp.ndarray
BoxFace = jnp.ndarray
Point = jnp.ndarray
Plane = jnp.ndarray
Box = jnp.ndarray

MAX_HALF_SIZE = 1.0
MAX_TRANSLATION = 2.0
MAX_DRAW_SIDE = 3**0.5 * (MAX_TRANSLATION + MAX_HALF_SIZE) + 0.01
RESOLUTION = 300
DEF_DEPTH = -MAX_DRAW_SIDE - 1e-4
MAX_LATENT = max([MAX_HALF_SIZE,MAX_TRANSLATION,1.0]) + 0.1


def rot_axis_to_quaternion(axis: Vector3, angle: jnp.ndarray) -> Quaternion:
    img = axis * jnp.sin(angle / 2)
    real = jax.lax.cos(angle / 2)[None]
    return jax.lax.concatenate([real, img], dimension=0)


def rotate(vec: Vector3, quat: Quaternion):
    """Rotates a vector vec by a unit quaternion quat.

    Args:
      vec: (3,) a vector
      quat: (4,) a quaternion

    Returns:
      ndarray(3) containing vec rotated by quat.
    """
    if len(vec.shape) != 1:
        raise AssertionError("vec must have no batch dimensions.")
    s, u = quat[0], quat[1:]
    r = 2 * (jnp.dot(u, vec) * u) + (s * s - jnp.dot(u, u)) * vec
    r = r + 2 * s * jnp.cross(u, vec)
    return r


def sample_quaternion(key: PRNGKey):
    # sample imaginary part of quaternion (axis of rotation)
    key, *subkeys = split(key, 4)
    axis = uniform(subkeys[0], minval=-1.0, maxval=1.0, shape=(3,)) + 1e-6
    axis = axis / jnp.linalg.norm(axis)

    # sample angle (how much to rotate)
    angle = uniform(subkeys[1], minval=-2 * jnp.pi, maxval=2 * jnp.pi)

    return rot_axis_to_quaternion(axis, angle)


def uniform_sample_plane(key) -> Plane:
    """samples a,b,c,d such that a+b+c = 1. A plane is characterized by equation ax + by + cz + d = 0"""
    eps = 1e-6
    keys = split(key, 3)
    a, b = jax.lax.sort(
        uniform(keys[0], shape=(2,)) * (1 - eps) + eps
    )  # avoids degenerate planes
    c = (1 - b) + eps
    b = b - a
    d = uniform(keys[1]) * 10 + eps
    return jnp.stack([a, b, c, d])


def make_corners_from_half_vectors(v1, v2, v3):
    """make corners keeping v1 static and adding +-v2 and +-v3 to make a box. The last corner isn't necessary."""
    a = v2 + v3
    b = v2 - v3
    c = -v2 - v3
    # TODO: how am I supposed to check if this works then?
    # assert jnp.all(jnp.abs(jnp.dot(a-b,c-b)) < 1e-4)
    return v1[None] + jnp.stack([a, b, c], axis=0)


def translate_plane(p: Plane, t: Vector3):
    assert p.shape == (4,)
    assert t.shape == (3,)
    d = p[-1] - (p[:3] * t).sum()[None]
    return jnp.concatenate([p[:3], d], axis=0)


def make_box(half_size_x, half_size_y, half_size_z, translation, rot_quaternion):
    """
    makes a box, which is a set of box_faces, each box face is a plane and a set of three points in such plane that delimit it.
    returns:
        Box:
            planes: (6,4) array with the 6 planes each belonging to a box_face, planes are parameterized such that ax + by + cz + d = 0
            corners: (6,3,3) array with 6 sets of 3 points (each of size 3) where each point is a corner delimiting the box_face. The second point is the corner.
    NOTE: the normal of the plane a,b,c is parameterized to look away from the origin. This can be used to check if a ray is hitting it from behind or
    from the front. I would assume front is
    """
    hx, hy, hz = half_size_x, half_size_y, half_size_z
    t = translation
    q = rot_quaternion

    planes_d = jnp.stack([hx, hy, hz]).reshape(-1, 1)
    planes_normals = jnp.eye(3) * planes_d

    v_rotate = vmap(rotate, in_axes=(0, None))

    planes_normals = v_rotate(planes_normals, q)
    planes_normals += jnp.where(jnp.isclose(planes_normals,0),1e-7,0.0)

    corners = []
    for i in range(3):
        k, j = {0, 1, 2}.difference({i})
        pos = make_corners_from_half_vectors(
            -planes_normals[i], planes_normals[k], planes_normals[j]
        )
        neg = make_corners_from_half_vectors(
            planes_normals[i], planes_normals[k], planes_normals[j]
        )
        corners.append([pos, neg])

    corners = jnp.array(corners)

    planes_d = jnp.concatenate([planes_d, planes_d], axis=0)

    normals = jnp.concatenate([jnp.eye(3), -jnp.eye(3)], axis=0)
    normals = v_rotate(normals, q)
    normals += jnp.where(jnp.isclose(normals,0),1e-7,0.0)

    box_planes = jnp.concatenate([normals, planes_d], axis=1)
    box_planes = vmap(translate_plane, in_axes=(0, None))(box_planes, t)

    faces_corners = jnp.concatenate([corners[:, 0], corners[:, 1]]) + t

    return dict(faces_planes=box_planes, faces_corners=faces_corners)


def point_in_box_face(d: Point, b: BoxFace):
    """
    Args:
        - p: (d,) shaped array including coordinates of point
        - b: (3,d) shaped array including 3 points determining the box face, the second (b[1]) must be the corner uniting the other points.
    returns:
        - true if the point is inside the boxface, i.e the projection of the point to each edge is less than the size of the edge.

    NOTE: this doesn't check it but d should be guaranteed to be on the plane of BoxFace or this would just not work.
    """
    assert d.shape == (3,) and b.shape == (3, 3)
    e1 = b[0] - b[1]
    e2 = b[2] - b[1]
    p = d - b[1]

    pr1 = jnp.dot(p, e1) / (jnp.linalg.norm(e1)) ** 2
    pr2 = jnp.dot(p, e2) / (jnp.linalg.norm(e2)) ** 2
    # print(p,e1,e2,pr1,pr2)
    def proj_within(pr):
        return (pr >= 0) & (pr <= 1)

    return proj_within(pr1) & proj_within(pr2)


def get_depth_pixel_from_plane(x, y, p: Plane):
    """does orthographic projection of a pixel and returns the z value, we assume right handed coordinates where z is coming into the camera
    the camera is assumed to be at the origin. This function doesn't work with degenerate planes. If the plane is not in "front" of the camera (i.e. the z value
    is less than 0) then it returs 0
    """
    # TODO should I ignore points that are looking "away"? i.e. taking into account the orientation of the normal?.
    v = (-p[3] - p[0] * x - p[1] * y) / p[2]
    # v = jax.lax.cond(v>0, lambda: 0.0, lambda: v)
    return v
    # return jnp.array([x,y,(-p.d - p.a * x - p.b*y)/p.c])


def get_depth_pixel_from_box_face(x, y, face_plane, face_corners):
    z = get_depth_pixel_from_plane(x, y, face_plane)
    plane_inter = jnp.array([x, y, z])
    return jax.lax.cond(
        point_in_box_face(plane_inter, face_corners), lambda: z, lambda: DEF_DEPTH
    )


def get_depth_pixel_from_box(x, y, box, return_coordinates=False):
    """takes a box dict with 'faces_planes' and 'faces_corners' and returns the depth of the closest intersection with all of the box's planes"""
    all_depths = vmap(get_depth_pixel_from_box_face, in_axes=(None, None, 0, 0))(
        x, y, box["faces_planes"], box["faces_corners"]
    )
    ret = all_depths.max()
    ret = (ret, x, y) if return_coordinates else ret
    return ret


render_face = vmap(
    vmap(get_depth_pixel_from_box_face, in_axes=(None, 0, None, None)),
    in_axes=(0, None, None, None),
)
render_box = vmap(
    vmap(get_depth_pixel_from_box, in_axes=(None, 0, None, None)), in_axes=(0, None, None, None)
)



@vmap
def make_p_mask(depth):
    assert len(depth.shape) == 0
    return jax.lax.cond(
        depth != DEF_DEPTH, lambda: jnp.array(1.0), lambda: jnp.array(0.0)
    )

def array_idx_to_coord(idx):
    assert len(idx.shape) == 1 and idx.dtype == jnp.int32
    return ((idx/RESOLUTION * 2 - 1.0) * MAX_DRAW_SIDE)

def coord_to_array_idx(x):
    assert len(x.shape) == 1 and x.dtype == jnp.float32
    return jnp.int32(((x/MAX_DRAW_SIDE + 1.0) / 2 * RESOLUTION).round())

def make_box_from_latents(latents):
    assert len(latents.shape) == 1 and latents.shape[0] == 10
    hx, hy, hz = jnp.abs(latents[:3])
    t = latents[3:6]
    q = latents[6:]
    
    box = make_box(hx, hy, hz, t, q/jnp.linalg.norm(q))
    return box

def simulate(key, latents, obs_noise_std: float, num_points: int):
    box = make_box_from_latents(latents)
    
    ks = split(key)

    xs, ys = [array_idx_to_coord(jnp.arange(RESOLUTION)) for _ in range(2)]
    depth, xs, ys = render_box(xs, ys, box, True)
    
    depth, xs, ys = jtu.tree_map(lambda x: x.flatten(), (depth, xs, ys))
    
    p_mask = make_p_mask(depth)
    idx = jax.random.choice(ks[0], jnp.arange(depth.shape[0]), shape=(num_points,), p=p_mask/p_mask.sum())
    depth, xs, ys = jtu.tree_map(lambda x: x[idx], (depth, xs, ys))
    
    #sort wrt the sum in x,y to make better comparisons
    idx = jnp.stack([xs,ys]).sum(0).argsort()
    depth, xs, ys = jtu.tree_map(lambda x: x[idx], (depth, xs, ys))

    depth += normal(ks[1], shape=(num_points,)) * obs_noise_std

    point_cloud = jnp.stack([xs,ys,depth], axis=1) 
    return point_cloud, box

def generative_model(
    key: PRNGKey, obs_noise_std: float, num_points: int
):
    eps = 1e-6
    ks = split(key, 4)

    # half sizes
    hx, hy, hz = uniform(ks[0], shape=(3,)) * MAX_HALF_SIZE + eps
    # hx,hy,hz = 1,1,1

    # traslation
    t = (
        uniform(ks[2], shape=(3,)) * 2 - 1
    ) * MAX_TRANSLATION + eps  # TODO: I need to calibrate this so that I always sample on the field of vision.
    # t = t - jnp.array([0,0,4])
    # t = jnp.array([0,0,0])

    # rotation
    q = sample_quaternion(ks[1])
    # q = rot_axis_to_quaternion(jnp.array([0,1,0]),angle=jnp.pi/4)

    latents = jnp.concatenate([jnp.stack([hx, hy, hz]), t, q])
    
    point_cloud, box = simulate(ks[3],latents, obs_noise_std, num_points)

    return point_cloud, latents, box


def draw_box(box):
    xs,ys = [array_idx_to_coord(jnp.arange(RESOLUTION)) for _ in range(2)]
    depth_map = render_box(xs, ys, box, False)
    fig1, ax2 = plt.subplots(constrained_layout=True)

    cs = ax2.contourf(xs, ys, depth_map, 1000,)
    cbar = fig1.colorbar(cs)
    return ax2, fig1

def draw_box_no_bar(box,ax):
    xs,ys = [array_idx_to_coord(jnp.arange(RESOLUTION)) for _ in range(2)]
    depth_map = render_box(xs,ys,box,False)
    ax.contourf(xs,ys,depth_map,100)


def overimpose_point_cloud(fig, ax, pc):
    xs = (pc[:,1])
    ys = (pc[:,0])
    cs = ax.scatter(xs, ys, c=pc[:, 2], s=2, cmap="hot")
    # fig.colorbar(cs)
    
    
#### scoring point clouds
    
def score_point_from_box(point, box):
    x,y,z = point
    z_box = get_depth_pixel_from_box(x,y,box)
    return jax.scipy.stats.norm.logpdf(z,loc=z_box)

def score_point_cloud_from_box(pc, box):
    all_scores = vmap(score_point_from_box, in_axes=(0,None))(pc, box)
    return all_scores.mean()

score_point_cloud_from_many_boxes = jax.jit(vmap(score_point_cloud_from_box, in_axes=(None, 0)))

def score_point_cloud_from_latents(pc,latents):
    box = make_box_from_latents(latents)
    return score_point_cloud_from_box(pc,box)

score_many_point_clouds_from_many_latents = vmap(score_point_cloud_from_latents,in_axes=(0,0))

if __name__ == "__main__":
    pc, latents, box = generative_model(
        PRNGKey(0), obs_noise_std=0.8, num_points=100
    )
