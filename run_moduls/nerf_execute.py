import numpy as np
import torch
import tqdm 

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 

def run_model(data_loader, optimizer, model, criterion, metric, configs, is_train=True):
    '''
    The batch size for this one must be 1.
    '''
    model_corse, model_fine = model
    cumulative_loss = 0
    dataset_size = len(data_loader.dataset)
    #@NOTE: Use the one-image version of loading
    #@TODO: Check the dataloader returns the desired elements
    for img_w, img_h, focal, intrinsics, extrinsics, target in data_loader:
        if is_train == 0:
            optimizer.zero_grad(set_to_none=True)
        outputs = model(batch)
        if is_train == 2:
            return outputs, None
        loss = criterion(outputs, labels)
        cumulative_loss += loss
        metric.batch_accum(batch, outputs, labels)
        if is_train == 0:
            loss.backward()
            #@TODO: Check the implemetation in engine.py
            optimizer.step()
    return cumulative_loss.item() / dataset_size, metric.epoch_result(dataset_size)

def sample_pdf(bins, weights, N_samples, det=False):

    # Get pdf
    weights += 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, dim=-1, keepdims=True)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., N_samples)
        u = torch.broadcast_to(u, list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Invert CDF
    inds = torch.searchsorted(cdf, u, side='right')
    below = torch.maximum(0, inds-1)
    above = torch.minimum(cdf.shape[-1]-1, inds)
    inds_g = torch.stack([below, above], -1)
    #@FIXME: The gather has pytorch incompatable keyword argument
    cdf_g = torch.gather(cdf, dim=-1, index = inds_g, batch_dims=len(inds_g.shape)-2)
    bins_g = torch.gather(bins, dim=-1, index = inds_g, batch_dims=len(inds_g.shape)-2)

    denom = (cdf_g[..., 1]-cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[..., 0])/denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1]-bins_g[..., 0])

    return samples


def get_rays(H, W, focal, c2w):
    """Get ray origins, directions from a pinhole camera."""
    i, j = torch.meshgrid(torch.arange(W, dtype=torch.float32),
                       torch.arange(H, dtype=torch.float32), indexing='xy')
    dirs = torch.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -torch.ones_like(i)], -1)
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], dim =-1)
    rays_o = torch.broadcast_to(c2w[:3, -1], rays_d.shape)
    return rays_o, rays_d


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches."""
    if chunk is None:
        return fn

    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM."""
    all_ret = []
    for i in range(0, rays_flat.shape[0], chunk):
        #@TODO: Make kwargs to actual arguments
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        all_ret.append(ret)
    return all_ret


def render_rays(ray_batch,
                network,
                N_samples,
                lindisp=False,
                perturb=0.,
                N_importance=0):

    def raw2outputs(network_output, sampled_z_vals, rays_d):
        def raw2alpha(in_put, dists): 
            return 1.0 - torch.exp(-in_put * dists)
        dists = sampled_z_vals[..., 1:] - sampled_z_vals[..., :-1]
        dists = torch.cat(
            [dists, torch.broadcast_to(torch.tensor([1e10],device=device), dists[..., :1].shape)],
            axis=-1)  # [N_rays, N_samples]
        dists = dists * torch.linalg.norm(rays_d[..., None, :], axis=-1)
        rgb = network_output[...,:3]
        rou = network_output[...,-1:]
        alpha = raw2alpha(rou, dists)  # [N_rays, N_samples]
        #@NOTE:Risky
        weights = alpha * torch.cumprod((1.-alpha + 1e-10)[:-1], dim=-1)
        rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)  # [N_rays, 3]
        return rgb_map, weights
    net_corse, net_fine = network
    #Ray batch contain: [ray origin 3, ray direction 3, ray bound 2, ray view direction (optional) 3]
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each
    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None
    bounds = ray_batch[..., 6:8].reshape([-1, 1, 2])
    near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]
    t_vals = torch.linspace(0., 1., N_samples, device=device)
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))
    z_vals = torch.broadcast_to(z_vals, [N_rays, N_samples])
    if perturb > 0.:
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
        lower = torch.cat([z_vals[..., :1], mids], dim=-1)
        t_rand = torch.rand(z_vals.shape)
        z_vals = lower + (upper - lower) * t_rand
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]
    #@TODO:Make the query function a simple function call 
    #@FIXME: Also not working
    raw = run_network(pts, viewdirs, net_corse)  # [N_rays, N_samples, 4]
    rgb_map, weights = raw2outputs(raw, z_vals, rays_d)
    if N_importance > 0:
        rgb_map_0 = rgb_map
        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(
            z_vals_mid, weights[..., 1:-1], N_importance, det=(perturb == 0.))
        z_samples.requires_grad = False 
        z_vals = torch.sort(torch.cat([z_vals, z_samples], -1), dim=-1)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * \
            z_vals[..., :, None]  # [N_rays, N_samples + N_importance, 3]
        raw = run_network(pts, viewdirs, net_fine)
        rgb_map,_= raw2outputs(raw, z_vals, rays_d)
    return rgb_map

def run_network(inputs, viewdirs, fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'."""
    inputs_flat =  inputs.reshape([-1, inputs.shape[-1]])
    input_dirs = torch.broadcast_to(viewdirs[:, None], inputs.shape)
    input_dirs_flat = input_dirs.reshape([-1, input_dirs.shape[-1]])
    combined_input = torch.cat([inputs_flat, input_dirs_flat], dim=-1)
    #@TODO: Check the dimension of the combined_input
    outputs_flat = batchify(fn, netchunk)(combined_input)
    outputs = torch.reshape(outputs_flat, list(
        inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs

#@TODO: Make sure the render arguments are correct
def render(H, W, focal,
           chunk=1024*32, rays=None, c2w=None, ndc=True,
           near=0., far=1.,
           use_viewdirs=False, c2w_staticcam=None,
           **kwargs):

    if c2w is not None:
        rays_o, rays_d = get_rays(H, W, focal, c2w)
    else:
        rays_o, rays_d = rays
    if use_viewdirs:
        viewdirs = rays_d
        if c2w_staticcam is not None:
            rays_o, rays_d = get_rays(H, W, focal, c2w_staticcam)
        viewdirs = viewdirs / torch.linalg.norm(viewdirs, dim=-1, keepdims=True)
        viewdirs = torch.type(viewdirs.reshape([-1, 3]), dtype=torch.float32)

    sh = rays_d.shape  # [..., 3]
    rays_o = torch.type(rays_o.reshape([-1, 3]), dtype=torch.float32)
    rays_d = torch.type(rays_d.reshape([-1, 3]), dtype=torch.float32)
    near, far = near * \
        torch.ones_like(rays_d[..., :1]), far * torch.ones_like(rays_d[..., :1])
    rays = torch.cat([rays_o, rays_d, near, far], dim=-1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], dim=-1)
    #@FIXME: Change the batchify argument to the correct ones
    all_ret = batchify_rays(rays, chunk, **kwargs)
    return all_ret


def render_path(render_poses, hwf, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):
    H, W, focal = hwf
    if render_factor != 0:
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []

    for i, c2w in tqdm.tqdm(enumerate(render_poses)):
        #@TODO: Make the render argument correct
        rgb = render(
            H, W, focal, chunk=chunk, c2w=c2w[:3, :4], **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
    rgbs = np.stack(rgbs, 0)
    return rgbs