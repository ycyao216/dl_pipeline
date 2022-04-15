import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def positional_encoding(x):
    """Position encoding using frequency function sin and cos

    Parameters
    ----------
    x : torch.tensor
        The inpu batch [batch num, 6(:3 position, 3: direction)]

    Returns
    -------
    tuple
        position encoding [ batch num, 3], direction encoding [batch num, 3]
    """
    pose_enc_len = 10
    dir_enc_len = 4
    x_position = x[:, :3]
    x_dir = x[:, 3:]
    indices_x_pose = torch.arange(0, pose_enc_len, dtype=torch.int, device=device)
    indices_x_dir = torch.arange(0, dir_enc_len, dtype=torch.int, device=device)
    coeffs_pose = torch.pi * torch.pow(
        (torch.ones(pose_enc_len, device=device) * 2), indices_x_pose
    )
    coeffs_dir = torch.pi * torch.pow(
        (torch.ones(dir_enc_len, device=device) * 2), indices_x_dir
    )
    sin_pos = torch.sin(coeffs_pose * x_position)
    cos_pos = torch.cos(coeffs_pose * x_position)
    pose_enc = (
        torch.stack((sin_pos, cos_pos), dim=0)
        .view(x.shape[0], 2 * pose_enc_len)
        .t()
        .contiguous()
        .view(x.shape[0], 2 * self.pose_enc_len)
    )
    sin_dir = torch.sin(coeffs_dir * x_dir)
    cos_dir = torch.cos(coeffs_dir * x_dir)
    dir_enc = (
        torch.stack((sin_dir, cos_dir), dim=0)
        .view(x.shape[0], 2 * dir_enc_len)
        .t()
        .contiguous()
        .view(x.shape[0], 2 * self.dir_enc_len)
    )
    return pose_enc, dir_enc


def posenc(x, enc_size):
    rets = []
    for i in range(enc_size):
        for fn in [torch.sin, torch.cos]:
            rets.append(fn(2.0**i * x))
    return torch.cat(rets, -1)


torch.cuda.empty_cache()
test_in = torch.randn((10, 3))
out = posenc(test_in)
print(out)
