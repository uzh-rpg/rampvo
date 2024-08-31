from ..lietorch import SE3
import numpy as np
import torch

def compute_distance_matrix_flow(poses, disps, intrinsics):
    """ compute flow magnitude between all pairs of frames """
    if not isinstance(poses, SE3):
        poses = torch.from_numpy(poses).float().cuda()[None]
        poses = SE3(poses).inv()

        disps = torch.from_numpy(disps).float().cuda()[None]
        intrinsics = torch.from_numpy(intrinsics).float().cuda()[None]

    N = poses.shape[1]
    
    ii, jj = torch.meshgrid(torch.arange(N), torch.arange(N))
    ii = ii.reshape(-1).cuda()
    jj = jj.reshape(-1).cuda()

    MAX_FLOW = 100.0
    matrix = np.zeros((N, N), dtype=np.float32)

    s = 2048
    for i in range(0, ii.shape[0], s):
        flow1, val1 = pops.induced_flow(poses, disps, intrinsics, ii[i:i+s], jj[i:i+s])
        flow2, val2 = pops.induced_flow(poses, disps, intrinsics, jj[i:i+s], ii[i:i+s])
        
        flow = torch.stack([flow1, flow2], dim=2)
        val = torch.stack([val1, val2], dim=2)
        
        mag = flow.norm(dim=-1).clamp(max=MAX_FLOW)
        mag = mag.view(mag.shape[1], -1)
        val = val.view(val.shape[1], -1)

        mag = (mag * val).mean(-1) / val.mean(-1)
        mag[val.mean(-1) < 0.7] = np.inf

        i1 = ii[i:i+s].cpu().numpy()
        j1 = jj[i:i+s].cpu().numpy()
        matrix[i1, j1] = mag.cpu().numpy()

    return matrix
