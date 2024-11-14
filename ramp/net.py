import time
import torch
import numpy as np
import torch.nn as nn
from functools import partial
import torch.nn.functional as F

from . import fastba
from . import altcorr
from .lietorch import SE3
from .blocks import GradientClip, GatedResidual, SoftAgg
from .extractor import (
    MergerLSTMsceneEncoder,
    MultiScaleMergerDoubleNet,
)
from .utils import (
    get_coords_from_topk_events,
    coords_grid_with_index,
    preprocess_input,
    get_channel_dim,
    flatmeshgrid,
    pyramidify,
    set_depth,
    timer,
)
from .pose_prediction.pose_pred_utils import motion_bootstrap
from .ba import BA
from . import projective_ops as pops

autocast = torch.amp.autocast("cuda", enabled=False)
DIM = 384


class Update(nn.Module):
    def __init__(self, p):
        super(Update, self).__init__()

        self.c1 = nn.Sequential(nn.Linear(DIM, DIM), nn.ReLU(inplace=True), nn.Linear(DIM, DIM))

        self.c2 = nn.Sequential(nn.Linear(DIM, DIM), nn.ReLU(inplace=True), nn.Linear(DIM, DIM))

        self.norm = nn.LayerNorm(DIM, eps=1e-3)

        self.agg_kk = SoftAgg(DIM)
        self.agg_ij = SoftAgg(DIM)

        self.gru = nn.Sequential(
            nn.LayerNorm(DIM, eps=1e-3),
            GatedResidual(DIM),
            nn.LayerNorm(DIM, eps=1e-3),
            GatedResidual(DIM),
        )

        self.corr = nn.Sequential(
            nn.Linear(2 * 49 * p * p, DIM),
            nn.ReLU(inplace=True),
            nn.Linear(DIM, DIM),
            nn.LayerNorm(DIM, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.Linear(DIM, DIM),
        )

        self.d = nn.Sequential(nn.ReLU(inplace=False), nn.Linear(DIM, 2), GradientClip())

        self.w = nn.Sequential(
            nn.ReLU(inplace=False), nn.Linear(DIM, 2), GradientClip(), nn.Sigmoid()
        )

    def forward(self, net, inp, corr, flow, ii, jj, kk):
        """update operator"""

        # net: the hidden state (size=#edges x 384)
        # inp: corr:
        net = net + inp + self.corr(corr)
        net = self.norm(net)

        ix, jx = fastba.neighbors(kk, jj)
        mask_ix = (ix >= 0).float().reshape(1, -1, 1)
        mask_jx = (jx >= 0).float().reshape(1, -1, 1)

        net = net + self.c1(mask_ix * net[:, ix])
        net = net + self.c2(mask_jx * net[:, jx])

        net = net + self.agg_kk(net, kk)
        net = net + self.agg_ij(net, ii * 12345 + jj)

        net = self.gru(net)

        # d: trajectory update | w: confidence score
        return net, (self.d(net), self.w(net), None)


class Patchifier(nn.Module):
    def __init__(self, channels_dim, patch_size=3, input_mode="MultiScale"):
        super(Patchifier, self).__init__()
        self.input_mode = input_mode
        self.P = patch_size
        # self.input_mode = "SingleScale"
        # self.input_mode = "MultiScale"

        if self.input_mode in ("SingleScale"):
            evs_ch_dim, img_ch_dim = channels_dim
            self.encoder = MergerLSTMsceneEncoder(
                evs_ch_dim=evs_ch_dim,
                img_ch_dim=img_ch_dim,
                output_lstm_dim=15,  # TODO tunable parameter
                output_dim_f=128,
                output_dim_i=DIM,
                norm_fn_fmap="instance",
                norm_fn_imap="none",
                kernel_size_superstate=1,
            )
        elif self.input_mode in ("MultiScale"):
            evs_ch_dim, img_ch_dim = channels_dim
            self.encoder = MultiScaleMergerDoubleNet(
                evs_ch_dim=evs_ch_dim,
                img_ch_dim=img_ch_dim,
                lstm_dim=16,
                output_dim_f=128,
                output_dim_i=DIM,
                norm_fn_fmap="instance",
                norm_fn_imap="none",
                norm_superstate=False,
            )
        else:
            raise ValueError(f"Invalid input mode: {self.input_mode}")

    def forward(
        self,
        input_,
        patches_per_image=80,
        reinit_hidden=False,
        disps=None,
        event_bias=False,
        gradient_bias=False,
    ):
        """Compute features and extract patches from input images"""

        events, images, mask = None, None, None

        if self.input_mode in ("SingleScale"):
            events, images, _ = input_
            fmap, imap, _ = self.encoder(events=events, images=images, reinit_hidden=reinit_hidden)
            fmap = fmap / 4.0
            imap = imap / 4.0
        elif self.input_mode in ("MultiScale"):
            events, images, mask = input_
            fmap, imap = self.encoder(
                events=events, images=images, mask=mask, reinit_hidden=reinit_hidden
            )
            events = events[mask]
            fmap = fmap / 4.0
            imap = imap / 4.0
        else:
            fmap = self.fnet(input_) / 4.0
            imap = self.inet(input_) / 4.0

        if mask is not None and not mask.any():
            return None, None, None, None, None, None

        b, n, c, h, w = fmap.shape

        # bias selection towards regions with highest value of event mean map
        if event_bias:
            event_inp = events if events is not None else input_
            coords = get_coords_from_topk_events(
                events=event_inp,
                patches_per_image=patches_per_image,
                border_suppression_size=0,
                non_max_supp_rad=11,
            )
        elif gradient_bias:
            images_inp = images if images is not None else input_
            g = self.__image_gradient(images_inp)
            x = torch.randint(1, w - 1, size=[n, 3 * patches_per_image], device="cuda")
            y = torch.randint(1, h - 1, size=[n, 3 * patches_per_image], device="cuda")

            coords = torch.stack([x, y], dim=-1).float()
            g = altcorr.patchify(g[0, :, None], coords, 0).view(n, 3 * patches_per_image)

            ix = torch.argsort(g, dim=1)
            x = torch.gather(x, 1, ix[:, -patches_per_image:])
            y = torch.gather(y, 1, ix[:, -patches_per_image:])
        # random patch selection
        else:
            x = torch.randint(1, w - 1, size=[n, patches_per_image], device="cuda")
            y = torch.randint(1, h - 1, size=[n, patches_per_image], device="cuda")
            coords = torch.stack([x, y], dim=-1).float()

        gmap = altcorr.patchify(fmap[0], coords, 1).view(b, -1, 128, self.P, self.P)
        imap = altcorr.patchify(imap[0], coords, 0).view(b, -1, DIM, 1, 1)

        if disps is None:
            disps = torch.ones(b, n, h, w, device="cuda")

        grid, _ = coords_grid_with_index(disps, device=fmap.device)
        patches = altcorr.patchify(grid[0], coords, self.P // 2).view(b, -1, 3, self.P, self.P)

        index = torch.arange(n, device="cuda").view(n, 1)
        index = index.repeat(1, patches_per_image).reshape(-1)
        
        clr = altcorr.patchify(images[0], 4*(coords + 0.5), 0).view(b, -1, 3)
        return fmap, gmap, imap, patches, index, clr


class CorrBlock:
    def __init__(self, fmap, gmap, radius=3, dropout=0.2, levels=[1, 4]):
        self.dropout = dropout
        self.radius = radius
        self.levels = levels

        self.gmap = gmap
        self.pyramid = pyramidify(fmap, lvls=levels)

    def __call__(self, ii, jj, coords):
        corrs = []
        for i in range(len(self.levels)):
            corrs += [
                altcorr.corr(
                    self.gmap,
                    self.pyramid[i],
                    coords / self.levels[i],
                    ii,
                    jj,
                    self.radius,
                    self.dropout,
                )
            ]
        return torch.stack(corrs, -1).view(1, len(ii), -1)


class VONet(nn.Module):
    def __init__(self, cfg):
        super(VONet, self).__init__()
        self.P = 3
        self.RES = 4
        self.DIM = DIM
        self.EVENT_BIAS = cfg["event_bias"]
        self.MOTION_MODEL = "DAMPED_LINEAR"
        self.MOTION_DAMPING = 0.5
        self.inp_channel_dims = get_channel_dim(cfg)
        self.input_mode = cfg["input_mode"]

        self.patchify = Patchifier(
            channels_dim=self.inp_channel_dims,
            patch_size=self.P,
            input_mode=self.input_mode
        )
        self.update = Update(self.P)

    @autocast
    def forward(self, input_: tuple, poses, disps, intrinsics, STEPS=12, structure_only=False):

        """Estimates SE3 or Sim3 between pair of frames"""

        input_ = preprocess_input(input_tensor=input_)

        intrinsics = intrinsics / 4.0
        disps = disps[:, :, 1::4, 1::4].float()

        # fmap: extracted feature map, gmap:, imap:,
        # patches: depths patches, ix: image indices (img 1, img 2, ...)
        fmap, gmap, imap, patches, ix = self.patchify(
            input_=input_,
            disps=disps,
            reinit_hidden=True,
            event_bias=self.EVENT_BIAS,
        )

        corr_fn = CorrBlock(fmap, gmap)

        b, N, c, h, w = fmap.shape
        p = self.P

        patches_gt = patches.clone()
        Ps = poses

        d = patches[..., 2, p // 2, p // 2]
        patches = set_depth(patches, torch.rand_like(d))

        kk, jj = flatmeshgrid(torch.where(ix < 8)[0], torch.arange(0, 8, device="cuda"))
        ii = ix[kk]

        imap = imap.view(b, -1, DIM)
        net = torch.zeros(b, len(kk), DIM, device="cuda", dtype=torch.float)

        Gs = SE3.IdentityLike(poses)

        if structure_only:
            Gs.data[:] = poses.data[:]

        # TODO train with bootstrap pose

        traj = []
        # bounds control the area where the patches are allowed to move
        # substitute filter features function
        bounds = [-64, -64, w + 64, h + 64]

        while len(traj) < STEPS:
            Gs = Gs.detach()
            patches = patches.detach()

            n = ii.max() + 1
            input_shape = input_[1].shape[1] if isinstance(input_, tuple) else input_.shape[1]
            # if initialized "len(traj) >= 8" and if analyzed images < existing images
            if len(traj) >= 8 and n < input_shape:

                if not structure_only:
                    Gs.data[:, n] = motion_bootstrap(
                        MOTION_DAMPING=self.MOTION_DAMPING,
                        MOTION_MODEL=self.MOTION_MODEL,
                        poses=Gs.data[0, :],
                        n=n,
                    )
                    # TODO init gs.data better to reduce sim2real gap
                    # Gs.data[:, n] = Gs.data[:, n - 1]

                kk1, jj1 = flatmeshgrid(
                    torch.where(ix < n)[0], torch.arange(n, n + 1, device="cuda")
                )
                kk2, jj2 = flatmeshgrid(
                    torch.where(ix == n)[0], torch.arange(0, n + 1, device="cuda")
                )

                ii = torch.cat([ix[kk1], ix[kk2], ii])
                jj = torch.cat([jj1, jj2, jj])
                kk = torch.cat([kk1, kk2, kk])

                net1 = torch.zeros(b, len(kk1) + len(kk2), DIM, device="cuda")
                net = torch.cat([net1, net], dim=1)

                if np.random.rand() < 0.1:
                    k = (ii != (n - 4)) & (jj != (n - 4))
                    ii = ii[k]
                    jj = jj[k]
                    kk = kk[k]
                    net = net[:, k]

                patches[:, ix == n, 2] = torch.median(patches[:, (ix == n - 1) | (ix == n - 2), 2])
                n = ii.max() + 1

            coords = pops.transform(Gs, patches, intrinsics, ii, jj, kk)
            coords1 = coords.permute(0, 1, 4, 2, 3).contiguous()

            corr = corr_fn(kk, jj, coords1)
            net, (delta, weight, _) = self.update(net, imap[:, kk], corr, None, ii, jj, kk)

            lmbda = 1e-4
            target = coords[..., p // 2, p // 2, :] + delta

            ep = 10
            for itr in range(2):
                Gs, patches = BA(
                    Gs,
                    patches,
                    intrinsics,
                    target,
                    weight,
                    lmbda,
                    ii,
                    jj,
                    kk,
                    bounds,
                    ep=ep,
                    fixedp=1,
                    structure_only=structure_only,
                )

            dij = (ii - jj).abs()
            k = (dij > 0) & (dij <= 2)

            coords = pops.transform(Gs, patches, intrinsics, ii[k], jj[k], kk[k])
            coords_gt, valid, _ = pops.transform(
                Ps, patches_gt, intrinsics, ii[k], jj[k], kk[k], jacobian=True
            )

            traj.append((valid, coords, coords_gt, Gs[:, :n], Ps[:, :n]))
        return traj
