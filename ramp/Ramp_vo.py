import torch
import numpy as np
import torch.nn.functional as F

from . import fastba
from . import altcorr
from . import lietorch
from .lietorch import SE3

from .net import VONet
from .utils import *
from .utils import preprocess_input, filter_features
from . import projective_ops as pops
from collections import OrderedDict

from ramp.pose_prediction.pose_pred_utils import (
    compute_patch_track__,
    motion_bootstrap,
    add_forward_elements,
    fit_model_patch_track,
    predict_patch_on_model,
    )
autocast = torch.amp.autocast('cuda', enabled=True)
Id = SE3.Identity(1, device="cuda")


class Ramp_vo:
    def __init__(self, cfg, network, train_cfg, ht=480, wd=640):
        self.cfg = cfg
        self.event_bias = train_cfg["event_bias"]
        self.train_cfg = train_cfg

        # attributes for pose prediction
        self.patch_dict_=None
        self.patches_models=None
        self.lmbda = torch.as_tensor([1e-4], device="cuda")

        self.load_weights(network)
        self.is_initialized = False
        self.enable_timing = False
        
        self.n = 0      # number of frames
        self.m = 0      # number of patches
        self.M = self.cfg.PATCHES_PER_FRAME
        self.N = self.cfg.BUFFER_SIZE

        self.ht = ht    # image height
        self.wd = wd    # image width

        DIM = self.DIM
        RES = self.RES

        ### state attributes ###
        self.tlist = []
        self.counter = 0

        # dummy image for visualization
        self.image_ = torch.zeros(self.ht, self.wd, 3, dtype=torch.uint8, device="cpu")

        self.tstamps_ = torch.zeros(self.N, dtype=torch.long, device="cuda")
        self.poses_ = torch.zeros(self.N, 7, dtype=torch.float, device="cuda")
        self.patches_ = torch.zeros(self.N, self.M, 3, self.P, self.P, dtype=torch.float, device="cuda")
        self.intrinsics_ = torch.zeros(self.N, 4, dtype=torch.float, device="cuda")

        self.points_ = torch.zeros(self.N * self.M, 3, dtype=torch.float, device="cuda")
        self.colors_ = torch.zeros(self.N, self.M, 3, dtype=torch.uint8, device="cuda")

        self.index_ = torch.zeros(self.N, self.M, dtype=torch.long, device="cuda")
        self.index_map_ = torch.zeros(self.N, dtype=torch.long, device="cuda")

        ### network attributes ###
        self.mem = 32

        if self.cfg.MIXED_PRECISION:
            self.kwargs = kwargs = {"device": "cuda", "dtype": torch.half}
        else:
            self.kwargs = kwargs = {"device": "cuda", "dtype": torch.float}
        
        self.imap_ = torch.zeros(self.mem, self.M, DIM, **kwargs)
        self.gmap_ = torch.zeros(self.mem, self.M, 128, self.P, self.P, **kwargs)

        ht = ht // RES
        wd = wd // RES

        self.fmap1_ = torch.zeros(1, self.mem, 128, ht // 1, wd // 1, **kwargs)
        self.fmap2_ = torch.zeros(1, self.mem, 128, ht // 4, wd // 4, **kwargs)

        # feature pyramid
        self.pyramid = (self.fmap1_, self.fmap2_)

        self.net = torch.zeros(1, 0, DIM, **kwargs)
        self.ii = torch.as_tensor([], dtype=torch.long, device="cuda")
        self.jj = torch.as_tensor([], dtype=torch.long, device="cuda")
        self.kk = torch.as_tensor([], dtype=torch.long, device="cuda")
        
        # initialize poses to identity matrix
        self.poses_[:,6] = 1.0

        # store relative poses for removed frames
        self.delta = {}


    def load_weights(self, network):
        # load network from checkpoint file
        if isinstance(network, str):
            checkpoint = torch.load(network)

            if checkpoint.get('model_state_dict'):
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if "update.lmbda" not in k:
                    new_state_dict[k.replace('module.', '')] = v
            
            self.network = VONet(cfg=self.train_cfg)
            self.network.load_state_dict(new_state_dict)

        else:
            self.network = network

        # steal network attributes
        self.DIM = self.network.DIM
        self.RES = self.network.RES
        self.P = self.network.P

        self.network.cuda()
        self.network.eval()

    @property
    def poses(self):
        return self.poses_.view(1, self.N, 7)

    @property
    def patches(self):
        return self.patches_.view(1, self.N*self.M, 3, 3, 3)

    @property
    def intrinsics(self):
        return self.intrinsics_.view(1, self.N, 4)

    @property
    def ix(self):
        return self.index_.view(-1)

    @property
    def imap(self):
        return self.imap_.view(1, self.mem * self.M, self.DIM)

    @property
    def gmap(self):
        return self.gmap_.view(1, self.mem * self.M, 128, 3, 3)

    def get_pose(self, t):
        if t in self.traj:
            return SE3(self.traj[t])

        t0, dP = self.delta[t]
        return dP * self.get_pose(t0)

    def terminate(self):
        """ interpolate missing poses """
        self.traj = {}
        for i in range(self.n):
            self.traj[self.tstamps_[i].item()] = self.poses_[i]

        poses = [self.get_pose(t) for t in range(self.counter)]
        poses = lietorch.stack(poses, dim=0)
        poses = poses.inv().data.cpu().numpy()
        tstamps = np.array(self.tlist, dtype=float)

        return poses, tstamps

    def corr(self, coords, indicies=None):
        """ local correlation volume """
        ii, jj = indicies if indicies is not None else (self.kk, self.jj)
        ii1 = ii % (self.M * self.mem)
        jj1 = jj % (self.mem)
        corr1 = altcorr.corr(self.gmap, self.pyramid[0], coords / 1, ii1, jj1, 3)
        corr2 = altcorr.corr(self.gmap, self.pyramid[1], coords / 4, ii1, jj1, 3)
        return torch.stack([corr1, corr2], -1).view(1, len(ii), -1)

    def reproject(self, indicies=None, poses=None, patches=None, intrinsics=None):
        """ reproject patch k from i -> j """
        (ii, jj, kk) = indicies if indicies is not None else (self.ii, self.jj, self.kk)
        poses = poses if poses is not None else self.poses
        patches = patches if patches is not None else self.patches
        intrinsics = intrinsics if intrinsics is not None else self.intrinsics

        coords = pops.transform(SE3(poses), patches, intrinsics, ii, jj, kk)
        return coords.permute(0, 1, 4, 2, 3).contiguous() # torch.Size([1, 96 * self.n * num_frame_connected, 2, 3, 3])

    def append_factors(self, ii, jj):
        """ add factors to the graph """
        self.jj = torch.cat([self.jj, jj])
        self.kk = torch.cat([self.kk, ii])
        self.ii = torch.cat([self.ii, self.ix[ii]])

        net = torch.zeros(1, len(ii), self.DIM, **self.kwargs)
        self.net = torch.cat([self.net, net], dim=1)

    def remove_factors(self, m):
        """ remove factors from the graph """
        self.ii = self.ii[~m]
        self.jj = self.jj[~m]
        self.kk = self.kk[~m]
        self.net = self.net[:,~m]

    def motion_probe(self):
        """ kinda hacky way to ensure enough motion for initialization """
        kk = torch.arange(self.m-self.M, self.m, device="cuda")
        jj = self.n * torch.ones_like(kk)
        ii = self.ix[kk]

        net = torch.zeros(1, len(ii), self.DIM, **self.kwargs) # torch.Size([1, 96, 384])
        coords = self.reproject(indicies=(ii, jj, kk)) # torch.Size([1, 96, 2, 3, 3])

        with autocast:
            corr = self.corr(coords, indicies=(kk, jj))
            ctx = self.imap[:,kk % (self.M * self.mem)]
            net, (delta, weight, _) = \
                self.network.update(net, ctx, corr, None, ii, jj, kk)

        return torch.quantile(delta.norm(dim=-1).float(), 0.5)

    def motionmag(self, i, j):
        """ compute motion magnitude (mean flow of patches) between frames i and j"""
        k = (self.ii == i) & (self.jj == j)
        ii = self.ii[k]
        jj = self.jj[k]
        kk = self.kk[k]

        flow = pops.flow_mag(SE3(self.poses), self.patches, self.intrinsics, ii, jj, kk, beta=0.5)
        return flow.mean().item()

    def keyframe(self):
        """ remove keyframe if motion is small """
        i = self.n - self.cfg.KEYFRAME_INDEX - 1
        j = self.n - self.cfg.KEYFRAME_INDEX + 1
        m = self.motionmag(i, j) + self.motionmag(j, i)
 
        if m / 2 < self.cfg.KEYFRAME_THRESH:
            k = self.n - self.cfg.KEYFRAME_INDEX
            t0 = self.tstamps_[k-1].item()
            t1 = self.tstamps_[k].item()

            dP = SE3(self.poses_[k]) * SE3(self.poses_[k-1]).inv()
            self.delta[t1] = (t0, dP)

            to_remove = (self.ii == k) | (self.jj == k)
            self.remove_factors(to_remove)

            self.kk[self.ii > k] -= self.M
            self.ii[self.ii > k] -= 1
            self.jj[self.jj > k] -= 1

            for i in range(k, self.n-1):
                self.tstamps_[i] = self.tstamps_[i+1]
                self.colors_[i] = self.colors_[i+1]
                self.poses_[i] = self.poses_[i+1]
                self.patches_[i] = self.patches_[i+1]
                self.intrinsics_[i] = self.intrinsics_[i+1]

                self.imap_[i%self.mem] = self.imap_[(i+1) % self.mem]
                self.gmap_[i%self.mem] = self.gmap_[(i+1) % self.mem]
                self.fmap1_[0,i%self.mem] = self.fmap1_[0,(i+1)%self.mem]
                self.fmap2_[0,i%self.mem] = self.fmap2_[0,(i+1)%self.mem]

            self.n -= 1
            self.m-= self.M

        to_remove = self.ix[self.kk] < self.n - self.cfg.REMOVAL_WINDOW
        self.remove_factors(to_remove)

    def update(self):
        with Timer("other", enabled=self.enable_timing):
            coords = self.reproject()

            with autocast:
                corr = self.corr(coords)
                ctx = self.imap[:,self.kk % (self.M * self.mem)]
                self.net, (delta, weight, _) = \
                    self.network.update(self.net, ctx, corr, None, self.ii, self.jj, self.kk)


            lmbda = torch.as_tensor([1e-4], device="cuda")
            weight = weight.float()
            target = coords[...,self.P//2,self.P//2] + delta.float()

            weight = filter_features(confidences=weight, 
                                     target=target, 
                                     data_shape=(self.ht//4, self.wd//4), 
                                    )

            self.last_weight = weight.clone()

        with Timer("BA", enabled=self.enable_timing):
            t0 = self.n - self.cfg.OPTIMIZATION_WINDOW if self.is_initialized else 1
            t0 = max(t0, 1)

            try:
                fastba.BA(self.poses, self.patches, self.intrinsics, 
                    target, weight, lmbda, self.ii, self.jj, self.kk, t0, self.n, M=self.M, iterations=2, eff_impl=False)
            except Exception as e:
                print(f"WARNING: BA failed...{e}")
            
            points = pops.point_cloud(SE3(self.poses), self.patches[:, :self.m], self.intrinsics, self.ix[:self.m])
            points = (points[...,1,1,:3] / points[...,1,1,3:]).reshape(-1, 3)
            self.points_[:len(points)] = points[:]
                
    def __edges_forw(self):
        r=self.cfg.PATCH_LIFETIME
        t0 = self.M * max((self.n - r), 0)
        t1 = self.M * max((self.n - 1), 0)
        return flatmeshgrid(
            torch.arange(t0, t1, device="cuda"),
            torch.arange(self.n-1, self.n, device="cuda"), indexing='ij')

    def __edges_back(self):
        r=self.cfg.PATCH_LIFETIME
        t0 = self.M * max((self.n - 1), 0)
        t1 = self.M * max((self.n - 0), 0)
        return flatmeshgrid(torch.arange(t0, t1, device="cuda"),
            torch.arange(max(self.n-r, 0), self.n, device="cuda"), indexing='ij')

    def __call__(self, tstamp, input_tensor, intrinsics):
        """ track new frame """
        input_ = preprocess_input(input_tensor=input_tensor)
        
        with autocast:
            fmap, gmap, imap, patches, _, clr = self.network.patchify(
                                            input_=input_,
                                            patches_per_image=self.cfg.PATCHES_PER_FRAME, 
                                            event_bias=self.event_bias,
                                            reinit_hidden=True if tstamp == 0 else False,
                                            )
        if len(input_) > 2:
            _, _, mask = input_
            if not mask and mask is not None:
                # if only events only update the super state but not the VO
                return

        ### update state attributes ###
        self.tlist.append(tstamp)
        self.tstamps_[self.n] = self.counter
        self.intrinsics_[self.n] = intrinsics / self.RES

        self.index_[self.n + 1] = self.n + 1
        self.index_map_[self.n + 1] = self.m + self.M
        
        # color info for visualization
        clr = (clr[0,:,[2,1,0]] + 0.5) * (255.0 / 2)
        self.colors_[self.n] = clr.to(torch.uint8)

        if self.n > 1:
            if self.cfg.MOTION_MODEL == 'DAMPED_LINEAR':
                P1 = SE3(self.poses_[self.n-1])
                P2 = SE3(self.poses_[self.n-2])
                
                xi = self.cfg.MOTION_DAMPING * (P1 * P2.inv()).log()
                tvec_qvec = (SE3.exp(xi) * P1).data
                self.poses_[self.n] = tvec_qvec
            else:
                tvec_qvec = self.poses[self.n-1]
                self.poses_[self.n] = tvec_qvec

        # TODO better depth initialization
        patches[:,:,2] = torch.rand_like(patches[:,:,2,0,0,None,None])
        if self.is_initialized:
            s = torch.median(self.patches_[self.n-3:self.n,:,2])
            patches[:,:,2] = s

        self.patches_[self.n] = patches

        ### update network attributes ###
        # every self.mem=32 times update imap memory with the new imap
        self.imap_[self.n % self.mem] = imap.squeeze()
        self.gmap_[self.n % self.mem] = gmap.squeeze()
        self.fmap1_[:, self.n % self.mem] = F.avg_pool2d(fmap[0], 1, 1)
        self.fmap2_[:, self.n % self.mem] = F.avg_pool2d(fmap[0], 4, 4)

        self.counter += 1        
        if self.n > 0 and not self.is_initialized:
            if self.motion_probe() < 2.0:
                self.delta[self.counter - 1] = (self.counter - 2, Id[0])
                return

        # update number of keyframes and number of total patches
        self.n += 1
        self.m += self.M

        # add edges to the graph
        self.append_factors(*self.__edges_forw())
        self.append_factors(*self.__edges_back())

        # initialize with 8 valid frames and do 12 slam updates
        if self.n == 8 and not self.is_initialized:
            self.is_initialized = True            

            for itr in range(12):
                self.update()
        
        elif self.is_initialized:
            self.update()
            self.keyframe()

        else:
            # if not time=8 and not SLAM initialized do nothing
            pass

    #################### pose prediction ####################

    def efficient_pose_prediction(self, sec_to_pred_future, abs_time, last_keyframe_number,  deg=3, frequency=30):
        next_frame_number = last_keyframe_number +1 # number starting from 1
        next_frame_index = next_frame_number -1 # index starting from 0

        poses = self.poses.clone()
        poses[:,next_frame_index] = motion_bootstrap(
                                            poses=poses[0,...], n=self.n, 
                                            MOTION_MODEL=self.cfg.MOTION_MODEL, 
                                            MOTION_DAMPING=self.cfg.MOTION_DAMPING)

        intrinsics = self.intrinsics.clone()
        intrinsics[:, next_frame_index] = intrinsics[:, next_frame_index-1]
        
        patches = self.patches.clone()
        weights = self.last_weight.clone()

        # add all edges from every other frame (self.ii) to the new virtual frame
        ii, jj, kk, weights_up = add_forward_elements(
                            frame_num=next_frame_number, 
                            patch_extracted_num=self.M, 
                            ii=self.ii, jj=self.jj, 
                            kk=self.kk, ix=self.ix,
                            r=self.cfg.PATCH_LIFETIME, 
                            weights=weights,
                            )

        # Reproject coords in the new virtual frame according to curr motion assumption
        coords = self.reproject(
            indicies=(ii, jj, kk), poses=poses, patches=patches, intrinsics=intrinsics
            )


    def predict_future_pose(self, sec_to_pred_future, abs_time, last_keyframe_number,  deg=3, frequency=30):
        #self.n  # add new virtual keyframe
        #last_keyframe_number = self.n
        next_frame_number = last_keyframe_number +1 # number starting from 1
        next_frame_index = next_frame_number -1 # index starting from 0

        poses = self.poses.clone()
        # TODO: use bootstrap from the last known poses not from the predicted one
        # i.e. substitute self.n with last_keyframe_number
        poses[:,next_frame_index] = motion_bootstrap(
                                            poses=poses[0,...], n=self.n, 
                                            MOTION_MODEL=self.cfg.MOTION_MODEL, 
                                            MOTION_DAMPING=self.cfg.MOTION_DAMPING)

        intrinsics = self.intrinsics.clone()
        intrinsics[:, next_frame_index] = intrinsics[:, next_frame_index-1]
        
        patches = self.patches.clone()
        weights = self.last_weight.clone()

        # add all edges from every other frame (self.ii) to the new virtual frame
        ii, jj, kk, weights_up = add_forward_elements(
                            frame_num=next_frame_number, 
                            patch_extracted_num=self.M, 
                            ii=self.ii, jj=self.jj, 
                            kk=self.kk, ix=self.ix,
                            r=self.cfg.PATCH_LIFETIME, 
                            weights=weights,
                            )

        # Reproject coords in the new virtual frame according to curr motion assumption
        coords = self.reproject(indicies=(ii, jj, kk), poses=poses, patches=patches, intrinsics=intrinsics)
        
        if self.patch_dict_ is None:
            self.patch_dict_ = compute_patch_track__(
                                coords=coords, 
                                ii=ii, jj=jj, kk=kk, 
                                image_to_proj=next_frame_index
                                )
        if self.patches_models is None:
            self.patches_models = fit_model_patch_track(
                                next_frame_index=next_frame_index, 
                                patch_dict=self.patch_dict_, 
                                img_to_keyframe_map=self.tstamps_, 
                                ii=ii, jj=jj, 
                                data_shape=(self.ht, self.wd), 
                                frequency=frequency, deg=deg)
        
        target, updated_weight = predict_patch_on_model(
                                patch_models=self.patches_models, 
                                step_to_pred_future=sec_to_pred_future, 
                                frequency=frequency, 
                                next_frame_index=next_frame_index, 
                                coords=coords, weights=weights_up, 
                                ii=ii, jj=jj, kk=kk)
        
        t0 = max(next_frame_number - self.cfg.OPTIMIZATION_WINDOW if self.is_initialized else 1, 1)
        t1 = next_frame_number

        #self.patches : coords of the patches extracted, coords: coordinate of their reprojection 
        try:
            fastba.BA(poses=poses, patches=patches, intrinsics=intrinsics,
                       target=target, weight=updated_weight, lmbda=self.lmbda, 
                       ii=ii, jj=jj, kk=kk, t0=t0, t1=t1, iterations=2, M=self.M, eff_impl=False)
        except Exception:
            print("WARNING: BA failed...")

        # update network attributes to use self.terminate interpolation
        self.update_attributes(abs_time=abs_time, next_frame_index=next_frame_index, poses=poses)
    

    def update_attributes(self, abs_time, next_frame_index, poses):
        """ update network attributes to use self.terminate interpolation """
        assert self.tstamps_[self.n-1] != 0
        
        self.tstamps_[self.n] = abs_time
        self.poses_[self.n] = poses[:, next_frame_index]
        self.tlist.append(abs_time)
        self.counter += 1
        self.n += 1 # to allow to see the last pose in terminate
    
    def remove_attributes(self):
        """ remove network updates for normal prediction """
        self.n -= 1
        self.counter -= 1
        self.tlist.pop()
        self.poses_[self.n] = torch.zeros(7, dtype=torch.float, device="cuda")
        self.poses_[:,6] = 1.0
        self.tstamps_[self.n] = 0

