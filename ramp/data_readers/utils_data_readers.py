import torch
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def set_random_sample_to_zero(
        images, events, img_to_zero_perc=0.5, datacouple_perc=0.2
    ):
    flatten_images = images.reshape(images.shape[0], -1)
    flatten_events = events.reshape(images.shape[0], -1)

    # put to zero only common elements to avoid the case of a indices couple missing
    non_zero_images = set(torch.where((flatten_images != 0).any(-1))[0].tolist())
    non_zero_events = set(torch.where((flatten_events != 0).any(-1))[0].tolist())
    common_elements = non_zero_images.intersection(non_zero_events)

    # Compute the number of tensors to set to zero among the common non zero elements
    non_zero_set_len = len(common_elements)
    num_images_to_zero = int(non_zero_set_len * img_to_zero_perc)
    num_images_to_retain = int(non_zero_set_len * datacouple_perc)

    # events indices must be (initially) the complement to images indices
    zero_images = set(random.sample(common_elements, num_images_to_zero))
    zero_events = common_elements - zero_images

    # then we remove the indices couple to retain
    retain_indices = set(random.sample(common_elements, num_images_to_retain))
    zero_images_with_retain = zero_images - retain_indices
    zero_events_with_retain = zero_events - retain_indices

    # Set the corresponding tensor elements to zero
    events[list(zero_images_with_retain)] = 0
    images[list(zero_events_with_retain)] = 0

    return events, images


def set_random_sequence_to_zero(
    images,
    events,
    perc_to_drop_img=0.4,
    perc_to_drop_evs=0.4,
    perc_to_drop_none=0.2,
):
    # perc of the full data to drop, imgs_perc --> how often drop images
    weights = [perc_to_drop_evs, perc_to_drop_img, perc_to_drop_none]
    assert sum(weights) == 1
    choices = ["drop-evs", "drop-imgs", "drop-none"]

    flatten_images = images.reshape(images.shape[0], -1)
    flatten_events = events.reshape(images.shape[0], -1)
    non_zero_images_num = len(
        torch.where((flatten_images != 0).any(-1))[0].tolist()
    )
    non_zero_events_num = len(
        torch.where((flatten_events != 0).any(-1))[0].tolist()
    )

    # put to zero only when all images & events are present to avoid when a indices couple is missing
    if non_zero_images_num != non_zero_events_num:
        return events, images
    sample = random.choices(choices, weights=weights, k=1)[0]
    if sample == "drop-evs":
        return torch.zeros_like(events), images
    if sample == "drop-imgs":
        return events, torch.zeros_like(images)
    return events, images


import torch as th

def dictionary_of_numpy_arrays_to_tensors(sample):
    """Transforms dictionary of numpy arrays to dictionary of tensors."""
    if isinstance(sample, dict):
        return {
            key: dictionary_of_numpy_arrays_to_tensors(value)
            for key, value in sample.items()
        }
    if isinstance(sample, np.ndarray):
        if len(sample.shape) == 2:
            return th.from_numpy(sample).float().unsqueeze(0)
        else:
            return th.from_numpy(sample).float()
    return sample

class EventSequenceToVoxelGrid_Pytorch(object):
    # Source: https://github.com/uzh-rpg/rpg_e2vid/blob/master/utils/inference_utils.py#L480
    def __init__(self, num_bins, gpu=False, gpu_nr=0, normalize=True, forkserver=True):
        if forkserver:
            try:
                th.multiprocessing.set_start_method('forkserver')
            except RuntimeError:
                pass
        self.num_bins = num_bins
        self.normalize = normalize
        if gpu:
            if not th.cuda.is_available():
                print('WARNING: There\'s no CUDA support on this machine!')
            else:
                self.device = th.device('cuda:' + str(gpu_nr))
        else:
            self.device = th.device('cpu')

    def __call__(self, event_sequence):
        """
        Build a voxel grid with bilinear interpolation in the time domain from a set of events.
        :param events: a [N x 4] NumPy array containing one event per row in the form: [timestamp, x, y, polarity]
        :param num_bins: number of bins in the temporal axis of the voxel grid
        :param width, height: dimensions of the voxel grid
        :param device: device to use to perform computations
        :return voxel_grid: PyTorch event tensor (on the device specified)
        """

        events = event_sequence.features.astype('float')

        width = event_sequence.image_width
        height = event_sequence.image_height

        assert (events.shape[1] == 4)
        assert (self.num_bins > 0)
        assert (width > 0)
        assert (height > 0)

        with th.no_grad():

            events_torch = th.from_numpy(events)
            # with DeviceTimer('Events -> Device (voxel grid)'):
            events_torch = events_torch.to(self.device)

            # with DeviceTimer('Voxel grid voting'):
            voxel_grid = th.zeros(self.num_bins, height, width, dtype=th.float32, device=self.device).flatten()

            # normalize the event timestamps so that they lie between 0 and num_bins
            last_stamp = events_torch[-1, 0]
            first_stamp = events_torch[0, 0]

            assert last_stamp.dtype == th.float64, 'Timestamps must be float64!'
            # assert last_stamp.item()%1 == 0, 'Timestamps should not have decimals'

            deltaT = last_stamp - first_stamp

            if deltaT == 0:
                deltaT = 1.0

            events_torch[:, 0] = (self.num_bins - 1) * (events_torch[:, 0] - first_stamp) / deltaT
            ts = events_torch[:, 0]
            xs = events_torch[:, 1].long()
            ys = events_torch[:, 2].long()
            pols = events_torch[:, 3].float()
            pols[pols == 0] = -1  # polarity should be +1 / -1


            tis = th.floor(ts)
            tis_long = tis.long()
            dts = ts - tis
            vals_left = pols * (1.0 - dts.float())
            vals_right = pols * dts.float()

            valid_indices = tis < self.num_bins
            valid_indices &= tis >= 0

            if events_torch.is_cuda:
                datatype = th.cuda.LongTensor
            else:
                datatype = th.LongTensor

            voxel_grid.index_add_(dim=0,
                index=(xs[valid_indices] + ys[valid_indices] * width + tis_long[valid_indices] * width * height).type(datatype),
                source=vals_left[valid_indices])

            valid_indices = (tis + 1) < self.num_bins
            valid_indices &= tis >= 0

            voxel_grid.index_add_(dim=0,
                                  index=(xs[valid_indices] + ys[valid_indices] * width + (tis_long[valid_indices] + 1) * width * height).type(datatype),
                                  source=vals_right[valid_indices])

            voxel_grid = voxel_grid.view(self.num_bins, height, width)

        if self.normalize:
            mask = th.nonzero(voxel_grid, as_tuple=True)
            if mask[0].size()[0] > 0:
                mean = voxel_grid[mask].mean()
                std = voxel_grid[mask].std()
                if std > 0:
                    voxel_grid[mask] = (voxel_grid[mask] - mean) / std
                else:
                    voxel_grid[mask] = voxel_grid[mask] - mean

        return voxel_grid

import os
def plot_video_aligment(image_list, ev, event_inds, start_ind=None, event_stream_size=None, scene_id=None):
    i0_, i1_ = event_inds
    for i in range(len(image_list)):
        if start_ind is not None and i < start_ind:
            continue
        plt.clf()
        if event_stream_size is not None:
            i0 = i1_[i] - event_stream_size
        else:
            i0 = i0_[i]
        i1 = i1_[i]
        events = ev.get_between_idx(i0, i1)
        image = np.array(Image.open(image_list[i]))
        plot_item = events.render(image)
        plt.imshow(plot_item)
        if scene_id is not None:
            file_format = "DEBUG/{}/video_aligment_{:05d}.png"
            os.makedirs(os.path.dirname(file_format.format(scene_id, i)), exist_ok=True)
            plt.savefig(file_format.format( scene_id, i))
        else:
            file_format = "DEBUG/video_aligment_{:05d}.png"
            plt.savefig(file_format.format(i))

