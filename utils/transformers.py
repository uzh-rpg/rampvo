import numpy as np
import torch as th
from data import Events


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
                print('Warning: There\'s no CUDA support on this machine!')
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


class EventToStack_Numpy(object):
    def __init__(self, num_bins):
        self.num_bins = num_bins

    def _draw_xy_to_voxel_grid(self, voxel_grid, x, y, b, value):
        if x.dtype == np.uint16:
            self._draw_xy_to_voxel_grid_int(voxel_grid, x, y, b, value)
            return

        x_int = x.astype("int32")
        y_int = y.astype("int32")
        for xlim in [x_int, x_int + 1]:
            for ylim in [y_int, y_int + 1]:
                weight = _bil_w(x, xlim) * _bil_w(y, ylim)
                self._draw_xy_to_voxel_grid_int(voxel_grid, xlim, ylim, b, weight * value)

    def _draw_xy_to_voxel_grid_int(self, voxel_grid, x, y, b, value):
        B, H, W = voxel_grid.shape
        mask = (x >= 0) & (y >= 0) & (x < W) & (y < H)
        np.add.at(voxel_grid, (b[mask], y[mask], x[mask]), value[mask])

    def __call__(self, events: Events) -> np.array:
        voxel_grid = np.zeros((self.num_bins, events.height, events.width), np.float32)

        if len(events) < 2:
            return voxel_grid

        # normalize the event timestamps so that they lie between 0 and num_bins
        t_norm = (self.num_bins * np.arange(len(events), dtype="float32") / len(events)).astype("int32")
        self._draw_xy_to_voxel_grid(voxel_grid, events.x, events.y, t_norm, events.p)

        voxel_grid = voxel_grid.astype("int8")

        return voxel_grid
