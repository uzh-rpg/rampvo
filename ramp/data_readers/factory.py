#from .tartan import TartanAir # RGBD-Dataset
from .TartanEvent import TartanEvent
from torch.utils.data import ConcatDataset

def dataset_factory(dataset_list, **kwargs):
    """ create a combined dataset """

    dataset_map = { 
        'tartanEvent': (TartanEvent, ),
    }

    db_list = []
    for key in dataset_list:
        # cache datasets for faster future loading
        db = dataset_map[key][0](**kwargs)

        print("Dataset {} has {} images".format(key, len(db)))
        db_list.append(db)

    return ConcatDataset(db_list)
