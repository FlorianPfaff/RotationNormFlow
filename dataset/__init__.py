from dataset.dataset_pascal import get_dataloader_pascal3d
from dataset.dataset_symsol import get_dataloader_symsol
from dataset.dataset_raw import get_dataloader_raw


def get_dataloader(dataset, phase, config):
    if dataset == "pascal3d":
        return get_dataloader_pascal3d(phase, config)
    elif dataset == "symsol":
        return get_dataloader_symsol(phase, config)
    elif dataset == "raw":
        return get_dataloader_raw(phase, config)
    
    raise NotImplementedError()