import numpy as np
from skimage.morphology import label
from scipy import ndimage
import scipy.io as scio

def get_rect_of_mask(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax
    
def get_size_of_mask(img):
    if np.max(img) == 0:
        return 0
    rmin, rmax, cmin, cmax = get_rect_of_mask(img)
    return max([rmax - rmin, cmax - cmin])
    
def remove_overlaps(instances, scores):
    if len(instances) == 0:
        return [], []
    lab_img = np.zeros(instances[0].shape, dtype=np.int32)
    for i, instance in enumerate(instances):
        lab_img = np.maximum(lab_img, instance * (i + 1))
    instances = []
    new_scores = []
    for i in range(1, lab_img.max() + 1):
        instance = (lab_img == i).astype(np.bool)
        if np.max(instance) == 0:
            continue
        instances.append(instance)
        new_scores.append(scores[i - 1])
    return instances, new_scores
    
def post_proc(output, cutoff=0.5, cutoff_instance_max=0.3, cutoff_instance_avg=0.2, post_dilation_iter=2, post_fill_holes=True):
    """
    Split 1-channel merged output for instance segmentation
    :param cutoff:
    :param output: (h, w, 1) segmentation image
    :return: list of (h, w, 1). instance-aware segmentations.
    """
    # The post processing function 'post_proc' is borrowed from the author of CIA-Net.
    
    cutoffed = output > cutoff
    lab_img = label(cutoffed, connectivity=1)
    instances = []
    # pdb.set_trace()
    for i in range(1, lab_img.max() + 1):
        instances.append((lab_img == i).astype(np.bool))

    filtered_instances = []
    scores = []
    for instance in instances:
        # TODO : max or avg?
        instance_score_max = np.max(instance * output)    # score max
        if instance_score_max < cutoff_instance_max:
            continue
        instance_score_avg = np.sum(instance * output) / np.sum(instance)   # score avg
        if instance_score_avg < cutoff_instance_avg:
            continue
        filtered_instances.append(instance)
        scores.append(instance_score_avg)
    instances = filtered_instances

    # dilation
    instances_tmp = []
    if post_dilation_iter > 0:
        for instance in filtered_instances:
            instance = ndimage.morphology.binary_dilation(instance, iterations=post_dilation_iter)
            instances_tmp.append(instance)
        instances = instances_tmp

    # sorted by size
    sorted_idx = [i[0] for i in sorted(enumerate(instances), key=lambda x: get_size_of_mask(x[1]))]
    instances = [instances[x] for x in sorted_idx]
    scores = [scores[x] for x in sorted_idx]

    # make sure there are no overlaps
    # todo: this dataset gt has overlap, so do not use this func
    instances, scores = remove_overlaps(instances, scores)

    # fill holes
    if post_fill_holes:
        instances = [ndimage.morphology.binary_fill_holes(i) for i in instances]
    
    # instances = [np.expand_dims(i, axis=2) for i in instances]
    # scores = np.array(scores)
    # scores = np.expand_dims(scores, axis=1)
    lab_img = np.zeros(instances[0].shape, dtype=np.int32)
    for i, instance in enumerate(instances):
        lab_img = np.maximum(lab_img, instance * (i + 1))
        
    return lab_img