import math
from typing import List, Mapping, Optional, Tuple, cast
import numpy as np
import scipy.ndimage as ndi


def _enlarge_slice(slices: Tuple[slice], padding):
    # Enlarge slices by provided padding
    return tuple(slice(max(0, s.start - padding), s.stop + padding) for s in slices)


def _windowed_distance_outside(
    mask: np.ndarray, max_distance: Optional[int] = None
) -> np.ndarray:
    # Calculate distance field outside of the provided background given a maximum distance
    if max_distance is None:
        return ndi.distance_transform_edt(~mask)  # type: ignore

    (slices,) = ndi.find_objects(mask, 1)
    slices = _enlarge_slice(slices, max_distance + 1)

    dist_sliced: np.ndarray = ndi.distance_transform_edt(~mask[slices])  # type: ignore

    result = np.full(mask.shape, dist_sliced.max())
    result[slices] = dist_sliced
    return result


def merge_labels(
    labels: np.ndarray,
    index: Optional[List[int]] = None,
    max_distance: Optional[float] = None,
    path_tolerance: float = 5,
    return_merge_distances=False,
    labels_out=None,
):
    """
    Merge neighboring labels.

    If two labeled segments are closer than max_distance, their closest points are connected and their labels are unified.

    Args:
        labels (ndarray): Label image.
        index (list, optional): List of labels to process. Defaults to all labels.
        max_distance (float, optional): Maximum merge distance. Merge all by default.
        path_tolerance (float, optional): Influences the width of the bridges.
        return_merge_distances (bool, optional): Return the distances at which each label in index[1:] was merged.
        labels_out (ndarray, optional): Output buffer. By default, a copy of labels is used.

    Returns:
        labels_out (ndarray): Relabeled image.
        merge_distances (list, optional): Distances at which each label in index[1:] was merged.
    """

    if index is None:
        unique_labels = np.unique(labels)
        index = cast(List[int], unique_labels[unique_labels > 0].tolist())

    if len(index) < 2:
        return (labels, []) if return_merge_distances else labels

    if labels_out is None:
        labels_out = labels.copy()

    # First label
    l0 = index.pop(0)
    mask = labels == l0
    labels_out[mask] = l0

    max_distance_int = math.ceil(max_distance) if max_distance is not None else None

    # Distance map: Distance to the nearest object
    distmap = _windowed_distance_outside(mask, max_distance_int)
    max_dist = distmap.max()

    # Label map: Label of the nearest object
    labelmap = np.full(labels.shape, l0, dtype=labels.dtype)

    merge_distances = []

    while index:
        # Find element in index that is closest to the current object
        min_idx = np.argmin([distmap[labels == l].min(initial=max_dist) for l in index])
        cur_l = index.pop(min_idx)

        # Calculate distance transform for cur_l
        cur_distmap = _windowed_distance_outside(labels == cur_l, max_distance_int)

        # Calculate sum
        sum_distmap = distmap + cur_distmap

        merge_dist = sum_distmap.min()

        if max_distance is not None and merge_dist > max_distance:
            # This is already the nearest merger, so we can stop
            break

        fill_mask = (labels == cur_l) | (sum_distmap <= merge_dist + path_tolerance)

        merge_distances.append(merge_dist)

        # Replace-label
        replace_label = np.unique(labelmap[fill_mask]).item()

        # Update result
        labels_out[fill_mask] = replace_label

        # Update labelmap and distmap
        update_mask = cur_distmap < distmap
        labelmap[update_mask] = replace_label
        distmap[update_mask] = cur_distmap[update_mask]

    return (labels_out, merge_distances) if return_merge_distances else labels_out
