import pdb

import funcy
import numpy as np
import torch
import torch.nn.functional as F

from copy import deepcopy
from shapely.geometry import LineString, Point
from vt_tools import (
    ARYTENOID_CARTILAGE,
    EPIGLOTTIS,
    LOWER_INCISOR,
    LOWER_LIP,
    PHARYNX,
    SOFT_PALATE_MIDLINE,
    THYROID_CARTILAGE,
    TONGUE,
    UPPER_INCISOR,
    UPPER_LIP,
    VOCAL_FOLDS
)
from vt_tools.bs_regularization import regularize_Bsplines
from vt_tools.metrics import distance_matrix, euclidean
from vt_tools.reconstruct_snail import reconstruct_snail_from_midline

from .helpers import load_articulator_array


def mm_to_percent(value_mm, resolution=136, pixel_spacing=1.412):
    return value_mm / (resolution * pixel_spacing)


def percent_to_mm(value_percent, resolution = 136, pixel_spacing=1.412):
    return value_percent * resolution * pixel_spacing


SNAIL_PARAMETERS = {
    SOFT_PALATE_MIDLINE: {
        "width_int": mm_to_percent(1.5),
        "width_ext": mm_to_percent(3.5),
        "width_apex_int": mm_to_percent(2.5),
        "width_apex_ext": mm_to_percent(1.)
    },
    EPIGLOTTIS: {
        "width_int": mm_to_percent(2.5),
        "width_ext": mm_to_percent(2.5),
        "width_apex_int": mm_to_percent(1.5),
        "width_apex_ext": mm_to_percent(1.5)
    }
}


def intersection(arr1, arr2):
    """
    Finds the intersection points between two arrays.

    Args:
    arr_1 (np.ndarray): (N,) shaped array
    arr_2 (np.ndarray): (M,) shaped array
    """
    line_string_1 = LineString(arr1)
    line_string_2 = LineString(arr2)

    is_contact = line_string_1.intersects(line_string_2)

    if not is_contact:
        return []

    contact = line_string_1.intersection(line_string_2)
    contact_list = (
        [(contact.x, contact.y)]
        if isinstance(contact, Point) else
        [(p.x, p.y) for p in contact.geoms]
    )

    return contact_list


def closest_point(arr_1, arr_2):
    """
    Calculates the closest point between two np.ndarray and return the indices for each array

    Args:
    arr_1 (np.ndarray): (N,) shaped array
    arr_2 (np.ndarray): (M,) shaped array
    """
    dist_mtx = distance_matrix(arr_1, arr_2)
    _, ncols = dist_mtx.shape

    amin_array_1 = dist_mtx.argmin() // ncols
    amin_array_2 = dist_mtx.argmin() % ncols

    return amin_array_1, amin_array_2


def interpolate_points(pt1, pt2):
    """
    Calculate the middle x and y coordinates of two points.

    Args:
    pt1 (Tuple): First point.
    pt2 (Tuple): Second point.
    """
    x1, y1 = pt1
    x2, y2 = pt2

    new_x = min(x1, x2) + abs(x1 - x2) / 2
    new_y = min(y1, y2) + abs(y1 - y2) / 2

    return new_x, new_y


def upsample_curve(curve, approx_n_samples):
    new_curve = deepcopy(curve)

    consecutive_points_distances = funcy.lmap(
        lambda ps: euclidean(*ps),
        zip(new_curve[:-1], new_curve[1:])
    )

    linear_curve_length = np.sum(consecutive_points_distances)
    segment_length = linear_curve_length / (approx_n_samples - 1)

    curve_segments = list(zip(consecutive_points_distances, new_curve[:-1], new_curve[1:]))
    while any([d > segment_length for d, _, _ in curve_segments]):
        interp_curve = []
        for dist, pt1, pt2 in curve_segments:
            if dist <= segment_length:
                interp_curve.append((pt1, pt2))
            else:
                new_point = interpolate_points(pt1, pt2)
                interp_curve.append((pt1, new_point))
                interp_curve.append((new_point, pt2))
        new_curve = [pt1 for (pt1, _) in interp_curve]

        consecutive_points_distances = funcy.lmap(
            lambda ps: euclidean(*ps),
            zip(new_curve[:-1], new_curve[1:])
        )
        curve_segments = list(zip(consecutive_points_distances, new_curve[:-1], new_curve[1:]))

    return np.array(new_curve)


def load_articulators_arrays(articulators_filepaths, snail_parameters=None, norm_value=None):
    """
    Load the articulators, processing the snail structures accordingly.

    Args:
    articulators_filepaths (dict): Dictionary containing the articulator name in the key
    and the filepath of the .npy file.
    snail_parameters (dict): Dictionary with the snail parameters for each snail articulator in
    articulators_filpaths.
    norm_value (float): Value to normalize the articulators array.
    """
    if snail_parameters is None:
        snail_parameters = SNAIL_PARAMETERS

    # Load soft palate and reconstruct snail
    fp_soft_palate = articulators_filepaths[SOFT_PALATE_MIDLINE]
    midline_soft_palate = load_articulator_array(fp_soft_palate, norm_value)
    params_soft_palate = snail_parameters[SOFT_PALATE_MIDLINE]
    snail_soft_palate = reconstruct_snail_from_midline(midline_soft_palate, **params_soft_palate)

    # Load epiglottis and reconstruct snail
    fp_epiglottis = articulators_filepaths[EPIGLOTTIS]
    midline_epiglottis = load_articulator_array(fp_epiglottis, norm_value)
    params_epiglottis = snail_parameters[EPIGLOTTIS]
    snail_epiglottis = reconstruct_snail_from_midline(midline_epiglottis, **params_epiglottis)

    # Load non-snail articulators
    articulators_arrays = {}
    for articulator, fp_articulator in articulators_filepaths.items():
        if articulator not in snail_parameters:
            articulators_arrays[articulator] = load_articulator_array(fp_articulator, norm_value)

    articulators_arrays[SOFT_PALATE_MIDLINE] = snail_soft_palate
    articulators_arrays[EPIGLOTTIS] = snail_epiglottis

    return articulators_arrays


def find_lip_end(lip_array):
    """
    Finds the point where the absissas starts decreasing.
    """
    half = len(lip_array) // 2
    lip_array_0 = lip_array[half:-1]
    lip_array_1 = lip_array[half+1:]

    offsets = list(enumerate(zip(lip_array_0, lip_array_1)))
    decreasing_absissas = funcy.lfilter(lambda t: t[1][0][0] < t[1][1][0], offsets)
    if len(decreasing_absissas) > 0:
        idx, (_, _) = min(decreasing_absissas, key=lambda t: t[0])
        idx += half
    else:
        idx = -1

    return idx


def shapes_to_articulators_dict(shapes, articulators=None, regularize=False):
    """
    Transform the predicted shapes into the articulators dict.

    Args:
    shapes (torch.tensor): Outputs of shapes generation of shape (seq_len, 11, 2, 50).
    articulators (list): List of articulators that are present in shapes.
    regularize (bool): If should apply bspline regularization before outputs.

    Return:
    (list): List of articulators dicts.
    """
    if articulators is None:
        articulators = sorted([
            ARYTENOID_CARTILAGE, EPIGLOTTIS, LOWER_INCISOR, LOWER_LIP,
            PHARYNX, SOFT_PALATE_MIDLINE, THYROID_CARTILAGE, TONGUE,
            UPPER_INCISOR, UPPER_LIP, VOCAL_FOLDS
        ])

    articulators_dicts = [
        {articulators[i]: articulator_array.T
        for i, articulator_array in enumerate(shape)}
        for shape in shapes
    ]

    if regularize:
        articulators_dicts = [
            {
                articulator: np.array(regularize_Bsplines(articulator_array, degree=2)).T
                for articulator, articulator_array in articulator_dict.items()
            } for articulator_dict in articulators_dicts
        ]

    for articulator_dict in articulators_dicts:
        params_soft_palate = SNAIL_PARAMETERS[SOFT_PALATE_MIDLINE]
        snail_soft_palate = reconstruct_snail_from_midline(
            articulator_dict[SOFT_PALATE_MIDLINE], **params_soft_palate
        )
        snail_soft_palate = np.array(regularize_Bsplines(snail_soft_palate, degree=2)).T
        articulator_dict[SOFT_PALATE_MIDLINE] = snail_soft_palate

        params_epiglottis = SNAIL_PARAMETERS[EPIGLOTTIS]
        snail_epiglottis = reconstruct_snail_from_midline(
            articulator_dict[EPIGLOTTIS], **params_epiglottis
        )
        snail_epiglottis = np.array(regularize_Bsplines(snail_epiglottis, degree=2)).T
        articulator_dict[EPIGLOTTIS] = snail_epiglottis

    return articulators_dicts


def build_internal_wall(articulators_arrays):
    """
    Connect the articulators to produce the vocal tract internal wall.

    Args:
    articulators_dict (Dict): Dictionary containing the articulator name in the key and the filepath
    of the .npy file.
    """
    order = [EPIGLOTTIS, TONGUE, LOWER_INCISOR, LOWER_LIP]

    # The internal wall starts at the left-most point in the vocal folds
    start_point = min(articulators_arrays[VOCAL_FOLDS], key=lambda t: t[0])
    internal_wall = np.array([start_point])

    vocal_folds = articulators_arrays[VOCAL_FOLDS]
    epiglottis = articulators_arrays[EPIGLOTTIS]
    idx1, idx2 = closest_point(vocal_folds, epiglottis)
    internal_wall = epiglottis[idx2:]

    for next_art in order[1:]:
        arr1 = internal_wall
        arr2 = articulators_arrays[next_art]

        points = intersection(arr1, arr2) if len(arr1) > 1 and len(arr2) > 1 else []
        if len(points) == 0:
            contact = np.zeros(shape=(0, 2))
            idx1, idx2 = closest_point(arr1, arr2)
        else:
            contact = sorted(points, key=lambda t: t[1])[0]
            contact = np.array([contact])
            idx1, _ = closest_point(internal_wall, contact)
            idx2, _ = closest_point(arr2, contact)

        arr1_cat = internal_wall[:idx1+1, :]

        if next_art == LOWER_LIP:
            lip_end = find_lip_end(arr2)
            arr2 = arr2[:lip_end]

        arr2_cat = arr2[idx2:]
        internal_wall = np.concatenate([arr1_cat, contact, arr2_cat])

    internal_wall = upsample_curve(internal_wall, approx_n_samples=300)
    internal_wall = torch.from_numpy(internal_wall).T.unsqueeze(dim=0)
    internal_wall = F.interpolate(internal_wall, size=100, mode="linear", align_corners=True)
    internal_wall = internal_wall.squeeze(dim=0).T.numpy()

    return internal_wall


def build_external_wall(articulators_arrays):
    """
    Connect the articulators to produce the vocal tract external wall.

    Args:
    articulators_dict (Dict): Dictionary containing the articulator name in the key and the filepath
    of the .npy file.
    """
    order = [ARYTENOID_CARTILAGE, PHARYNX, SOFT_PALATE_MIDLINE, UPPER_INCISOR, UPPER_LIP]

    should_flip = lambda art: art in [ARYTENOID_CARTILAGE, SOFT_PALATE_MIDLINE]

    # The internal wall starts at the left-most point in the vocal folds
    vocal_folds = articulators_arrays[VOCAL_FOLDS]
    arytenoid_cartilage = np.flip(articulators_arrays[ARYTENOID_CARTILAGE], axis=0)
    idx1, idx2 = closest_point(vocal_folds, arytenoid_cartilage)
    external_wall = arytenoid_cartilage[idx2:]

    for next_art in order[1:]:
        arr1 = external_wall
        arr2 = articulators_arrays[next_art]

        points = intersection(arr1, arr2) if len(arr1) > 1 and len(arr2) > 1 else []
        if len(points) == 0:
            contact = np.zeros(shape=(0, 2))
            idx1, idx2 = closest_point(arr1, arr2)
        else:
            contact = sorted(points, key=lambda t: t[1])[-1]
            contact = np.array([contact])
            idx1, _ = closest_point(external_wall, contact)
            idx2, _ = closest_point(arr2, contact)

        arr1_cat = external_wall[:idx1+1, :]

        if next_art == UPPER_LIP:
            lip_end = find_lip_end(arr2)
            arr2 = arr2[:lip_end]

        arr2_cat = arr2[idx2:] if not should_flip(next_art) else np.flip(arr2[:idx2+1], axis=0)
        external_wall = np.concatenate([arr1_cat, contact, arr2_cat])

    external_wall = upsample_curve(external_wall, approx_n_samples=300)
    external_wall = torch.from_numpy(external_wall).T.unsqueeze(dim=0)
    external_wall = F.interpolate(external_wall, size=100, mode="linear", align_corners=True)
    external_wall = external_wall.squeeze(dim=0).T.numpy()

    return external_wall


def generate_vocal_tract_tube(articulators_dict, load=True, norm_value=None):
    """
    Connect the articulators to produce a single shape for the entire vocal tract.

    Args:
    articulators_dict (Dict): Dictionary containing the articulator name in the key and the filepath
    of the .npy file.
    load (bool): If should load the articulators
    norm_value (float): Value to normalize the articulators array. Only used if load == True.
    """
    if load:
        articulators_arrays = load_articulators_arrays(articulators_dict, norm_value=norm_value)
    else:
        articulators_arrays = articulators_dict

    internal_wall = build_internal_wall(articulators_arrays)
    external_wall = build_external_wall(articulators_arrays)

    return internal_wall, external_wall
