
import higra as hg

import matplotlib.pyplot as plt
from typing import Tuple, List
from skimage.data import binary_blobs, astronaut
from skimage.color import rgb2gray
from skimage.exposure import rescale_intensity
from skimage.transform import resize
from scipy.ndimage import rotate
from skimage.io import imread

import numpy as np
import cv2

from tqdm import tqdm

NOISE = 0
GRADIENT = 1
BLOB = 2
PHOTO = 3
BRAIN = 4
FULL_BRAIN = 5

def xy_from_index(index: int, size: Tuple[int, int]) -> Tuple[int, int]:
    width = size[1]
    x = index % width
    y = int(index/width)
    return x, y

def surprint(img: np.ndarray, pattern: np.ndarray, color: List[int] = [125, 0, 255]) -> np.ndarray:
    """
    Print a pattern on a grascale image.

    Args:
        img (np.ndarray): grayscale image
        pattern (np.ndarray): pattern as binary image. 1 for pattern and 0 otherwise
        color (List[int], optional): RGB color for pattern. Defaults to [125, 0, 255].

    Returns:
        np.ndarray: RGB image with the pattern printed above the grayscale input image.
    """
    assert(len(img.shape) == 2)
    assert(len(pattern.shape) == 2)

    if len(img.shape) != 3:
        im = np.repeat(img.reshape(*img.shape, 1), 3, axis=2)
        im = ((im/np.max(im))*255).astype(np.uint8)
    else:
        im = np.copy(img)

    im_out = np.copy(im)
    im_out[np.where(pattern > 0)] = color
    return im_out

def get_image(size: int, mode: str = NOISE, dtype = np.int16, **kwargs) -> np.ndarray:
    image = None

    if mode is NOISE:
        np.random.seed(42)
        image = np.random.randint(0, 256, size)

    if mode is GRADIENT:
        angle = 0
        new_size = size[0]
        if 'angle' in kwargs:
            if kwargs['angle'] == 45:
                angle = 45
                new_size = int(np.ceil(np.sin(np.deg2rad(angle))*size[0]*2))

        x = np.linspace(0, 2, new_size)
        x = np.tile(x, (new_size, 1))
        image = rotate(x, angle=np.rad2deg(angle))
        if new_size != size[0]:
            crop = (image.shape[0]-size[0])//2
            image = image[crop:crop+size[0], crop:crop+size[0]]

    if mode is BLOB:
        sigma = kwargs['sigma'] if 'sigma' in kwargs else 0.1
        blob_size_fraction = kwargs['blob_size_fraction'] if 'blob_size_fraction' in kwargs else 0.02
        volume_fraction = kwargs['volume_fraction'] if 'volume_fraction' in kwargs else 0.3

        image = binary_blobs(length=size[0], seed=1, blob_size_fraction=blob_size_fraction,
                             volume_fraction=volume_fraction).astype(np.float64)

        rng = np.random.default_rng(seed=42)
        image += rng.normal(loc=0, scale=sigma, size=image.shape)

    if mode is PHOTO:
        image = astronaut()
        image = rgb2gray(image)
        image = resize(image, size,
                       anti_aliasing=True)

    if mode is BRAIN:
        image = imread(r'images/brain_800-1800_4000-5000.png')
        sigma = kwargs['sigma'] if 'sigma' in kwargs else 0
        rng = np.random.default_rng(seed=42)
        image = image.astype(np.float64) + rng.normal(loc=0,
                                                      scale=sigma, size=image.shape)

    if mode is FULL_BRAIN:
        image = imread(r'images/fish_brain_scaning_em.tif')
        if 'resize' in kwargs and kwargs['resize']:
            image = resize(image, (size, size),
                           anti_aliasing=True)

    image = rescale_intensity(image, out_range=(0, 255)).astype(dtype)
    return image

def draw_graph(sources: np.ndarray, targets: np.ndarray, im_shape: Tuple[int, int]) -> np.ndarray:
    """
        Draw edges on an output image with shape 3Hx3W. 1 for edges and 0 otherwise.

    Args:
        sources (np.ndarray): Source vertices.
        targets (np.ndarray): Target vertices
        im_shape (Tuple[int, int]): Shape of the input image

    Returns:
        np.ndarray: Output binary image of shape 3Hx3W with 1 for edges and 0 otherwise.
    """
    img_out = np.zeros((im_shape[0]*3, im_shape[1]*3), dtype=np.uint8)

    for i in range(len(sources)):
        xu, yu = xy_from_index(sources[i], im_shape)
        xv, yv = xy_from_index(targets[i], im_shape)
        img_out = cv2.line(img_out, (3*xv+1, 3*yv+1),
                           (3*xu+1, 3*yu+1), (255, 255, 255), 1)

    return img_out

def print_causal_part(img: np.ndarray, causal_graph, slice_number: int) -> None:
    img_up = cv2.resize(
        img, (img.shape[1]*3, img.shape[0]*3), interpolation=cv2.INTER_NEAREST)

    graph = causal_graph.get_edge_weighted_graph(slice_number)[0]
    vertex_map = causal_graph.get_vertex_map(slice_number)
    sources = vertex_map[graph[0]]
    targets = vertex_map[graph[1]]
    del(graph)
    del(vertex_map)

    img_out = draw_graph(sources, targets, img.shape)

    plt.figure(figsize=(15, 15))
    plt.imshow(surprint(img_up, img_out, color=[255, 0, 0]))

def print_full_bpt(img: np.ndarray) -> None:
    graph = hg.get_4_adjacency_graph(img.shape)
    weights = hg.weight_graph(graph, img, hg.WeightFunction.L1)
    tree, _ = hg.bpt_canonical(graph, weights)
    edges = graph.edge_list()
    img_up = cv2.resize(
        img, (img.shape[1]*3, img.shape[0]*3), interpolation=cv2.INTER_NEAREST)
    img_out = np.zeros_like(img_up)

    mst_edge_map = tree.mst_edge_map
    sources = edges[0][mst_edge_map]
    targets = edges[1][mst_edge_map]
    print(len(tree.parents()))

    img_out = draw_graph(sources, targets, img.shape)

    plt.figure(figsize=(15, 15))
    plt.imshow(surprint(img_up, img_out, color=[255, 0, 0]))

def compute_visualization(img: np.ndarray, hierarchies: List, causal_graph) -> np.ndarray:
    """
    Compute visualization of the given distributed hierachy for each causal partition.

    Args:
        img (np.ndarray): Grayscale image.
        hierarchies (List): List of hierarchies i.e. hg.tree.
        causal_graph (_type_): Causal partition used to compute the distributed hierarchy.

    Returns:
        np.ndarray: binary matrix.
    """
    mask_common = np.empty(
        (causal_graph.num_slices, img.shape[0]*3, img.shape[1]*3), dtype=np.bool_)

    for j in tqdm(range(len(hierarchies))):

        mask = np.zeros((img.shape[0]*3, img.shape[1]*3), dtype=np.bool_)
        node_map = np.sort(hierarchies[j].node_map[hierarchies[j].num_leaves():])

        for idx in range(causal_graph.num_slices):

            start, stop = causal_graph._get_slice_interval(idx)
            graph = causal_graph.get_edge_weighted_graph(idx)[0]
            graph = (graph[0] + start * causal_graph.width,
                     graph[1] + start * causal_graph.width)
            edge_map = causal_graph.get_edge_map(idx)

            # add border edges
            if idx != causal_graph.num_slices-1:
                edge_sources, edge_targets, edge_map_border, _ = causal_graph.get_weighted_border_edges(
                    idx)
                edge_map = np.concatenate([edge_map, edge_map_border])
                graph = (np.concatenate([graph[0], edge_sources]), np.concatenate(
                    [graph[1], edge_targets]))

            def c1(x):
                return (x >= start * (causal_graph.width - 1)
                    ) & (x < stop * (causal_graph.width - 1))

            def c2(x):
                return (x >= causal_graph._num_h_edges + start * causal_graph.width) & (x <
                    causal_graph._num_h_edges + stop * causal_graph.width)

            current_nmap = node_map[np.where(c1(node_map) | c2(node_map))]

            edges = [None] * len(current_nmap)
            init = 0
            for i in range(len(current_nmap)):
                for k in range(init, len(edge_map)):
                    if current_nmap[i] == edge_map[k]:
                        edges[i] = k
                        init = k
                        break

            del(edge_map)
            del(current_nmap)

            sources = graph[0][edges]
            targets = graph[1][edges]

            del(graph)
            del(edges)

            img_out = draw_graph(sources, targets, img.shape)
            mask = np.any(np.stack([img_out, mask]), axis=0)

        mask_common[j] = mask

    return mask_common
