#!/usr/bin/env python
from __future__ import division

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import io
from skimage.color import rgb2gray
from skimage.measure import label
from skimage.measure import regionprops
from skimage.segmentation import clear_border
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import black_tophat, skeletonize, convex_hull_image
from skimage.morphology import disk, square
from skimage import img_as_float, img_as_uint
from toolbox.

def patch_local_max(image, response, threshold, shape):
    img_copy = image
    detected = rgb2gray(response)
    height, width = detected.shape
    sm = ndimage.gaussian_filter(detected, sigma=5)
    bw = sm > threshold  #70
    selem = shape   #disk(15)
    dilated = dilation(bw, selem)

    # to be safe, remove any artifacts connect to border
    cleared = dilated.copy()
    clear_border(cleared)

    # label the remain regions (should be "signs")
    label_image = label(cleared)
    borders = np.logical_xor(dilated, cleared)
    label_image[borders] = -1

    patches = []
    # plot the brighest points over original image
    region_num = 0
    for region in regionprops(label_image):
        # find center of region
        minr, minc, maxr, maxc = region.bbox
        #check aspect ratio
        aspect_ratio = (maxc - minc) / (maxr - minr)
        #print aspect_ratio
        if (aspect_ratio > 1.3) or (aspect_ratio < 0.3):
            continue
        if region.area < 900: # or region.area > 2000:
            continue
        patches.append([img_copy[minr:maxr,minc:maxc].copy(),((minr+maxr)//2,(minc+maxc)//2),region.bbox])
    return patches


def selective_search(image):
    # initial segmentaiton
    labels = segmentation.slic(img, compactness=0.1, n_segments=290)

    # setup custome RAG.  See define_regions() for node and edge weights
    rag = graph.region_adjacency_graph(labels, image=img, connectivity=2,
                                       describe_func=__define_regions_rag,
                                       extra_arguments=[],
                                       extra_keywords={'depth':depth, 'weight':
                                                       (1,1,1,1,0)})

    # merge, stop when final merge is the entire image
    merged, trace = graph.merge_hierarchical(labels, rag, thresh=np.inf,
                                             rag_copy=True,
                                             in_place_merge=False,
                                             merge_trace=True,
                                             merge_func=__merge_selective,
                                             weight_func=__similar_selective)

    # Add regions from initial segmentation
    add_bbox(labels, detections)

    # Add regions from each hierarchical merge step
    label_map = np.arange(labels.max() + 1)
    for merge in trace:
        label_map[:] = 2
        for label in merge:
            label_map[label] = 1
        add_bbox(label_map[labels], detections)
    return detections


def __similar_selective(graph, src, dst, n):
    """Callback to handle merging nodes by recomputing similarity measure.

    The method expects that the node `dst` has already been updated.

    Similarity of histograms is measure using histogram intersection.  Size is
    measured as a proportion of the image.  So large regions don't dominate
    (gobble up others) the proportion is subtracted from one. The fill measures
    how well two regions fit into each others, that aims to fill gaps.  The
    final similarity measure is:

      w_0*s_colour + w_1*s_texture + w_2*s_size + w_3*s_fill + w_4*s_depth

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    n : int
        A neighbor of `src` or `dst` or both.
    extra_arguments : sequence, optional
        Allows extra positional arguments passed.
    extra_keywords : dictionary, optional
        allows extra keyword arguments passed.
        weight : tuple
           denotes is the similarity measure is used, not used.

    Returns
    -------
    weight : float
        The similarity measure between node `dst` and `n`.
    """
    s_colour = hist_intersection(graph.node[dst]['colour hist'],
                                      graph.node[n]['colour hist'])
    s_texture = hist_intersection(graph.node[dst]['texture hist'],
                                       graph.node[n]['texture hist'])
    s_size = 1 - (graph.node[dst]['pixel count'] +
                  graph.node[n]['pixel count']) / graph.graph['image size']
    bbox  = enclosing_bbox(graph.node[dst]['bbox'], graph.node[n]['bbox'])
    s_fill = (size_box(bbox) - graph.node[dst]['pixel count'] -
              graph.node[n]['pixel count'])/ graph.graph['image size']
    s_depth = graph.node[dst]['mean depth'] - graph.node[n]['mean depth']

    w = graph.graph['weight']
    return (w[0]*s_colour + w[1]*s_texture + w[2]*s_size + w[3]*s_fill +
            w[4]*s_depth)


def __merge_selective(graph, src, dst):
    """Callback called before merging two nodes of a mean color distance graph.

    Updates colour histogram, texture histogram, pixel count and bbox fields.
    The histograms can be efficiently propagated through the hierarchy by:

        size_dst x hist_dst  +  size_src * hist_src
        -------------------------------------------
                   size_dst + size_src

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    """
    size = graph.node[dst]['pixel count'] + graph.node[src]['pixel count']
    graph.node[dst]['color hist'] = ((graph.node[dst]['pixel count'] *
                                      graph.node[dst]['colour hist']) +
                                     (graph.node[src]['pixel count'] *
                                      graph.node[src]['colour hist'])) / size
    graph.node[dst]['texture hist'] = ((graph.node[dst]['pixel count'] *
                                      graph.node[dst]['texture hist']) +
                                     (graph.node[src]['pixel count'] *
                                      graph.node[src]['texture hist'])) / size
    graph.node[dst]['bbox'] = enclosing_bbox(graph.node[dst]['bbox'],
                                             graph.node[src]['bbox'])
    graph.node[dst]['pixel count']  = size
    graph.node[dst]['total depth'] += graph.node[src]['total depth']
    graph.node[dst]['mean depth'] = graph.node[dst]['total depth'] / size


def __define_regions_rag(graph, labels, image, extra_arguments=[],
                       extra_keywords={'depth':None, 'weight':(1, 1, 1, 1, 1)}):
    """Callback to handle describing nodes and calculating edge weights.

    Nodes can have arbitrary Python objects assigned as attributes. This method
    expects a valid graph and computes the similarity measure of the node.

    Similarity of histograms is measure using histogram intersection.  Size is
    measured as a proportion of the image.  So large regions don't dominate
    (gobble up others) the proportion is subtracted from 1. The fill measures
    how well two regions fit into each others, that aims to fill gaps.  The
    final similarity measure is:

      w_0*s_colour + w_1*s_texture + w_2*s_size + w_3*s_fill + w_4*s_depth

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    labels : ndarray, shape(M, N, [..., P,])
        The labelled image. This should have one dimension less than
        `image`. If `image` has dimensions `(M, N, 3)` `labels` should have
        dimensions `(M, N)`.
    image : ndarray, shape(M, N, [..., P,] 3)
        Input image.
    extra_arguments : sequence, optional
        Allows extra positional arguments passed.
    extra_keywords : dictionary, optional
        allows extra keyword arguments passed.
        weight : tuple
           denotes is the similarity measure is used, not used.
    """
    # Describe the nodes
    graph.graph['image size'] = image.size
    graph.graph['weight'] = extra_keywords['weight']
    size = 75 if image.ndim == 3 else 25
    for n in graph:
        graph.node[n].update({'labels': [n],
                              'pixel count': 0,
                              'total depth': 0,
                              'bbox': (0,0,0,0),
                              'mean depth' : 0,
                              'colour hist': np.zeros(size, dtype=np.double),
                              'texture hist': np.zeros(20, dtype=np.double)})

    depth = extra_keywords['depth']
    for index in np.ndindex(labels.shape):
        current = labels[index]
        graph.node[current]['pixel count'] += 1
        if depth != None:
            graph.node[current]['total depth'] += depth[index]

    for n in graph:
        patch = labels == n
        graph.node[n]['mean depth'] = (graph.node[n]['total depth'] /
                                       graph.node[n]['pixel count'])
        graph.node[n]['colour hist'] = chist(image,patch)
        regions = measure.regionprops(patch.astype(int))
        for props in regions:
            graph.node[n]['bbox'] = props.bbox
            graph.node[n]['texture hist'] = thist(image, props.bbox)

    # Calcuate the edge weights
    w = extra_keywords['weight']
    for x, y, d in graph.edges_iter(data=True):
        s_colour = hist_intersection(graph.node[x]['colour hist'],
                                     graph.node[y]['colour hist'])
        s_texture = hist_intersection(graph.node[x]['texture hist'],
                                      graph.node[y]['texture hist'])
        s_size = 1 - (graph.node[x]['pixel count'] +
                      graph.node[y]['pixel count']) / graph.graph['image size']
        bbox  = enclosing_bbox(graph.node[x]['bbox'], graph.node[y]['bbox'])
        s_fill = (size_box(bbox) - graph.node[x]['pixel count'] -
                  graph.node[y]['pixel count'])/ graph.graph['image size']
        s_depth = graph.node[x]['mean depth'] - graph.node[y]['mean depth']

        d['weight'] = (w[0]*s_colour + w[0]*s_texture + w[1]*s_size +
                       w[2]*s_fill + w[3]*s_depth)
