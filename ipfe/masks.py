import numpy as np

sobel_masks = (
    np.array([
             [-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]]).astype(float) / 4.0,
    np.array([
             [0, 1, 2],
             [-1, 0, 1],
             [-2, -1, 0]]).astype(float) / 4.0,
    np.array([
             [1, 2, 1],
             [0, 0, 0],
             [-1, -2, -1]]).astype(float) / 4.0,
    np.array([
             [2, 1, 0],
             [1, 0, -1],
             [0, -1, -2]]).astype(float) / 4.0)

scharr_masks = (
    np.array([
             [3, 0, -3],
             [10, 0, -10],
             [3, 0, -3]]).astype(float) / 4.0,
    np.array([
             [0, -3, -10],
             [3, 0, -3],
             [10, 3, 0]]).astype(float) / 4.0,
    np.array([
             [-3, -10, -3],
             [0, 0, 0],
             [3, 10, 3]]).astype(float) / 4.0,
    np.array([
             [-10, -3, 0],
             [-3, 0, 3],
             [0, 3, 10]]).astype(float) / 4.0)

prewitt_masks = (
    np.array([
             [-1, 1, 1],
             [-1, -2, 1],
             [-1, 1, 1]]).astype(float) / 3.0,
    np.array([
             [1, 1, 1],
             [-1, -2, 1],
             [-1, -1, 1]]).astype(float) / 3.0,
    np.array([
             [1, 1, 1],
             [1, -2, 1],
             [-1, -1, -1]]).astype(float) / 3.0,
    np.array([
             [1, 1, 1],
             [1, -2, -1],
             [1, -1, -1]]).astype(float) / 3.0)

kirsch_masks = (
    np.array([
             [-3, -3, 5],
             [-3, 0, 5],
             [-3, -3, 5]]).astype(float) / 4.0,
    np.array([
             [-3, 5, 5],
             [-3, 0, 5],
             [-3, -3, -3]]).astype(float) / 4.0,
    np.array([
             [5, 5, 5],
             [-3, 0, -3],
             [-3, -3, -3]]).astype(float) / 4.0,
    np.array([
             [5, 5, -3],
             [5, 0, -3],
             [-3, -3, -3]]).astype(float) / 4.0)

robinson_masks = (
    np.array([
             [-1, 0, 1],
             [-1, 0, 1],
             [-1, 0, 1]]).astype(float) / 4.0,
    np.array([
             [0, 1, 1],
             [-1, 0, 1],
             [-1, -1, 0]]).astype(float) / 4.0,
    np.array([
             [1, 1, 1],
             [0, 0, 0],
             [-1, -1, -1]]).astype(float) / 4.0,
    np.array([
             [1, 1, 0],
             [1, 0, -1],
             [0, -1, -1]]).astype(float) / 4.0)

