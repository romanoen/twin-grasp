"""Fit a table-supported cube (x, y, yaw) to silhouettes from one or more calibrated views. No hardware."""
import cv2
import numpy as np
from scipy.optimize import least_squares

EDGES = [(0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 7), (6, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
FACES = [(0, 2, 6, 4), (1, 5, 7, 3), (0, 4, 5, 1), (2, 3, 7, 6), (0, 1, 3, 2), (4, 6, 7, 5)]


def cube_vertices(q, edge, tag_height):
    angle = np.deg2rad(q[2])
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    points = np.array([[x, y, z] for z in (0., edge) for y in (-edge / 2, edge / 2) for x in (-edge / 2, edge / 2)])
    points[:, :2] = points[:, :2] @ rotation.T + np.array(q[:2]) / 1000
    points[:, 2] -= tag_height
    return points


def project(q, view, edge, tag_height):
    points = cube_vertices(q, edge, tag_height)
    transform = view["transform"]
    if np.min((transform[:3, :3] @ points.T + transform[:3, 3:4])[2]) <= 0:
        raise ValueError("Cube behind camera")
    pixels = cv2.projectPoints(points, cv2.Rodrigues(transform[:3, :3])[0], transform[:3, 3],
                               view["matrix"], view["distortion"])[0].reshape(-1, 2)
    indices = cv2.convexHull(pixels.astype(np.float32), returnPoints=False).ravel()
    return pixels, pixels[indices]


def sample_boundary(polygon, count=48):
    polygon = np.asarray(polygon, np.float64)
    vectors = np.roll(polygon, -1, axis=0) - polygon
    lengths = np.linalg.norm(vectors, axis=1)
    if lengths.sum() <= 0:
        raise ValueError("Degenerate boundary")
    starts = np.r_[0., np.cumsum(lengths)]
    positions = np.arange(count) * starts[-1] / count
    segments = np.searchsorted(starts[1:], positions, side="right")
    fraction = (positions - starts[segments]) / np.maximum(lengths[segments], 1e-12)
    return polygon[segments] + vectors[segments] * fraction[:, None]


def boundary_distances(points, polygon):
    a = np.asarray(polygon, np.float64)
    vectors = np.roll(a, -1, axis=0) - a
    offset = np.asarray(points)[:, None, :] - a[None, :, :]
    fraction = np.clip(np.sum(offset * vectors, axis=2) / np.maximum(np.sum(vectors ** 2, axis=1), 1e-12), 0, 1)
    squared = np.sum((offset - fraction[..., None] * vectors) ** 2, axis=2)
    return np.sqrt(np.maximum(squared.min(axis=1), 1e-20))


def view_residual(q, view, edge, tag_height):
    _, predicted = project(q, view, edge, tag_height)
    return np.r_[boundary_distances(view["observed_samples"], predicted),
                 boundary_distances(sample_boundary(predicted), view["contour"])]


def residual(q, views, edge, tag_height):
    return np.concatenate([view_residual(q, view, edge, tag_height) for view in views])


def seed_position(views, edge, tag_height):
    seeds = []
    for view in views:
        moments = cv2.moments(view["contour"].astype(np.float32))
        center = np.array([moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]])
        normalized = cv2.undistortPoints(center.reshape(1, 1, 2), view["matrix"], view["distortion"]).ravel()
        transform = view["transform"]
        camera = -transform[:3, :3].T @ transform[:3, 3]
        ray = transform[:3, :3].T @ np.r_[normalized, 1.]
        if abs(ray[2]) < 1e-6:
            raise ValueError("Viewing ray parallel to table")
        point = camera + ray * ((edge / 2 - tag_height - camera[2]) / ray[2])
        seeds.append(point[:2] * 1000)
    return np.median(seeds, axis=0)


def fit(views, edge, tag_height):
    if len(views) < 2:
        raise ValueError("At least two fit views required")
    if any(view.get("role") != "fit" for view in views):
        raise ValueError("Only fit views may enter optimization")
    center = seed_position(views, edge, tag_height)
    lower, upper = np.r_[center - 75, -180.], np.r_[center + 75, 180.]
    runs = []
    for angle in (0., 15., 30., 45., 60., 75.):
        result = least_squares(residual, np.r_[center, angle], args=(views, edge, tag_height),
                               bounds=(lower, upper), loss="soft_l1", f_scale=1.5, max_nfev=200,
                               diff_step=1e-5, x_scale=np.array([10., 10., 20.]))
        runs.append(result)
    runs.sort(key=lambda item: float(item.cost))
    best = runs[0]
    q = best.x.copy()
    q[2] %= 90
    return q, best, [{"q_mm_mm_deg": item.x.tolist(), "robust_cost": float(item.cost),
                      "converged": bool(item.success), "evaluations": int(item.nfev)} for item in runs]
