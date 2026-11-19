"""Red cube segmentation and face proposals in one wrist-camera image. No hardware.

Colour regions stay unnamed: the brightest face is not necessarily the top. Several plausible red components are
reported as ambiguous, never chosen by size.
"""
import cv2
import numpy as np


def _hsv_red(hsv, low_max, high_min, sat, val):
    return cv2.inRange(hsv, (0, sat, val), (low_max, 255, 255)) | cv2.inRange(hsv, (high_min, sat, val), (179, 255, 255))


def red_mask(frame, settings):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    broad = _hsv_red(hsv, settings["hsv_red_low_max"], settings["hsv_red_high_min"], settings["min_saturation"], settings["min_value"])
    if "seed_hue_low_max" not in settings:
        return broad
    # Two-stage (optional): a strict seed (clearly red, the cube) excludes warm-lit wood, which falls inside the broad
    # filter; broad pixels within seed_grow_px of a seed recover the cube's shaded, less saturated faces.
    seed = _hsv_red(hsv, settings["seed_hue_low_max"], settings["hsv_red_high_min"], settings["seed_min_saturation"], settings["seed_min_value"])
    seed = cv2.morphologyEx(seed, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    k = settings.get("seed_close_px", 0)
    if k:   # bridge thin specular highlights on cube edges that would split the cube into two components
        seed = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    r = settings["seed_grow_px"]
    if r <= 0:
        return seed
    near = cv2.dilate(seed, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r+1, 2*r+1)))
    return seed | (broad & near)


def surface_regions(frame, mask, settings):
    """Photometric proposals only; no top/side classification or 3D fitting."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    colors = lab[mask > 0].astype(np.float32)
    if len(colors) < 20:
        return [], np.zeros(mask.shape, np.uint8)
    partitions = []
    for count in (2, 3):
        cv2.setRNGSeed(0)
        compactness, labels, centers = cv2.kmeans(colors, count, None,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1), 3, cv2.KMEANS_PP_CENTERS)
        fractions = np.bincount(labels.ravel(), minlength=count) / len(colors)
        distances = [np.linalg.norm(centers[a] - centers[b]) for a in range(count) for b in range(a)]
        if min(fractions) >= settings["min_color_region_fraction"] and min(distances) >= settings["min_lab_center_separation"]:
            partitions.append((count, compactness, labels, centers))
    selected = next((item for item in partitions if item[0] == 2), None)
    third = next((item for item in partitions if item[0] == 3), None)
    if selected is not None and third is not None and third[1] < selected[1] * settings["third_cluster_max_relative_compactness"]:
        selected = third
    regions = []
    label_image = np.zeros(mask.shape, np.uint8)
    if selected is None:
        return regions, label_image
    count, _, labels, centers = selected
    # Stable ordering by Lab lightness; this order carries no physical semantics.
    for region_id, cluster in enumerate(np.argsort(centers[:, 0]), 1):
        region = np.zeros(mask.shape, np.uint8)
        region[mask > 0] = (labels.ravel() == cluster).astype(np.uint8) * 255
        contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        if area < settings["min_color_region_fraction"] * np.count_nonzero(mask):
            continue
        polygon = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        label_image[region > 0] = region_id
        regions.append({"id": region_id, "kind": "unassigned_color_region", "area_px": float(area),
                        "polygon_px": polygon.reshape(-1, 2).tolist(), "lab_center": centers[cluster].tolist(),
                        "physical_face_verified": False})
    return regions, label_image


def extract(frame, settings, tag_area_px):
    mask = red_mask(frame, settings)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    candidates = []
    component_masks = {}
    height, width = mask.shape
    for label in range(1, count):
        x, y, w, h, area = [int(item) for item in stats[label]]
        if area < settings["min_component_area_px"]:
            continue
        region = (labels == label).astype(np.uint8) * 255
        contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
        hull_area = cv2.contourArea(cv2.convexHull(contour))
        solidity = cv2.contourArea(contour) / hull_area if hull_area > 0 else 0
        ratio = float(area / tag_area_px)
        margin = settings["border_margin_px"]
        reasons = []
        if x <= margin or y <= margin or x + w >= width - margin or y + h >= height - margin:
            reasons.append("component_touches_image_border")
        if not settings["min_area_relative_to_tag"] <= ratio <= settings["max_area_relative_to_tag"]:
            reasons.append("area_inconsistent_with_initial_cube_tag_ratio")
        if solidity < settings["min_solidity"]:
            reasons.append("nonconvex_or_fragmented_component")
        candidates.append({"component": label, "bbox_xywh": [x, y, w, h], "area_px": area,
                           "area_relative_to_tag": ratio, "solidity": float(solidity), "rejection_reasons": reasons})
        component_masks[label] = (region, contour)
    plausible = [item for item in candidates if not item["rejection_reasons"]]
    result = {"status": "no_target" if not plausible else "ambiguous_target", "candidates": candidates,
              "scene_valid_for_planning": False, "physical_edges_verified": False}
    if len(plausible) != 1:
        return result, mask, np.zeros(mask.shape, np.uint8)
    selected = plausible[0]
    region, contour = component_masks[selected["component"]]
    polygon = cv2.approxPolyDP(contour, settings["polygon_epsilon_perimeter_fraction"] * cv2.arcLength(contour, True), True)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0.7), settings["canny_low"], settings["canny_high"])
    inside = cv2.erode(region, np.ones((5, 5), np.uint8))
    edges = cv2.bitwise_and(edges, inside)
    minimum = max(8, round(min(selected["bbox_xywh"][2:]) * 0.18))
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=8, minLineLength=minimum, maxLineGap=3)
    segments = [] if lines is None else [line.reshape(2, 2).tolist() for line in lines]
    regions, label_image = surface_regions(frame, region, settings)
    result.update(status="feature_candidate_requires_3d_validation", selected_component=selected["component"],
                  bbox_xywh=selected["bbox_xywh"], silhouette_contour_px=contour.reshape(-1, 2).tolist(),
                  silhouette_polygon_px=polygon.reshape(-1, 2).tolist(), color_regions=regions,
                  internal_line_candidates_px=segments,
                  interpretation="Silhouette vertices, color regions and Hough lines are proposals; no physical corner correspondence or top-face assignment")
    return result, region, label_image
