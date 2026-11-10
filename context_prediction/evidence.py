"""E1 component assignment and coordinate helpers, independent of LogiCo."""
import math
import numpy as np
from scipy.optimize import linear_sum_assignment
from PIL import Image


def component_cost(expected, observed, global_query=False):
    a = np.asarray(expected['appearance'], dtype=float)
    b = np.asarray(observed['appearance'], dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if not denom:
        raise ValueError('appearance vectors must be nonzero')
    appearance = np.clip((1 - np.dot(a, b) / denom) / 2, 0, 1)
    color = np.clip(np.linalg.norm(np.asarray(expected['color']) - np.asarray(observed['color'])) / 100, 0, 1)
    position = np.clip(np.linalg.norm(np.asarray(expected['center']) - np.asarray(observed['center'])) / math.sqrt(2), 0, 1)
    if expected['area'] <= 0 or observed['area'] <= 0:
        raise ValueError('component areas must be positive')
    area = np.clip(abs(math.log(expected['area'] / observed['area'])) / math.log(4), 0, 1)
    if global_query:
        return float((.5 * appearance + .2 * color + .15 * area) / .85)
    return float(.5 * appearance + .2 * color + .15 * position + .15 * area)


def match(expected, observed, global_query=False, unmatched_cost=.35):
    """Match ID-sorted components, returning pairs and both unmatched lists."""
    expected = sorted(expected, key=lambda item: str(item['id']))
    observed = sorted(observed, key=lambda item: str(item['id']))
    m, n = len(expected), len(observed)
    if len({str(x['id']) for x in expected}) != m or len({str(x['id']) for x in observed}) != n:
        raise ValueError('component IDs must be unique within each set')
    if not m or not n:
        return {'pairs': [], 'missing': [x['id'] for x in expected], 'extra': [x['id'] for x in observed]}
    cost = np.full((m+n, m+n), np.inf)
    for i, e in enumerate(expected):
        for j, o in enumerate(observed):
            cost[i, j] = component_cost(e, o, global_query)
        cost[i, n+i] = unmatched_cost
    for j in range(n):
        cost[m+j, j] = unmatched_cost
    cost[m:, n:] = 0
    rows, cols = linear_sum_assignment(cost)
    pairs, missing, extra = [], [], []
    for i, j in zip(rows, cols):
        if i < m and j < n:
            pairs.append((expected[i]['id'], observed[j]['id'], float(cost[i, j])))
        elif i < m:
            missing.append(expected[i]['id'])
        elif j < n:
            extra.append(observed[j]['id'])
    return {'pairs': pairs, 'missing': missing, 'extra': extra}


def rasterize_box(box, shape):
    """Clip an integer xyxy, left-closed/right-open box to a (height,width) image."""
    h, w = shape
    x1, y1, x2, y2 = map(int, box)
    x1, x2 = np.clip([x1, x2], 0, w)
    y1, y2 = np.clip([y1, y2], 0, h)
    result = np.zeros((h, w), dtype=np.uint8)
    if x2 > x1 and y2 > y1:
        result[y1:y2, x1:x2] = 1
    return result


def resize_mask(mask, shape):
    h, w = shape
    return np.asarray(Image.fromarray(np.asarray(mask, dtype=np.uint8)).resize((w, h), Image.Resampling.NEAREST))


def fuse_with_evidence(a, evidence, scale):
    a, evidence = np.asarray(a), np.asarray(evidence)
    if a.shape != evidence.shape:
        raise ValueError('A and E must have identical shapes')
    if not np.any(evidence):
        return a.copy()
    return a + scale * evidence
