"""Offline ball trajectory smoother, evaluated against the online tracker.

`pool` merges per-frame candidates from two detectors: candidates within
`merge_px` of each other are one candidate whose confidence is the noisy-or of
the two. `smooth` then picks one candidate or none per frame with a
second-order Viterbi pass over past and future frames: the unary cost is the
negative log confidence (or `none_cost` for no ball), and the transition cost
penalizes acceleration, capped at `acc_cap` so hits and bounces stay possible,
and forbids steps longer than `max_step` pixels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

Cand = Tuple[float, float, float]


@dataclass
class SmootherParams:
    merge_px: float = 4.0
    max_step: float = 70.0
    acc_sigma: float = 4.0
    acc_cap: float = 6.0
    none_cost: float = 1.2


def pool(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]], params: SmootherParams = SmootherParams()) -> List[Cand]:
    merged = []
    for x, y, c in list(a) + list(b):
        for m in merged:
            if np.hypot(m[0] - x, m[1] - y) <= params.merge_px:
                if c > m[2]:
                    m[0], m[1] = x, y
                m[3] = 1 - (1 - m[3]) * (1 - c)
                m[2] = max(m[2], c)
                break
        else:
            merged.append([x, y, c, c])
    merged.sort(key=lambda m: -m[3])
    return [(m[0], m[1], m[3]) for m in merged[:8]]


def smooth(cands: Sequence[Sequence[Cand]], params: SmootherParams = SmootherParams()) -> List[Optional[Cand]]:
    T = len(cands)
    opts = [[None] + list(c) for c in cands]
    unary = [np.array([params.none_cost] + [-np.log(max(c[2], 1e-3)) for c in o[1:]]) for o in opts]

    def trans(a, b, c):
        if c is None or b is None:
            return 0.0
        step = np.hypot(c[0] - b[0], c[1] - b[1])
        if step > params.max_step:
            return 1e6
        if a is None:
            return 0.0
        acc = np.hypot(c[0] - 2 * b[0] + a[0], c[1] - 2 * b[1] + a[1])
        return min((acc / params.acc_sigma) ** 2, params.acc_cap)

    if T == 0:
        return []
    cost = {(0, j): unary[0][j] for j in range(len(opts[0]))}
    if T == 1:
        j = min(cost, key=cost.get)[1]
        return [opts[0][j]]
    back = []
    prev = {}
    for i in range(len(opts[0])):
        for j in range(len(opts[1])):
            prev[(i, j)] = unary[0][i] + unary[1][j] + trans(None, opts[0][i], opts[1][j])
    for t in range(2, T):
        cur, bp = {}, {}
        for (i, j), cval in prev.items():
            a, b = opts[t - 2][i], opts[t - 1][j]
            for k in range(len(opts[t])):
                v = cval + unary[t][k] + trans(a, b, opts[t][k])
                if v < cur.get((j, k), np.inf):
                    cur[(j, k)] = v
                    bp[(j, k)] = i
        back.append(bp)
        prev = cur
    j, k = min(prev, key=prev.get)
    path = [k, j]
    for bp in reversed(back):
        path.append(bp[(path[-1], path[-2])])
    path.reverse()
    return [opts[t][path[t]] for t in range(T)]
