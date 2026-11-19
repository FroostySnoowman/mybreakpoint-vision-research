"""Per-frame ball candidates from exported ONNX models.

Three model shapes are evaluated, all at 512x288:

- `heatmap_model`: five consecutive frames (RGB scaled to 0..1) in, one heatmap
  for the newest frame out; candidates are the top four local maxima (`top4`).
- `triplet_model`: three frames (raw RGB 0..255) in, top-k sub-pixel candidates
  for each of the three frames out. Frames are fed as non-overlapping triplets.
- `candidate_model`: five frames (raw RGB 0..255) in, top-k sub-pixel candidates
  for the newest frame out (the bundled on-device model).

Candidates below the tracker's `min_candidate_confidence` are dropped, so the
same threshold applies as in the tracker configuration.
"""

from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

import numpy as np

PREFERRED_PROVIDERS = ('CoreMLExecutionProvider', 'CPUExecutionProvider')
CandidateList = List[Tuple[float, float, float]]


def session(path: str):
    import onnxruntime as ort

    available = set(ort.get_available_providers())
    return ort.InferenceSession(str(path), providers=[p for p in PREFERRED_PROVIDERS if p in available])


def top4(hm: np.ndarray, min_confidence: float) -> CandidateList:
    """The four strongest local maxima (11x11 neighborhood) above `min_confidence`."""
    import cv2

    pooled = cv2.dilate(hm, np.ones((11, 11), np.uint8))
    pk = np.argwhere((hm >= pooled) & (hm > min_confidence))
    vals = hm[pk[:, 0], pk[:, 1]]
    order = np.argsort(-vals)[:4]
    return [(float(pk[i, 1]), float(pk[i, 0]), float(vals[i])) for i in order]


def heatmap_model(path: str, min_confidence: float) -> Callable[[Sequence[np.ndarray]], List[CandidateList]]:
    sess = session(path)
    name = sess.get_inputs()[0].name

    def cands(frames):
        out = [[] for _ in frames]
        for i in range(4, len(frames)):
            x = np.concatenate([f.astype(np.float32).transpose(2, 0, 1) / 255.0 for f in frames[i - 4:i + 1]], 0)[None]
            out[i] = top4(sess.run(None, {name: x})[0][0, 0], min_confidence)
        return out
    return cands


def triplet_model(path: str, min_confidence: float) -> Callable[[Sequence[np.ndarray]], List[CandidateList]]:
    sess = session(path)

    def cands(frames):
        out = [[] for _ in frames]
        start = len(frames) % 3
        for i in range(start, len(frames), 3):
            x = np.concatenate([f.astype(np.float32).transpose(2, 0, 1) for f in frames[i:i + 3]], 0)[None]
            pk = sess.run(None, {'frames': x})[0][0]
            for j in range(3):
                out[i + j] = [tuple(map(float, c)) for c in pk[j] if c[2] >= min_confidence]
        return out
    return cands


def candidate_model(path: str, min_confidence: float) -> Callable[[Sequence[np.ndarray]], List[CandidateList]]:
    sess = session(path)

    def cands(frames):
        out = [[] for _ in frames]
        for i in range(4, len(frames)):
            x = np.concatenate([f.astype(np.float32).transpose(2, 0, 1) for f in frames[i - 4:i + 1]], 0)[None]
            pk = sess.run(None, {'frames': x})[0][0, 0]
            out[i] = [tuple(map(float, c)) for c in pk if c[2] >= min_confidence]
        return out
    return cands


MODEL_KINDS = {'heatmap': heatmap_model, 'triplet': triplet_model, 'candidate': candidate_model}


def parse_model_spec(spec: str) -> Tuple[str, str, str]:
    """`kind:path` or `name=kind:path` into (name, kind, path)."""
    name, _, rest = spec.rpartition('=')
    kind, _, path = rest.partition(':')
    if kind not in MODEL_KINDS or not path:
        raise ValueError(f'expected [name=]kind:path with kind in {sorted(MODEL_KINDS)}, got {spec!r}')
    return name or kind, kind, path
