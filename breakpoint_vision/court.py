"""Real-world tennis court model, in meters.

Coordinate system (world plane, looking down from above):
  origin (0, 0) = center of the net
  +x            = toward the right sideline
  +y            = toward the far baseline (the near side is negative y)

All dimensions are the ITF regulation court. The court is symmetric about both
axes. A homography (see `geometry`) maps image pixels onto this plane.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

Point = Tuple[float, float]
Polygon = List[Point]

COURT_LENGTH = 23.77
COURT_DOUBLES_WIDTH = 10.97
COURT_SINGLES_WIDTH = 8.23
SERVICE_LINE_FROM_NET = 6.40
LINE_WIDTH = 0.05

HALF_LENGTH = COURT_LENGTH / 2.0
HALF_DOUBLES = COURT_DOUBLES_WIDTH / 2.0
HALF_SINGLES = COURT_SINGLES_WIDTH / 2.0


@dataclass(frozen=True)
class CourtKeypoints:
    """The 4 court corners (doubles) plus useful line intersections.

    The order here is the canonical order the court-keypoint model is trained to
    predict and the order the homography solver expects. Do not reorder without
    updating both.
    """

    names: Tuple[str, ...] = (
        "bl_doubles_near_left",
        "bl_doubles_near_right",
        "bl_doubles_far_right",
        "bl_doubles_far_left",
        "sl_near_left",
        "sl_near_right",
        "sl_far_right",
        "sl_far_left",
        "svc_near_left",
        "svc_near_right",
        "svc_far_right",
        "svc_far_left",
        "center_svc_near",
        "center_svc_far",
    )

    def world_points(self) -> Dict[str, Point]:
        near = -HALF_LENGTH
        far = HALF_LENGTH
        svc_near = -SERVICE_LINE_FROM_NET
        svc_far = SERVICE_LINE_FROM_NET
        return {
            "bl_doubles_near_left": (-HALF_DOUBLES, near),
            "bl_doubles_near_right": (HALF_DOUBLES, near),
            "bl_doubles_far_right": (HALF_DOUBLES, far),
            "bl_doubles_far_left": (-HALF_DOUBLES, far),
            "sl_near_left": (-HALF_SINGLES, near),
            "sl_near_right": (HALF_SINGLES, near),
            "sl_far_right": (HALF_SINGLES, far),
            "sl_far_left": (-HALF_SINGLES, far),
            "svc_near_left": (-HALF_SINGLES, svc_near),
            "svc_near_right": (HALF_SINGLES, svc_near),
            "svc_far_right": (HALF_SINGLES, svc_far),
            "svc_far_left": (-HALF_SINGLES, svc_far),
            "center_svc_near": (0.0, svc_near),
            "center_svc_far": (0.0, svc_far),
        }

    def ordered_world_array(self):
        pts = self.world_points()
        return [pts[name] for name in self.names]


def singles_court_polygon() -> Polygon:
    return [
        (-HALF_SINGLES, -HALF_LENGTH),
        (HALF_SINGLES, -HALF_LENGTH),
        (HALF_SINGLES, HALF_LENGTH),
        (-HALF_SINGLES, HALF_LENGTH),
    ]


def doubles_court_polygon() -> Polygon:
    return [
        (-HALF_DOUBLES, -HALF_LENGTH),
        (HALF_DOUBLES, -HALF_LENGTH),
        (HALF_DOUBLES, HALF_LENGTH),
        (-HALF_DOUBLES, HALF_LENGTH),
    ]
