"""
Multi-camera fusion module for Phase 3 industrial surveillance system.

Associates detections across overlapping camera pairs using spatial
proximity (Hungarian algorithm) and propagates stable global IDs via
a Union-Find structure with path compression.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from detection import Detection


class MultiCameraFusion:
    def __init__(
        self,
        config: dict,
        distance_threshold_m: float = 1.0,
        time_window_s: float = 0.1,
    ) -> None:
        self._distance_threshold_m = distance_threshold_m
        self._time_window_s = time_window_s
        self._overlap_pairs: list[tuple[str, str]] = self._load_overlap_pairs(config)

        # Union-Find parent map keyed on (cam_id, track_id)
        self._parent: dict[tuple[str, int], tuple[str, int]] = {}

        # Persistent global ID registry: survives across associate() calls
        self._track_to_global: dict[tuple[str, int], int] = {}
        self._next_global_id: int = 1

        print(
            f"[INFO] MultiCameraFusion initialised: "
            f"{len(self._overlap_pairs)} overlap pairs, "
            f"dist_thresh={distance_threshold_m}m, "
            f"time_window={time_window_s}s"
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def associate(
        self, detections_by_cam: dict[str, list[Detection]]
    ) -> list[Detection]:
        """Associate detections across overlapping camera pairs.

        Args:
            detections_by_cam: Mapping from camera ID to list of detections
                               for the current frame batch.

        Returns:
            Flat list of all detections with global_id populated.
        """
        all_dets: list[Detection] = [
            det for dets in detections_by_cam.values() for det in dets
        ]

        # Seed Union-Find with every node present this frame
        self._parent = {}
        for det in all_dets:
            key = (det.cam_id, det.track_id)
            if key not in self._parent:
                self._parent[key] = key

        # Step 1: run Hungarian association for each overlap pair
        for cam_a, cam_b in self._overlap_pairs:
            dets_a = detections_by_cam.get(cam_a, [])
            dets_b = detections_by_cam.get(cam_b, [])

            # Skip pairs where at least one camera has no detections
            if not dets_a or not dets_b:
                continue

            # Only consider detections that have a floor position
            valid_a = [d for d in dets_a if d.foot_point_m is not None]
            valid_b = [d for d in dets_b if d.foot_point_m is not None]

            if not valid_a or not valid_b:
                continue

            C = self._build_cost_matrix(valid_a, valid_b)
            row_inds, col_inds = linear_sum_assignment(C)

            for r, c in zip(row_inds, col_inds):
                if C[r, c] < self._distance_threshold_m:
                    self._union(
                        (valid_a[r].cam_id, valid_a[r].track_id),
                        (valid_b[c].cam_id, valid_b[c].track_id),
                    )

        # Step 2: gather components and assign / propagate global IDs
        components: dict[tuple[str, int], list[tuple[str, int]]] = {}
        for key in self._parent:
            root = self._find(key)
            components.setdefault(root, []).append(key)

        for members in components.values():
            # Collect any already-known global IDs for members of this component
            existing_ids = [
                self._track_to_global[m] for m in members if m in self._track_to_global
            ]
            # Oldest (smallest) ID wins — most stable across time
            gid = min(existing_ids) if existing_ids else self._allocate_global_id()
            for m in members:
                self._track_to_global[m] = gid

        # Step 3: detections without floor position also get a persistent global ID
        for det in all_dets:
            key = (det.cam_id, det.track_id)
            if key not in self._track_to_global:
                # Node was not reachable through Union-Find (foot_point_m is None)
                self._track_to_global[key] = self._allocate_global_id()

            det.global_id = self._track_to_global[key]

        return all_dets

    def reset(self) -> None:
        """Clear persistent tracking state between video sessions."""
        self._parent.clear()
        self._track_to_global.clear()
        self._next_global_id = 1
        print("[INFO] MultiCameraFusion state reset.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_overlap_pairs(self, config: dict) -> list[tuple[str, str]]:
        """Extract all camera overlap pairs from config, across all rooms."""
        pairs: list[tuple[str, str]] = []
        overlap_section = config.get("camera_overlaps", {})
        for room_pairs in overlap_section.values():
            for pair in room_pairs:
                if len(pair) == 2:
                    pairs.append((pair[0], pair[1]))
                else:
                    print(f"[WARN] Malformed overlap pair ignored: {pair}")
        return pairs

    def _build_cost_matrix(
        self, dets_a: list[Detection], dets_b: list[Detection]
    ) -> np.ndarray:
        """Build an (len_a × len_b) cost matrix of floor-plane distances.

        Entries outside the time window are set to 1e6 so the Hungarian
        algorithm will never choose them as optimal assignments.
        """
        n, m = len(dets_a), len(dets_b)
        C = np.empty((n, m), dtype=np.float64)
        for i, da in enumerate(dets_a):
            for j, db in enumerate(dets_b):
                if abs(da.timestamp - db.timestamp) > self._time_window_s:
                    C[i, j] = 1e6
                else:
                    dx = da.foot_point_m[0] - db.foot_point_m[0]  # type: ignore[index]
                    dy = da.foot_point_m[1] - db.foot_point_m[1]  # type: ignore[index]
                    C[i, j] = math.sqrt(dx * dx + dy * dy)
        return C

    def _find(self, x: tuple[str, int]) -> tuple[str, int]:
        """Find root with path-compression (one-pass halving)."""
        while self._parent[x] != x:
            # Path compression: skip one level
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def _union(self, a: tuple[str, int], b: tuple[str, int]) -> None:
        """Merge the components containing a and b."""
        # Nodes from detections without floor position may not be in parent
        for node in (a, b):
            if node not in self._parent:
                self._parent[node] = node

        root_a = self._find(a)
        root_b = self._find(b)
        if root_a != root_b:
            # Attach root_b under root_a (no rank — small number of cameras)
            self._parent[root_b] = root_a

    def _allocate_global_id(self) -> int:
        """Return the next available global ID and advance the counter."""
        gid = self._next_global_id
        self._next_global_id += 1
        return gid


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import yaml

    config_path = "config.yaml"
    with open(config_path, "r") as fh:
        cfg = yaml.safe_load(fh)

    fusion = MultiCameraFusion(cfg, distance_threshold_m=1.0, time_window_s=0.1)

    t = 1743000000.0

    det_cam03_1 = Detection(
        cam_id="cam_03",
        track_id=1,
        global_id=None,
        timestamp=t,
        bbox_px=(100.0, 200.0, 180.0, 400.0),
        foot_point_px=(140.0, 400.0),
        foot_point_m=(5.0, 2.0),
        confidence=0.92,
        class_id=0,
        class_name="person",
    )
    det_cam07_1 = Detection(
        cam_id="cam_07",
        track_id=1,
        global_id=None,
        timestamp=t + 0.03,
        bbox_px=(200.0, 210.0, 280.0, 410.0),
        foot_point_px=(240.0, 410.0),
        foot_point_m=(5.1, 2.05),   # close to det_cam03_1 → should merge
        confidence=0.88,
        class_id=0,
        class_name="person",
    )
    det_cam07_2 = Detection(
        cam_id="cam_07",
        track_id=2,
        global_id=None,
        timestamp=t,
        bbox_px=(400.0, 210.0, 480.0, 410.0),
        foot_point_px=(440.0, 410.0),
        foot_point_m=(8.0, 0.5),   # far from cam_03 detections → different ID
        confidence=0.75,
        class_id=0,
        class_name="person",
    )
    det_cam05_no_floor = Detection(
        cam_id="cam_05",
        track_id=3,
        global_id=None,
        timestamp=t,
        bbox_px=(50.0, 50.0, 100.0, 200.0),
        foot_point_px=(75.0, 200.0),
        foot_point_m=None,          # no homography → unique ID
        confidence=0.60,
        class_id=0,
        class_name="person",
    )

    result = fusion.associate(
        {
            "cam_03": [det_cam03_1],
            "cam_07": [det_cam07_1, det_cam07_2],
            "cam_05": [det_cam05_no_floor],
        }
    )

    print(f"\n[INFO] Frame 1 — {len(result)} detections after fusion:")
    for d in result:
        print(f"  {d}")

    # cam_03:1 and cam_07:1 should share the same global_id
    assert det_cam03_1.global_id == det_cam07_1.global_id, (
        f"[ERR] Expected cam_03:1 and cam_07:1 to share a global_id, "
        f"got {det_cam03_1.global_id} vs {det_cam07_1.global_id}"
    )
    assert det_cam07_2.global_id != det_cam03_1.global_id, (
        "[ERR] cam_07:2 should have a different global_id"
    )
    assert det_cam05_no_floor.global_id is not None, (
        "[ERR] Detection without floor point should still get a global_id"
    )

    # Second frame: same physical person, IDs must be stable
    det_cam03_1b = Detection(
        cam_id="cam_03",
        track_id=1,
        global_id=None,
        timestamp=t + 0.5,
        bbox_px=(101.0, 201.0, 181.0, 401.0),
        foot_point_px=(141.0, 401.0),
        foot_point_m=(5.02, 2.01),
        confidence=0.91,
        class_id=0,
        class_name="person",
    )
    det_cam07_1b = Detection(
        cam_id="cam_07",
        track_id=1,
        global_id=None,
        timestamp=t + 0.53,
        bbox_px=(201.0, 211.0, 281.0, 411.0),
        foot_point_px=(241.0, 411.0),
        foot_point_m=(5.12, 2.06),
        confidence=0.87,
        class_id=0,
        class_name="person",
    )

    result2 = fusion.associate(
        {"cam_03": [det_cam03_1b], "cam_07": [det_cam07_1b]}
    )

    print(f"\n[INFO] Frame 2 — {len(result2)} detections after fusion:")
    for d in result2:
        print(f"  {d}")

    assert det_cam03_1b.global_id == det_cam03_1.global_id, (
        "[ERR] global_id must be stable across frames for the same track"
    )

    print("\n[INFO] All assertions passed.")
