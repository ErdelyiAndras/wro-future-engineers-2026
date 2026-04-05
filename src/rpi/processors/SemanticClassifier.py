from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation
from sklearn.cluster import DBSCAN

from processors.Processor import Processor
from control import FieldMap, CellLabel
from utils import mm, degree

@dataclass
class LineSegment:
    endpoints:       np.ndarray  # (2, 2)
    inlier_points:   np.ndarray  # (N, 2)
    linearity_ratio: float = 0.0
    rms_distance:    mm    = 0.0
    point_density:   float = 0.0 # points / mm

    @property
    def centroid(self) -> np.ndarray:
        return (self.endpoints[0] + self.endpoints[1]) / 2

    @property
    def direction(self) -> np.ndarray:
        diff = self.endpoints[1] - self.endpoints[0]
        norm = np.linalg.norm(diff)
        return diff / norm if norm > 1e-10 else diff

    @property
    def length(self) -> mm:
        return mm(np.linalg.norm(self.endpoints[1] - self.endpoints[0]))


@dataclass
class PointCluster:
    points: np.ndarray # (N, 2)

    @property
    def centroid(self) -> np.ndarray:
        return self.points.mean(axis = 0)

    @property
    def bounding_box_extent(self) -> mm:
        return mm(np.max(self.points.max(axis = 0) - self.points.min(axis = 0)))

    @property
    def linearity_ratio(self) -> float:
        if len(self.points) < 2:
            return 0.0
        centered = self.points - self.centroid
        _, s, _  = np.linalg.svd(centered, full_matrices = False)
        return float(s[0] ** 2 / (s[0] ** 2 + s[1] ** 2 + 1e-10))


class SemanticClassifier(Processor):
    # DBSCAN
    _DBSCAN_EPS:         mm  = 30.0
    _DBSCAN_MIN_SAMPLES: int = 5

    # RANSAC
    _RANSAC_ITERATIONS:      int = 50
    _INLIER_THRESHOLD:       mm  = 10.0
    _MAX_CONSECUTIVE_MISSES: int = 4

    # Obstacle
    _MAX_OBSTACLE_EXTENT: mm  = 100.0
    _MIN_OBSTACLE_POINTS: int = 5

    # Segment quality
    _MIN_LINEARITY_RATIO: float = 0.92
    _MAX_RMS_DISTANCE:    mm    = 8.0

    # Gap splitting
    _MAX_GAP_LENGTH: mm = 80.0

    # Rejoin and wall classification
    _MIN_SUBSEGMENT_LENGTH: mm  = 150.0
    _MIN_WALL_LENGTH:       mm  = 600.0
    _MAX_NUMBER_OF_WALLS:   int = 8

    # Parking wall
    _MIN_PARKING_WALL_LENGTH:     mm     = 80.0
    _MAX_PARKING_WALL_LENGTH:     mm     = 260.0
    _MAX_NUMBER_OF_PARKING_WALLS: int    = 2
    _MAX_PARKING_PAIR_ANGLE:      degree = 10.0
    _MIN_PARKING_PAIR_GAP:        mm     = 150.0
    _MAX_PARKING_PAIR_GAP:        mm     = 600.0

    # Label propagation
    _PROPAGATION_CELLS: int = 2

    def __init__(self, field_map: FieldMap) -> None:
        super().__init__()
        self._field_map: FieldMap = field_map

    def _process(self) -> None:
        coords, labels = self._field_map.get_occupied_world()

        # unknown_mask = labels == int(CellLabel.UNKNOWN)
        # coords       = coords[unknown_mask]
        if len(coords) < 2:
            return

        clusters = self._dbscan_cluster(coords)

        obstacle_clusters, wall_points = self._classify_clusters(clusters)

        self._field_map.update_obstacles([c.points for c in obstacle_clusters])

        if len(wall_points) < 2:
            self._propagate_labels()
            return

        wall_groups, parking_candidates = self._detect_walls(wall_points)

        for group in wall_groups:
            for seg in group:
                self._field_map.set_label(
                    self._sample_segment(seg.endpoints),
                    CellLabel.WALL,
                    occupancy_threshold = 0.0,
                )

        parking_walls = self._validate_parking_wall_pair(parking_candidates)
        for seg in parking_walls:
            self._field_map.set_label(
                self._sample_segment(seg.endpoints),
                CellLabel.PARKING_WALL,
                occupancy_threshold = 0.0,
            )

        self._propagate_labels()

    def _dbscan_cluster(self, points: np.ndarray) -> list[PointCluster]:
        db     = DBSCAN(eps = self._DBSCAN_EPS, min_samples = self._DBSCAN_MIN_SAMPLES)
        labels = db.fit_predict(points)

        clusters = []
        for label in set(labels):
            if label == -1:
                continue
            clusters.append(PointCluster(points[labels == label]))

        return clusters

    def _classify_clusters(
        self,
        clusters: list[PointCluster],
    ) -> tuple[list[PointCluster], np.ndarray]:
        obstacle_clusters  = []
        wall_point_batches = []

        for cluster in clusters:
            if cluster.bounding_box_extent <= self._MAX_OBSTACLE_EXTENT:
                if len(cluster.points) >= self._MIN_OBSTACLE_POINTS:
                    obstacle_clusters.append(cluster)
            else:
                wall_point_batches.append(cluster.points)

        wall_points = np.vstack(wall_point_batches) \
                      if wall_point_batches \
                      else np.empty((0, 2))

        return obstacle_clusters, wall_points

    def _detect_walls(
        self,
        points: np.ndarray,
    ) -> tuple[list[list[LineSegment]], list[LineSegment]]:
        max_iter  = self._MAX_NUMBER_OF_WALLS + self._MAX_NUMBER_OF_PARKING_WALLS
        remaining = points.copy()
        wall_groups:        list[list[LineSegment]] = []
        parking_candidates: list[LineSegment]       = []
        misses = 0

        for _ in range(max_iter):
            if len(remaining) < 2:
                break
            if misses >= self._MAX_CONSECUTIVE_MISSES:
                break

            inlier_mask  = self._ransac_line(remaining)
            if not inlier_mask.any():
                break

            inliers      = remaining[inlier_mask]
            sub_segments = self._fit_line_segments_with_metrics(inliers)

            quality_subs = [
                s for s in sub_segments
                if s.linearity_ratio >= self._MIN_LINEARITY_RATIO and
                   s.rms_distance    <= self._MAX_RMS_DISTANCE
            ]

            if not quality_subs:
                misses += 1
                continue

            misses    = 0
            remaining = remaining[~inlier_mask]

            new_wall_groups, new_parking = self._rejoin_subsegments(quality_subs)
            wall_groups.extend(new_wall_groups)
            parking_candidates.extend(new_parking)

        return wall_groups, parking_candidates

    def _rejoin_subsegments(
        self,
        sub_segments: list[LineSegment],
    ) -> tuple[list[list[LineSegment]], list[LineSegment]]:
        groups:  list[list[LineSegment]] = []
        current: list[LineSegment]       = []

        for seg in sub_segments:
            if seg.length >= self._MIN_SUBSEGMENT_LENGTH:
                current.append(seg)
            else:
                if current:
                    groups.append(current)
                current = []

        if current:
            groups.append(current)

        wall_groups:        list[list[LineSegment]] = []
        parking_candidates: list[LineSegment]       = []

        for group in groups:
            span = mm(np.linalg.norm(
                group[-1].endpoints[1] - group[0].endpoints[0]
            ))

            if span >= self._MIN_WALL_LENGTH:
                wall_groups.append(group)
            elif len(group) == 1 and \
                 self._MIN_PARKING_WALL_LENGTH <= span <= self._MAX_PARKING_WALL_LENGTH:
                parking_candidates.append(group[0])

        return wall_groups, parking_candidates

    def _validate_parking_wall_pair(
        self,
        candidates: list[LineSegment],
    ) -> list[LineSegment]:
        if len(candidates) < 2:
            return []

        best_pair:  list[LineSegment] = []
        best_score: float             = float('inf')

        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                a, b = candidates[i], candidates[j]

                dot       = min(abs(float(a.direction @ b.direction)), 1.0)
                angle_deg = np.degrees(np.arccos(dot))
                if angle_deg > self._MAX_PARKING_PAIR_ANGLE:
                    continue

                gap = mm(np.linalg.norm(a.centroid - b.centroid))
                if not (self._MIN_PARKING_PAIR_GAP <= gap <= self._MAX_PARKING_PAIR_GAP):
                    continue

                nominal_gap = (self._MIN_PARKING_PAIR_GAP + self._MAX_PARKING_PAIR_GAP) / 2
                score       = angle_deg + abs(gap - nominal_gap)
                if score < best_score:
                    best_score = score
                    best_pair  = [a, b]

        return best_pair

    def _propagate_labels(self) -> None:
        occupancy, semantic, _ = self._field_map.snapshot()

        unknown_occupied = (semantic == int(CellLabel.UNKNOWN)) & (occupancy > 0.0)
        updates: dict[CellLabel, np.ndarray] = {}

        for label in (CellLabel.WALL, CellLabel.PARKING_WALL):
            label_mask = semantic == int(label)
            if not label_mask.any():
                continue

            dilated = binary_dilation(label_mask, iterations = self._PROPAGATION_CELLS)
            to_fill = dilated & unknown_occupied
            if not to_fill.any():
                continue

            updates[label] = to_fill
            unknown_occupied &= ~to_fill

        if not updates:
            return

        self._field_map.apply_semantic_updates(updates)

    def _ransac_line(self, points: np.ndarray) -> np.ndarray:
        n         = len(points)
        best_mask = np.zeros(n, dtype = bool)

        for _ in range(self._RANSAC_ITERATIONS):
            i, j = np.random.choice(n, 2, replace = False)
            d    = points[j] - points[i]
            norm = np.linalg.norm(d)
            if norm < 1e-10:
                continue

            d   /= norm
            mask = self._point_line_distances(points, points[i], d) < self._INLIER_THRESHOLD

            if mask.sum() > best_mask.sum():
                best_mask = mask

        return best_mask

    def _fit_line_segments_with_metrics(self, points: np.ndarray) -> list[LineSegment]:
        centroid    = points.mean(axis = 0)
        _, _, vh    = np.linalg.svd(points - centroid, full_matrices = False)
        direction   = vh[0]
        projections = (points - centroid) @ direction

        order       = np.argsort(projections)
        sorted_proj = projections[order]
        sorted_pts  = points[order]

        gaps       = np.diff(sorted_proj) > self._MAX_GAP_LENGTH
        boundaries = np.concatenate([[0], np.where(gaps)[0] + 1, [len(sorted_proj)]])

        normal   = np.array([-direction[1], direction[0]])
        segments = []

        for k in range(len(boundaries) - 1):
            start = boundaries[k]
            end   = boundaries[k + 1]
            if end - start < 2:
                continue

            group_pts  = sorted_pts[start:end]
            group_proj = sorted_proj[start:end]

            endpoint1 = centroid + group_proj[ 0] * direction
            endpoint2 = centroid + group_proj[-1] * direction
            length    = mm(np.linalg.norm(endpoint2 - endpoint1))
            if length == 0:
                continue

            group_centered  = group_pts - group_pts.mean(axis = 0)
            _, s_sub, _     = np.linalg.svd(group_centered, full_matrices = False)
            linearity_ratio = float(s_sub[0] ** 2 / (s_sub[0] ** 2 + s_sub[1] ** 2 + 1e-10))

            distances     = np.abs((group_pts - centroid) @ normal)
            rms_distance  = mm(np.sqrt(np.mean(distances ** 2)))
            point_density = len(group_pts) / length

            segments.append(LineSegment(
                endpoints       = np.stack([endpoint1, endpoint2]),
                inlier_points   = group_pts,
                linearity_ratio = linearity_ratio,
                rms_distance    = rms_distance,
                point_density   = point_density,
            ))

        return segments

    @staticmethod
    def _point_line_distances(
        points:    np.ndarray,
        centroid:  np.ndarray,
        direction: np.ndarray,
    ) -> np.ndarray:
        normal = np.array([-direction[1], direction[0]])
        return np.abs((points - centroid) @ normal)

    @staticmethod
    def _sample_segment(endpoints: np.ndarray) -> np.ndarray:
        p1, p2 = endpoints[0].astype(np.float64), endpoints[1].astype(np.float64)
        length = mm(np.linalg.norm(p2 - p1))
        if length == 0:
            return p1[np.newaxis]

        n = max(2, int(length / FieldMap.CELL_SIZE) + 1)
        t = np.linspace(0.0, 1.0, n)[:, np.newaxis]
        return p1 + t * (p2 - p1)
