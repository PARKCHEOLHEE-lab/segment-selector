# noqa: E722

import traceback
import numpy as np

from pydantic import BaseModel
from typing import List, Tuple, Union
from shapely.geometry import Polygon, LineString
from agents import ModelSettings, FunctionTool, function_tool


class SegmentSelectorOutput(BaseModel):
    """`final_output` of the segment selector agent"""

    selected_indices: list[int]

    directions: list[str]
    selection_count: Union[float, int]
    similarity_threshold: float


class SegmentSelectorConfiguration:
    """Configuration for the segment selector agent"""

    SELECTION_COUNT_RATIO_MIN = 0.0
    SELECTION_COUNT_RATIO_MAX = 1.0
    SELECTION_COUNT_RATIO_DEFAULT = 0.5
    SIMILARITY_THRESHOLD_MIN = 0.25
    SIMILARITY_THRESHOLD_MAX = 0.95
    SEGMENT_DIVISION_COUNT = 6
    MASK_MATCHING_DIVIDER = 2.5

    NAME = "segment_selector"
    MODEL_NAME = "gpt-4o-2024-08-06"
    MODEL_SETTINGS = ModelSettings(
        temperature=0.0,
        top_p=0.0,
        max_tokens=1000,
        tool_choice="required",
    )

    DIRECTION_VECTORS = {
        # base directions
        "right": (1, 0),  # 0 degrees
        "left": (-1, 0),  # 180 degrees
        "top": (0, 1),  # 90 degrees
        "bottom": (0, -1),  # -90 degrees
        # diagonal directions
        "right_top": (1, 1),  # 45 degrees
        "left_top": (-1, 1),  # 135 degrees
        "left_bottom": (-1, -1),  # -135 degrees
        "right_bottom": (1, -1),  # -45 degrees
        # intermediate diagonal directions
        "right_up": (0.866, 0.5),  # 30 degrees
        "up_right": (0.5, 0.866),  # 60 degrees
        "up_left": (-0.5, 0.866),  # 120 degrees
        "left_up": (-0.866, 0.5),  # 150 degrees
        "left_down": (-0.866, -0.5),  # -150 degrees
        "down_left": (-0.5, -0.866),  # -120 degrees
        "down_right": (0.5, -0.866),  # -60 degrees
        "right_down": (0.866, -0.5),  # -30 degrees
    }

    # normalize direction vectors
    for key, value in DIRECTION_VECTORS.items():
        DIRECTION_VECTORS[key] = np.array(value) / np.linalg.norm(np.array(value))

    assert np.allclose(np.linalg.norm(list(DIRECTION_VECTORS.values()), axis=1), 1.0)

    INSTRUCTIONS = f"""
    You are an agent that helps select polygon segments based on direction.
    You can select segments from the following directions:

    Basic directions:
    - right: {DIRECTION_VECTORS["right"]} (0 degrees)
    - left: {DIRECTION_VECTORS["left"]} (180 degrees)
    - top: {DIRECTION_VECTORS["top"]} (90 degrees)
    - bottom: {DIRECTION_VECTORS["bottom"]} (-90 degrees)

    Diagonal directions:
    - right_top: {DIRECTION_VECTORS["right_top"]} (45 degrees)
    - left_top: {DIRECTION_VECTORS["left_top"]} (135 degrees)
    - left_bottom: {DIRECTION_VECTORS["left_bottom"]} (-135 degrees)
    - right_bottom: {DIRECTION_VECTORS["right_bottom"]} (-45 degrees)

    Intermediate diagonal directions:
    - right_up: {DIRECTION_VECTORS["right_up"]} (30 degrees)
    - up_right: {DIRECTION_VECTORS["up_right"]} (60 degrees)
    - up_left: {DIRECTION_VECTORS["up_left"]} (120 degrees)
    - left_up: {DIRECTION_VECTORS["left_up"]} (150 degrees)
    - left_down: {DIRECTION_VECTORS["left_down"]} (-150 degrees)
    - down_left: {DIRECTION_VECTORS["down_left"]} (-120 degrees)
    - down_right: {DIRECTION_VECTORS["down_right"]} (-60 degrees)
    - right_down: {DIRECTION_VECTORS["right_down"]} (-30 degrees)

    The polygon coordinates should be provided as a list of [x, y] coordinates.

    ## Direction selection guidelines:
    - You can select segments from multiple directions at once
    - If the multiple directions are requested, the order of the directions will be like:
        - left -> right -> top -> bottom -> left_top -> right_top -> left_bottom -> right_bottom ...
    - When a specific direction is requested, only include the exact direction unless asked for related directions:
        * For "bottom" requests:
            - Example: "Select bottom segments" → ["bottom", "right_bottom", "left_bottom", "right_down", "left_down"]
        * For "right" requests:
            - Example: "Select right segments" → ["right", "right_top", "right_bottom", "right_up", "right_down"]
        * For "top" requests:
            - Example: "Select top segments" → ["top", "right_top", "left_top", "up_right", "up_left"]
        * For "left" requests:
            - Example: "Select left segments" → ["left", "left_top", "left_bottom", "left_up", "left_down"]
        * ...

    ## Threshold guidelines:
    - similarity_threshold: Controls how strictly segments must match the target direction
        * Base directions (right, left, top, bottom):
            - Use higher threshold (0.75 ~ 0.85) for strict alignment
        * Diagonal directions (right_top, right_bottom, left_bottom, left_top):
            - Use medium threshold (0.25 ~ 0.35) for diagonal alignment
        * Intermediate directions (right_up, up_right, up_left, left_up, left_down, down_left, down_right, right_down):
            - Use lower threshold (0.25 ~ 0.55) for more flexible selection
        * Opposite directions requested (left and right, top and bottom, etc.):
            - Use lower threshold (0.45 ~ 0.65) for more flexible selection
        * Default value is {SIMILARITY_THRESHOLD_MIN} if not specified

    ## IMPORTANT:
    - You must return a JSON object with a single key "selected_indices" containing the list of indices.
    - DO NOT QUESTION THE INSTRUCTIONS, like:
        - I need to know which specific directions or criteria you want to use for the selection" or anything like that.
    - example:
        ```json
        {{"selected_indices": [1, 2, 3]}}
        ```
    """


class SegmentSelector:
    def __init__(self, polygon_coords: List[List[float]]):
        self.polygon = Polygon(polygon_coords)
        self.polygon_segments = SegmentSelector.explode_to_segments(self.polygon)
        self.polygon_centroid = self.polygon.centroid
        self.polygon_centroid_np = np.array(self.polygon_centroid.coords)[0]

    def _divide_segment(self, segment: LineString, n: int) -> List[List[float]]:
        assert n > 0

        t = segment.length / n
        t /= segment.length

        start, end = np.array(segment.coords)

        divided_points = []
        for f in range(n + 1):
            divided_points.append((1 - t * f) * start + t * f * end)

        return np.array(divided_points).tolist()

    def _compute_segments_vectors(self, segments: List[LineString]) -> List[List[float]]:
        vectors = []
        for segment in segments:
            divided_points = self._divide_segment(segment, SegmentSelectorConfiguration.SEGMENT_DIVISION_COUNT)

            centroid_to_segment = divided_points - self.polygon_centroid_np
            centroid_to_segment /= np.linalg.norm(centroid_to_segment, axis=1, keepdims=True)

            assert np.allclose(np.linalg.norm(centroid_to_segment, axis=1), 1.0)

            vectors.append(centroid_to_segment.tolist())

        return vectors

    def _get_segment_similarities(
        self,
        target_vector: Tuple[float, float],
        similarity_threshold: float,
    ) -> List[Tuple[int, float]]:
        segments_vectors = self._compute_segments_vectors(self.polygon_segments)

        similarities = []
        for idx, vector in enumerate(segments_vectors):
            similarity = np.dot(vector, target_vector)
            mask = similarity >= similarity_threshold
            if mask.sum().item() >= len(vector) // SegmentSelectorConfiguration.MASK_MATCHING_DIVIDER:
                similarities.append((idx, similarity[mask].sum()))

        return similarities

    def _select_segments_by_vector(
        self,
        target_vector: Tuple[float, float],
        n_per_direction: int,
        similarity_threshold: float,
    ) -> List[int]:
        similarities = self._get_segment_similarities(
            target_vector,
            similarity_threshold,
        )

        similarities.sort(key=lambda x: x[1], reverse=True)

        sorted_indices = [idx for idx, _ in similarities]

        return sorted_indices[:n_per_direction]

    @staticmethod
    def explode_to_segments(polygon: Polygon) -> List[LineString]:
        coords = list(polygon.exterior.coords)
        segments = []
        for i in range(len(coords) - 1):
            segments.append(LineString([coords[i], coords[i + 1]]))
        return segments

    @staticmethod
    def tools() -> List[FunctionTool]:
        """Automatically collect all function tools"""
        tools = []
        tools_str = "tools:\n"
        for value in SegmentSelector.__dict__.values():
            if isinstance(value, staticmethod) and isinstance(value.__func__, FunctionTool):
                tools.append(value.__func__)
                tools_str += f"    {value.__func__.name}()\n"
        print(tools_str)

        return tools

    @staticmethod
    @function_tool
    def select_randomly(selection_count: int, segments_count: int) -> List[int]:
        """Select a random number of segments from the polygon"""
        try:
            return np.random.permutation(segments_count)[:selection_count].tolist()

        except:
            traceback.print_exc()
            raise Exception

    @staticmethod
    @function_tool
    def calculate_segments_count(polygon_coords: List[List[float]]) -> int:
        """Calculate the number of segments in the polygon"""

        try:
            polygon = Polygon(polygon_coords)
            segments = SegmentSelector.explode_to_segments(polygon)
            return len(segments)

        except:
            traceback.print_exc()
            raise Exception

    @staticmethod
    @function_tool
    def reset_directions_order(directions: List[str]) -> List[str]:
        """Reset the order of directions in the list"""

        try:
            order_map = {
                "right": 0,
                "left": 1,
                "top": 2,
                "bottom": 3,
                "right_top": 4,
                "left_top": 5,
                "right_bottom": 6,
                "left_bottom": 7,
                "right_up": 8,
                "left_up": 9,
                "up_right": 10,
                "up_left": 11,
                "down_right": 12,
                "down_left": 13,
                "right_down": 14,
                "left_down": 15,
            }

            return sorted(directions, key=lambda x: order_map.get(x, float("inf")))

        except:
            traceback.print_exc()
            raise Exception

    @staticmethod
    @function_tool
    def get_target_vector(direction: str) -> List[float]:
        """Get the vector for a given direction"""
        try:
            return SegmentSelectorConfiguration.DIRECTION_VECTORS[direction].tolist()

        except:
            traceback.print_exc()
            raise Exception

    @staticmethod
    @function_tool
    def calculate_n_per_direction(
        selection_count: Union[float, int], directions_count: int, polygon_coords_count: int
    ) -> int:
        """Calculate the number of segments to select per direction"""

        try:
            if isinstance(selection_count, int):
                return int(max(1, selection_count // directions_count))
            elif isinstance(selection_count, float):
                return int(max(1, selection_count * (polygon_coords_count - 1)) // directions_count)

        except:
            traceback.print_exc()
            raise Exception

    @staticmethod
    @function_tool
    def select_all_indices(segments_count: int) -> List[int]:
        """Get all indices of the segments"""
        return list(range(segments_count))

    @staticmethod
    @function_tool
    def select_segments_indices_by_index(
        selected_indices: List[int],
    ) -> List[int]:
        """Select segments by index"""
        try:
            return selected_indices

        except:
            traceback.print_exc()
            raise Exception

    @staticmethod
    @function_tool
    def select_segments_indices_by_target_vector(
        polygon_coords: List[List[float]],
        target_vector: List[float],
        n_per_direction: int,
        similarity_threshold: float,
    ) -> List[int]:
        """Select segments for a specific direction"""
        try:
            segment_selector = SegmentSelector(polygon_coords)
            return segment_selector._select_segments_by_vector(
                target_vector,
                n_per_direction,
                similarity_threshold,
            )

        except:
            traceback.print_exc()
            raise Exception
