import os
import traceback
import numpy as np

from shapely import wkt
from pydantic import BaseModel
from typing import List, Tuple, Union
from shapely.geometry import Polygon, LineString
from agents import Agent, Runner, ModelSettings, function_tool


class SegmentSelectorOutput(BaseModel):
    """Output of the segment selector agent
    """
    # output
    selected_indices: list[int]

    # determined parameters by agent based on the prompt
    directions: list[str]
    selection_count: Union[float, int]
    exclude_core: bool
    exclude_count: Union[float, int]
    core_distance_threshold: float
    similarity_threshold: float
    shuffle: bool


class SegmentSelectorConfiguration:
    """Configuration for the segment selector agent
    """
    SELECTION_COUNT_RATIO_MIN = 0.0
    SELECTION_COUNT_RATIO_MAX = 1.0
    SELECTION_COUNT_RATIO_DEFAULT = 0.5
    CORE_DISTANCE_THRESHOLD_MIN = 0.0
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
        "right": (1, 0),          # 0 degrees
        "left": (-1, 0),          # 180 degrees
        "top": (0, 1),            # 90 degrees
        "bottom": (0, -1),        # -90 degrees

        # diagonal directions
        "right_top": (1, 1),      # 45 degrees
        "left_top": (-1, 1),      # 135 degrees
        "left_bottom": (-1, -1),  # -135 degrees
        "right_bottom": (1, -1),  # -45 degrees

        # intermediate diagonal directions
        "right_up": (0.866, 0.5),      # 30 degrees
        "up_right": (0.5, 0.866),      # 60 degrees
        "up_left": (-0.5, 0.866),      # 120 degrees
        "left_up": (-0.866, 0.5),      # 150 degrees
        "left_down": (-0.866, -0.5),   # -150 degrees
        "down_left": (-0.5, -0.866),   # -120 degrees
        "down_right": (0.5, -0.866),   # -60 degrees
        "right_down": (0.866, -0.5)    # -30 degrees
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

    ## Core polygon guidelines:
    - If core polygons are provided but no explicit instruction about excluding segments near them,
        do not exclude segments near core polygons (exclude_core will be False)
    - Only set exclude_core to True if the prompt explicitly requests to exclude segments near core polygons

    ## Direction selection guidelines:
    - You can select segments from multiple directions at once
    - If the multiple directions are requested, the order of the directions will be like:
        - left -> right -> top -> bottom -> left_top -> right_top -> left_bottom -> right_bottom ...
    - When a specific direction is requested, only include the exact direction unless explicitly asked for related directions:
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
    - core_distance_threshold: Controls how close to core polygons segments can be
        * Higher values exclude segments further from core polygons
        * Lower values only exclude segments very close to core polygons
        * Default is {CORE_DISTANCE_THRESHOLD_MIN} if not specified
        * Only used when exclude_core is True

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
    - DO NOT QUESTION THE INSTRUCTIONS, like "I need to know which specific directions or criteria you want to use for the selection" or anything like that.
    - example:
        ```json
        {{"selected_indices": [1, 2, 3]}}
        ```
    """
    

class SegmentSelector:
    def __init__(self, polygon_coords: List[List[float]], core_polygons: List[List[List[float]]], shuffle: bool):
        self.polygon = Polygon(polygon_coords)
        self.polygon_segments = SegmentSelector.explode_to_segments(self.polygon)
        self.polygon_centroid = self.polygon.centroid
        self.polygon_centroid_np = np.array(self.polygon_centroid.coords)[0]
        self.core_polygons = [Polygon(coords) for coords in core_polygons]
        self.shuffle = shuffle

        if self.shuffle:
            np.random.shuffle(self.polygon_segments)

    @staticmethod
    def explode_to_segments(polygon: Polygon) -> List[LineString]:
        coords = list(polygon.exterior.coords)
        segments = []
        for i in range(len(coords) - 1):
            segments.append(LineString([coords[i], coords[i+1]]))
        return segments

    def _divide_segment(self, segment: LineString, n: int) -> np.ndarray:

        assert n > 0

        t = segment.length / n
        t /= segment.length

        start, end = np.array(segment.coords)

        divided_points = []
        for f in range(n + 1):
            divided_points.append((1 - t * f) * start + t * f * end)

        return np.array(divided_points)


    def _compute_segments_vectors(self, segments: List[LineString]) -> List[np.ndarray]:

        vectors = []
        for segment in segments:

            divided_points = self._divide_segment(segment, SegmentSelectorConfiguration.SEGMENT_DIVISION_COUNT)

            centroid_to_segment = divided_points - self.polygon_centroid_np
            centroid_to_segment /= np.linalg.norm(centroid_to_segment, axis=1, keepdims=True)

            assert np.allclose(np.linalg.norm(centroid_to_segment, axis=1), 1.0)

            vectors.append(centroid_to_segment)

        return vectors

    def _compute_segment_distances(self, segments: List[LineString]) -> List[Tuple[int, float]]:
        if not self.core_polygons:
            return [np.inf] * len(segments)

        distances = []
        for segment in segments:
            min_distance = min(core.distance(segment) for core in self.core_polygons)
            distances.append(min_distance)

        return distances

    def _get_segment_similarities(
        self,
        target_vector: Tuple[float, float],
        exclude_core: bool,
        n_per_exclusion: int,
        core_distance_threshold: float,
        similarity_threshold: float,
    ) -> List[Tuple[int, float]]:
        segments_vectors = self._compute_segments_vectors(self.polygon_segments)
        distances_between_cores_and_segments = self._compute_segment_distances(self.polygon_segments)
        
        similarities = []
        for idx, (vector, distance) in enumerate(zip(segments_vectors, distances_between_cores_and_segments)):
            if exclude_core:
                if distance < core_distance_threshold and n_per_exclusion > 0:
                    n_per_exclusion -= 1
                    continue

            similarity = np.dot(vector, target_vector)
            mask = similarity >= similarity_threshold
            if mask.sum().item() >= len(vector) // SegmentSelectorConfiguration.MASK_MATCHING_DIVIDER:
                similarities.append((idx, similarity[mask].sum()))

        return similarities

    def _select_segments_by_vector(
        self,
        target_vector: Tuple[float, float],
        n_per_direction: int,
        exclude_core: bool,
        n_per_exclusion: int,
        core_distance_threshold: float,
        similarity_threshold: float,
    ) -> List[int]:
        similarities = self._get_segment_similarities(
            target_vector,
            exclude_core,
            n_per_exclusion,
            core_distance_threshold,
            similarity_threshold,
        )

        similarities.sort(key=lambda x: x[1], reverse=True)

        sorted_indices = [idx for idx, _ in similarities]
        
        n_segments = min(n_per_direction, len(sorted_indices))

        return sorted_indices[:n_segments]

    @staticmethod
    @function_tool
    def select_polygon_segments(
        polygon_coords: List[Tuple[float, float]],
        core_polygons: List[List[Tuple[float, float]]] = None,
        directions: List[str] = None,
        selection_count: Union[float, int] = None,
        exclude_core: bool = None,
        exclude_count: Union[float, int] = None,
        core_distance_threshold: float = None,
        similarity_threshold: float = None,
        shuffle: bool = None,
    ) -> List[int]:

        try:
            if selection_count is None:
                selection_count = np.random.randint(low=0, high=len(polygon_coords) - 1)
            if exclude_core is None:
                exclude_core = False
            if exclude_core is False or core_polygons is None:
                core_polygons = []
            if exclude_count is None:
                exclude_count = 0
            if core_distance_threshold is None:
                core_distance_threshold = SegmentSelectorConfiguration.CORE_DISTANCE_THRESHOLD_MIN
            if similarity_threshold is None:
                similarity_threshold = SegmentSelectorConfiguration.SIMILARITY_THRESHOLD_MIN
            if shuffle is None:
                shuffle = False
            if directions is None:
                directions = list(SegmentSelectorConfiguration.DIRECTION_VECTORS.keys())

            segment_selector = SegmentSelector(polygon_coords, core_polygons, shuffle)

            assert all(direction in SegmentSelectorConfiguration.DIRECTION_VECTORS for direction in directions)
            assert isinstance(selection_count, (float, int))
            assert isinstance(exclude_count, (float, int))
            
            if isinstance(selection_count, int):
                n_per_exclusion = exclude_count // len(directions)
                n_per_direction = max(1, selection_count // len(directions))
            elif isinstance(selection_count, float):
                n_per_exclusion = exclude_count // int(selection_count * (len(polygon_coords) - 1)) // len(directions)
                n_per_direction = max(
                    1, int(selection_count * (len(polygon_coords) - 1)) // len(directions)
                )

            # select segments by direction
            all_selected_indices = []
            for direction in directions:
                target_vector = SegmentSelectorConfiguration.DIRECTION_VECTORS[direction]
                selected_indices = segment_selector._select_segments_by_vector(
                    target_vector,
                    n_per_direction,
                    exclude_core,
                    n_per_exclusion,
                    core_distance_threshold,
                    similarity_threshold,
                )

                all_selected_indices.extend(selected_indices)

                if len(all_selected_indices) >= selection_count:
                    break

            # remove duplicates, sort, and slice
            return sorted(list(set(all_selected_indices)))[:selection_count]

        except:
            traceback.print_exc()
            return []

def main():

    # agent
    segment_selection_agent = Agent(
        name=SegmentSelectorConfiguration.NAME,
        instructions=SegmentSelectorConfiguration.INSTRUCTIONS,
        tools=[SegmentSelector.select_polygon_segments],
        model=SegmentSelectorConfiguration.MODEL_NAME,
        model_settings=SegmentSelectorConfiguration.MODEL_SETTINGS,
        output_type=SegmentSelectorOutput,
    )

    # boundary polygon
    boundary_wkt = "POLYGON ((-20.593598660838524 16.302766385488482, -20.593598660838524 6.415479726335377, -31.31879096907241 6.415479726335377, -31.31879096907241 -4.309712581898495, -19.839483576665856 -4.309712581898495, -19.839483576665856 -9.58851817110736, -6.768155451005811 -9.58851817110736, -6.768155451005811 -4.309712581898495, 4.45978024667653 -4.309712581898495, 4.45978024667653 -13.02393133233852, -4.505810198487723 -13.02393133233852, -4.505810198487723 -21.151616128422006, 21.217893228291956 -21.151616128422006, 21.217893228291956 16.38655695039656, -20.593598660838524 16.302766385488482))"
    boundary_geom = wkt.loads(boundary_wkt)
    boundary_coords = list(map(list, list(boundary_geom.exterior.coords)))

    # core polygon
    core_wkt = ["POLYGON ((-4.086857373947339 4.069343908909218, 9.738585835885374 4.069343908909218, 9.738585835885374 9.264358933209998, -4.086857373947339 9.264358933209998, -4.086857373947339 4.069343908909218))"]
    core_geom = [wkt.loads(x) for x in core_wkt]
    core_coords = [list(map(list, list(x.exterior.coords))) for x in core_geom]

    result = Runner.run_sync(
        segment_selection_agent,
        # f"Select all the right bottom segments of the following polygon: {boundary_coords}"
        # f"Select the top left segments of the following polygon: {boundary_coords}"
        # f"Select all segments of the following polygon, excluding those within 10 meters of the core: {boundary_coords}, core: {core_coords}"
        # f"Select randomly half number of segments of the following polygon: {boundary_coords}, the polygon has {len(boundary_coords) - 1} segments"
        f"Select randomly 3 segments of the following polygon: {boundary_coords}, the polygon has {len(boundary_coords) - 1} segments"
        # f"Select 4 segments in the right and left of the following polygon: {boundary_coords}"
    )

    output = result.final_output
    selected_segments = [SegmentSelector.explode_to_segments(boundary_geom)[i] for i in output.selected_indices]
    a=1

if __name__ == "__main__":
    main()
