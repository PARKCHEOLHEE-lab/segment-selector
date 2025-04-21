import matplotlib.pyplot as plt

from shapely.geometry import Polygon, LineString
from typing import List

from selector import SegmentSelectorOutput


class Plotter:
    @staticmethod
    def plot(
        polygon: Polygon,
        segments: List[LineString],
        final_output: SegmentSelectorOutput,
        title: str = "",
    ):
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.scatter(polygon.exterior.xy[0], polygon.exterior.xy[1], color="black", s=5)
        ax.plot(polygon.exterior.xy[0], polygon.exterior.xy[1], color="black", linewidth=0.5)

        for si, segment in enumerate(segments):
            color = "black"
            if si in final_output.selected_indices:
                color = "red"

            ax.plot(segment.xy[0], segment.xy[1], color=color, linewidth=0.9)

            ax.text(
                segment.centroid.xy[0][0],
                segment.centroid.xy[1][0],
                f"{si}",
                fontsize=7,
                ha="center",
                va="bottom",
                color=color,
            )

        title = f"{title}\n selected indices: {final_output.selected_indices}"

        ax.set_xlim(auto=True)
        ax.set_ylim(auto=True)
        ax.set_title(title, fontsize=8)
        ax.tick_params(axis="both", which="major", labelsize=7)
        ax.tick_params(axis="both", which="minor", labelsize=7)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.5)

        return fig
