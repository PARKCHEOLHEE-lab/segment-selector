from shapely.geometry import Polygon


class TestcaseA:
    BOUNDARY = Polygon(
        [
            [10.574254, 12.994713],
            [10.574254, 10.994713],
            [4.574254, 10.994713],
            [4.574254, 6.994713],
            [7.574254, 6.994713],
            [7.574254, 3.994713],
            [14.574254, 3.994713],
            [14.574254, 1.994713],
            [19.574254, 1.994713],
            [19.574254, 8.994713],
            [16.574254, 8.994713],
            [16.574254, 12.994713],
            [10.574254, 12.994713],
        ]
    )
    BOUNDARY_COORDS = list(BOUNDARY.exterior.coords)
    TEXTS = [
        f"Select all segments in the right bottom of the following polygon: {BOUNDARY_COORDS}",
        f"Select the top left segments of the following polygon: {BOUNDARY_COORDS}",
        f"Select 4 segments in the right and left of the following polygon: {BOUNDARY_COORDS}",
        f"Select randomly 3 segments of the following polygon: {BOUNDARY_COORDS}",
    ]


class TestcaseB:
    BOUNDARY = Polygon(
        [
            [7.563796, 15.682965],
            [3.081784, 10.548234],
            [6.950549, 4.058692],
            [17.308855, 4.058692],
            [17.308855, -3.928435],
            [24.683857, -7.087369],
            [36.884057, 1.346326],
            [31.971224, 9.861902],
            [26.812751, 9.861902],
            [26.812751, 15.682965],
            [7.563796, 15.682965],
        ]
    )
    BOUNDARY_COORDS = list(BOUNDARY.exterior.coords)
    TEXTS = [
        f"Select all segments of the following polygon: {BOUNDARY_COORDS}",
        f"Select any one segment of the following polygon: {BOUNDARY_COORDS}",
        f"Select 2 segments in the right and left of the following polygon: {BOUNDARY_COORDS}",
        f"Select randomly half of the segments of the following polygon: {BOUNDARY_COORDS}",
    ]


class TestcaseC:
    BOUNDARY = Polygon(
        [
            [-3.646809, -17.731839],
            [3.558678, -23.709118],
            [19.11598, -8.72498],
            [23.455649, -18.468764],
            [35.81961, -21.00706],
            [40.814323, -14.292856],
            [33.035671, 0.773162],
            [29.596689, -2.66582],
            [16.495803, 6.668561],
            [9.372196, -1.683254],
            [-9.296566, -6.432325],
            [-3.646809, -17.731839],
        ]
    )
    BOUNDARY_COORDS = list(BOUNDARY.exterior.coords)
    TEXTS = [
        f"Select index 3 segment of the following polygon: {BOUNDARY_COORDS}",
        f"Select indices 3, 5, 6, 1 segment of the following polygon: {BOUNDARY_COORDS}",
        f"Select segments with odd indices of the following polygon: {BOUNDARY_COORDS}",
        f"Select indices any 3 segments from the indices 0, 1, 2, 3, 4 of the following polygon: {BOUNDARY_COORDS}",
    ]


class TestSets:
    def __iter__(self):
        for testcase in [v for k, v in globals().items() if k.startswith("Testcase")]:
            assert hasattr(testcase, "BOUNDARY")
            assert hasattr(testcase, "BOUNDARY_COORDS")
            assert hasattr(testcase, "TEXTS")

            yield testcase
