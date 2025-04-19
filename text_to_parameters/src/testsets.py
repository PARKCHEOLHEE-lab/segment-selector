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
    CORES = []
    CORES_COORDS = []
    TEXTS = [
        f"다음 다각형의 우측 하단의 모든 선분을 선택하시오: {BOUNDARY_COORDS}",
        f"Select the top left segments of the following polygon: {BOUNDARY_COORDS}",
        f"Select 4 segments in the right and left of the following polygon: {BOUNDARY_COORDS}",
        f"Select randomly 3 segments of the following polygon: {BOUNDARY_COORDS}, \
          the polygon has {len(BOUNDARY_COORDS) - 1} segments",
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
    CORES = []
    CORES_COORDS = []
    TEXTS = [
        f"Select all segments of the following polygon: {BOUNDARY_COORDS}",
        f"다음 다각형의 아무 선분 1개를 선택하시오: {BOUNDARY_COORDS}",
        f"Select 2 segments in the right and left of the following polygon: {BOUNDARY_COORDS}",
        f"Select randomly half of the segments of the following polygon: {BOUNDARY_COORDS}, \
          the polygon has {len(BOUNDARY_COORDS) - 1} segments",
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
    CORES = [
        Polygon(
            [
                [27.631556, -13.228409],
                [32.79003, -13.228409],
                [32.79003, -8.397458],
                [27.631556, -8.397458],
                [27.631556, -13.228409],
            ]
        ),
        Polygon(
            [
                [0.201576, -13.883454],
                [5.36005, -13.883454],
                [5.36005, -9.052502],
                [0.201576, -9.052502],
                [0.201576, -13.883454],
            ]
        ),
    ]
    CORES_COORDS = [list(core.exterior.coords) for core in CORES]
    TEXTS = []


class TestcaseD:
    BOUNDARY = Polygon(
        [
            [29.842331, 10.435066],
            [26.988442, 17.324962],
            [20.098547, 20.17885],
            [13.208651, 17.324962],
            [10.354763, 10.435066],
            [13.208651, 3.54517],
            [20.098547, 0.691282],
            [26.988442, 3.54517],
            [29.842331, 10.435066],
        ]
    )
    BOUNDARY_COORDS = list(BOUNDARY.exterior.coords)
    CORES = []
    CORES_COORDS = []
    TEXTS = []


class TestcaseE:
    BOUNDARY = Polygon(
        [
            [11.224343, 25.934186],
            [11.224343, 13.200125],
            [15.800646, 13.200125],
            [15.800646, 16.58261],
            [23.361495, 16.58261],
            [23.361495, 19.368186],
            [28.99897, 19.368186],
            [28.99897, 21.225237],
            [36.957759, 21.225237],
            [36.957759, 31.571661],
            [31.917193, 31.571661],
            [31.917193, 28.918732],
            [25.417516, 28.918732],
            [25.417516, 28.122853],
            [17.259758, 28.122853],
            [17.259758, 25.934186],
            [11.224343, 25.934186],
        ]
    )
    BOUNDARY_COORDS = list(BOUNDARY.exterior.coords)
    CORES = [
        Polygon(
            [
                [28.203091, 23.612873],
                [34.304829, 23.612873],
                [34.304829, 26.730065],
                [28.203091, 26.730065],
                [28.203091, 23.612873],
            ]
        )
    ]
    CORES_COORDS = [list(core.exterior.coords) for core in CORES]
    TEXTS = []


class TestSets:
    for testcase in [v for k, v in globals().items() if k.startswith("Testcase")]:
        assert hasattr(testcase, "BOUNDARY")
        assert hasattr(testcase, "BOUNDARY_COORDS")
        assert hasattr(testcase, "CORES")
        assert hasattr(testcase, "CORES_COORDS")
        assert hasattr(testcase, "TEXTS")

        locals()[testcase.__name__] = testcase
        del testcase
