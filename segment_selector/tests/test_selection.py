import os
import sys
import pytest

from agents import Agent, Runner

sys.path.append(os.path.abspath(os.path.join(__file__, "../../")))

from src.testsets import TestSets
from src.selector import SegmentSelector, SegmentSelectorConfiguration, SegmentSelectorOutput


@pytest.fixture(scope="module")
def agent():
    assert os.environ.get("OPENAI_API_KEY") is not None
    return Agent(
        name=SegmentSelectorConfiguration.NAME,
        model=SegmentSelectorConfiguration.MODEL_NAME,
        model_settings=SegmentSelectorConfiguration.MODEL_SETTINGS,
        instructions=SegmentSelectorConfiguration.INSTRUCTIONS,
        output_type=SegmentSelectorOutput,
        tools=SegmentSelector.tools(),
    )


@pytest.fixture(scope="module")
def testsets():
    return TestSets()


def test_a(agent, testsets):
    testcase_a = testsets.TestcaseA

    # f"Select all segments in the right bottom of the following polygon: {BOUNDARY_COORDS}"
    request = testcase_a.TEXTS[0]
    response = Runner.run_sync(agent, request)
    assert 8 in response.final_output.selected_indices
    assert 7 in response.final_output.selected_indices
    assert 1 not in response.final_output.selected_indices
    assert 2 not in response.final_output.selected_indices

    # f"Select the top left segments of the following polygon: {BOUNDARY_COORDS}"
    request = testcase_a.TEXTS[1]
    response = Runner.run_sync(agent, request)
    assert 1 in response.final_output.selected_indices
    assert 2 in response.final_output.selected_indices
    assert 8 not in response.final_output.selected_indices
    assert 7 not in response.final_output.selected_indices

    # f"Select 4 segments in the right and left of the following polygon: {BOUNDARY_COORDS}"
    request = testcase_a.TEXTS[2]
    response = Runner.run_sync(agent, request)
    assert 2 in response.final_output.selected_indices
    assert 8 in response.final_output.selected_indices
    assert len(response.final_output.selected_indices) == 4

    # f"Select randomly 3 segments of the following polygon: {BOUNDARY_COORDS}"
    request = testcase_a.TEXTS[3]
    response = Runner.run_sync(agent, request)
    assert len(response.final_output.selected_indices) == 3


def test_b(agent, testsets):
    testcase_b = testsets.TestcaseB

    # f"Select all segments of the following polygon: {BOUNDARY_COORDS}"
    request = testcase_b.TEXTS[0]
    response = Runner.run_sync(agent, request)
    assert len(response.final_output.selected_indices) == 10

    # f"Select any one segment of the following polygon: {BOUNDARY_COORDS}"
    request = testcase_b.TEXTS[1]
    response = Runner.run_sync(agent, request)
    assert len(response.final_output.selected_indices) == 1

    # f"Select 2 segments in the right and left of the following polygon: {BOUNDARY_COORDS}"
    request = testcase_b.TEXTS[2]
    response = Runner.run_sync(agent, request)
    assert len(response.final_output.selected_indices) == 2
    assert 0 in response.final_output.selected_indices or 1 in response.final_output.selected_indices
    assert 5 in response.final_output.selected_indices or 6 in response.final_output.selected_indices

    # f"Select randomly half of the segments of the following polygon: {BOUNDARY_COORDS}"
    request = testcase_b.TEXTS[3]
    response = Runner.run_sync(agent, request)
    assert len(response.final_output.selected_indices) == 5


def test_c(agent, testsets):
    testcase_c = testsets.TestcaseC

    # f"Select index 3 segment of the following polygon: {BOUNDARY_COORDS}"
    request = testcase_c.TEXTS[0]
    response = Runner.run_sync(agent, request)
    assert len(response.final_output.selected_indices) == 1
    assert response.final_output.selected_indices[0] == 3

    # f"Select indices 3, 5, 6, 1 segment of the following polygon: {BOUNDARY_COORDS}"
    request = testcase_c.TEXTS[1]
    response = Runner.run_sync(agent, request)
    assert len(response.final_output.selected_indices) == 4
    assert sorted(response.final_output.selected_indices) == sorted([3, 5, 6, 1])

    # f"Select segments with odd indices of the following polygon: {BOUNDARY_COORDS}"
    request = testcase_c.TEXTS[2]
    response = Runner.run_sync(agent, request)
    assert all(i % 2 == 1 for i in response.final_output.selected_indices)

    # f"Select indices any 3 segments from the indices 0, 1, 2, 3, 4 of the following polygon: {BOUNDARY_COORDS}"
    request = testcase_c.TEXTS[3]
    response = Runner.run_sync(agent, request)
    assert all(i in [0, 1, 2, 3, 4] for i in response.final_output.selected_indices)
