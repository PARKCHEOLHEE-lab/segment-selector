import os

from agents import Agent, Runner

from plotter import Plotter
from testsets import TestSets
from selector import SegmentSelector, SegmentSelectorConfiguration, SegmentSelectorOutput


if __name__ == "__main__":
    assert os.environ.get("OPENAI_API_KEY") is not None
    agent = Agent(
        name=SegmentSelectorConfiguration.NAME,
        model=SegmentSelectorConfiguration.MODEL_NAME,
        model_settings=SegmentSelectorConfiguration.MODEL_SETTINGS,
        instructions=SegmentSelectorConfiguration.INSTRUCTIONS,
        output_type=SegmentSelectorOutput,
        tools=SegmentSelector.tools(),
    )

    runs = os.path.abspath(os.path.join(__file__, "../../runs"))

    os.makedirs(runs, exist_ok=True)

    for testcase in TestSets():
        for rqi, request in enumerate(testcase.TEXTS):
            response = Runner.run_sync(agent, request)
            print(response.final_output.__dict__)

            fig = Plotter.plot(
                testcase.BOUNDARY,
                SegmentSelector.explode_to_segments(testcase.BOUNDARY),
                response.final_output,
                title=f"{request[:request.index(':')]}",
            )

            fig.savefig(os.path.join(runs, f"{testcase.__name__}_{rqi}.png"))
