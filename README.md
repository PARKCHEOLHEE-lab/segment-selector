# segment-selector

A naive polygon segment selection test using [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) function calling for prompt-based segment selection. The repository explores possibilities for AI-assisted geometric task.

<br>

<div style="display: flex">
    <img src="segment_selector/runs/TestcaseA_0.png" width="33%">
    <img src="segment_selector/runs/TestcaseB_2.png" width="33%">
    <img src="segment_selector/runs/TestcaseC_2.png" width="33%">
</div>
<br>
<p align="center" color="gray">
  <i>Selected Segments by Agent</i>
</p>

# Installation

This repository uses the [image](/.devcontainer/Dockerfile) named `python:3.12` for running devcontainer.

1. Ensure you have Docker and Visual Studio Code with the Remote - Containers extension installed.
2. Clone the repository.

    ```
        git clone https://github.com/PARKCHEOLHEE-lab/segment-selector.git
    ```

3. Open the project with VSCode.
4. When prompted at the bottom left on the VSCode, click `Reopen in Container` or use the command palette (F1) and select `Remote-Containers: Reopen in Container`.
5. VS Code will build the Docker container and set up the environment.
6. Once the container is built and running, you're ready to start working with the project.

<br>

# File Details

### src
- `selector.py`: Implementation of the segment selector agent
- `plotter.py`: Visualizer
- `testsets.py`: Test cases, each case has a polygon boundary and prompt for selecting segments
- `main.py`: Entry point for running the segment selector

### runs
- `Testcase*.png`: Visualization results of different test cases
