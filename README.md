# Conversation Evaluation Framework



## Purpose

The Conversation Evaluation project uses Large Language Models (LLMs) as a judge to evaluate conversational agents. This helps to reduce the manual effort required for evaluation.

<img width="524" height="250" alt="image" src="https://github.com/user-attachments/assets/266cab39-78af-4923-9fab-5de269f03443" />

## Target Audience

This project is for anyone who wants to reduce the manual effort involved in evaluating conversational agents.

## Key Features

*   **Simulation:** Simulate conversations with a user simulation.
*   **AI Library Support:** Supports majority AI inference libraries Together, OpenAI, Google and served vLLM, OLLAMA models.
*   **LLM as Judge:** Uses LLMs to evaluate conversational agents.

## Installation

To install the project, install the required packages from the requirements file:

```bash
pip install -r requirements.txt
```

## Usage

To run the evaluation, you need to provide a configuration file in YAML or JSON format to the `main.py` script.
Create a .env file define the api_keys there.
```bash
TOGETHER_API_KEY="your_together_api_key"
OPENAI_API_KEY="your_openai_key"
```

```bash
python main.py config.yaml
```
It will return a json file containing the evaluation results.


## Configuration
A yaml config file is created you can see the reference config.yaml file and output result.json

#TODO
Define the updated readme file
Create a UI Dashboard
Guiding to use conversation_insight_generation tool.
