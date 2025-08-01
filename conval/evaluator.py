import json
from conval.agent import Agent, AsyncAgent
from conval.parameters import JUDGE_PROMPT, JUDGE_PROMPT1, METRIC_GENERATION_PROMPT
from typing import List
from conval.utils import get_metrics_instructions
import asyncio
from typing import List, Dict, Any
class Metric(object):
    """
    A class to represent a metric.
    Attributes
    ----------
    name : str
        name of the metric
    definition : str
        definition of the metric
    scoring_criteria : str
        scoring criteria for the metric
    """
    def __init__(self, name: str, definition: str, scoring_criteria: str=None):
        self.name = name
        self.definition = definition
        self.scoring_criteria = scoring_criteria
    
    def __str__(self):
        metric_repr = f"Metric: {self.name}\nDefinition: {self.definition}"
        if self.scoring_criteria:
            metric_repr += f"\nScoring Criteria: {self.scoring_criteria}"
        return metric_repr
        return f"Metric(name={self.name}, definition={self.definition}, scoring_criteria={self.scoring_criteria})"
    #return json
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the Metric instance to a dictionary.
        Returns
        -------
        Dict[str, Any]
            Dictionary representation of the Metric instance.
        """
        return {
            "name": self.name,
            "definition": self.definition,
            "scoring_criteria": self.scoring_criteria
        }
    def from_dict(cls, data: Dict[str, Any]) -> 'Metric':
        """
        Create a Metric instance from a dictionary.
        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary containing the metric data.
        Returns
        -------
        Metric
            A Metric instance created from the provided dictionary.
        """
        return cls(
            name=data.get("name", ""),
            definition=data.get("definition", ""),
            scoring_criteria=data.get("scoring_criteria")
        )

class TaskCompleteness(Metric):
    def __init__(self, definition: str, scoring_crieteria: str=None):
        self.name = "Task Completeness"
        
        self.definition = """
        Task Completeness: Compare the conversation between the user and LLM agent with the user goal. Check if the agent's response is complete and covers all the aspects of the user goal. Task Completeness can be determined by checking the following.

        {definition}""".format(definition=definition)

        self.scoring_criteria = """Give a binary PASS/FAIL DECISION if the AGENT has fully completed the TASK. ALSO GIVE the REASONING of the DECISION.
        Return 0 if Task is Failed else 1 if Task is Passed.
        """
        if scoring_crieteria is not None:
            self.scoring_criteria = scoring_crieteria


    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the Metric instance to a dictionary.
        Returns
        -------
        Dict[str, Any]
            Dictionary representation of the Metric instance.
        """
        return {
            "name": self.name,
            "definition": self.definition,
            "scoring_criteria": self.scoring_criteria
        }
    def from_dict(cls, data: Dict[str, Any]) -> 'Metric':
        """
        Create a Metric instance from a dictionary.
        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary containing the metric data.
        Returns
        -------
        Metric
            A Metric instance created from the provided dictionary.
        """
        return cls(
            name=data.get("name", ""),
            definition=data.get("definition", ""),
            scoring_criteria=data.get("scoring_criteria", None)
        )


async def evaluate_metric_with_gold(gold_conversation: str, simulated_conversation: str, metrics: List[Metric]=[], judge_agent: AsyncAgent=None):
    """
    Evaluate the simulated conversation against the golden conversation using the provided metrics.
    Parameters
    ----------
    gold_conversation : str
        The golden conversation string.
    simulated_conversation : str
        The simulated conversation string.
    metrics : List[Metrics]
        List of metrics to evaluate the conversation.
    Returns
    -------
    str
        The evaluation result as a dict.
    """
    evaluation_results = {}
    tasks = []
    for metric in metrics:
        judge_agent.prompt = JUDGE_PROMPT1.format(metric_name=metric.name, metric=metric.definition+'\n'+metric.scoring_criteria, golden_conversation=gold_conversation, user_agent_conversation=simulated_conversation)
            
        tasks.append(judge_agent.generate_response(json_resp=True))
    scores = await asyncio.gather(*tasks)

    for metric, score in zip(metrics, scores):
        evaluation_results[metric.name] = score

    return evaluation_results



async def evaluate_metric(conversation: str, metrics: List[Metric]=[], judge_agent: AsyncAgent=None):
    """
    Evaluate the simulated conversation against the user goal using the provided metrics.
    Parameters
    ----------
    conversation : str
        conversation string.
    metrics : List[Metrics]
        List of metrics to evaluate the conversation.
    Returns
    -------
    str
        The evaluation result as a dict.
    """
    evaluation_results = {}
    tasks = []
    for metric in metrics:
        judge_agent.prompt = JUDGE_PROMPT.format(metric_name=metric.name, metric=metric.definition+'\n'+metric.scoring_criteria, user_agent_conversation=conversation)
        tasks.append(judge_agent.generate_response(json_resp=True))
    scores = await asyncio.gather(*tasks)

    for metric, score in zip(metrics, scores):
        evaluation_results[metric.name] = score

    return evaluation_results
    
async def evaluate_metric_on_listofconversation(conversation: List[str], metric: Metric, model="gpt-4o-mini",provider="openai", gather=False):
    """
    Evaluate a list of conversations against the user goal using the provided metric.
    Parameters
    ----------
    conversation : List[str]
        List of conversation strings.
    metric : Metric
        Metric to evaluate the conversation.
    Returns
    -------
    # Output is List of 
        "metric": "{metric_name}",
        "metric_score": <completeness_score>,
        "maximum_score": <maximum score>,
        "explanation": "<explanation>"

    List[Dict[str, str]]
        The evaluation results as a list of dicts.
    """
    evaluation_results = []
    judge = AsyncAgent("judge",prompt="", model=model, provider=provider)
    
    tasks = []
    for conv in conversation:
        judge.prompt = JUDGE_PROMPT.format(metric_name=metric.name, metric=metric, user_agent_conversation=conv)
        tasks.append(judge.generate_response(json_resp=True))
    if gather:
        scores = await asyncio.gather(*tasks)
    else:
        scores = [await task for task in tasks]

    for score in scores:
        evaluation_results.append(score)

    # sum up the scores and then divide by and add to the result
    if evaluation_results:
        total_score = sum(item['metric_score']/item['maximum_score'] for item in evaluation_results)/ len(evaluation_results)
        # Append all the explanations
        explanations = [item['explanation'] for item in evaluation_results]
        evaluation_results = {
            "metric": metric.name,
            "metric_score": total_score*100,
            "explanation": "\n".join(explanations)
        }
    return evaluation_results

async def metric_generation(theme_description: List[str], model="gpt-4o", provider="openai"):
    """
    Generate metrics based on the provided theme descriptions.
    Parameters
    ----------
    theme_description : List[str]
        List of theme descriptions for which metrics need to be generated.
    Returns
    -------
    List[Metric]
        List of generated metrics.
    """
    prompt = METRIC_GENERATION_PROMPT.format(list_of_themes_as_json_array=json.dumps(theme_description, indent=4))
    judge = AsyncAgent("judge", prompt=prompt, model=model, provider=provider)
    response = await judge.generate_response(json_resp=True)
    response = response['evaluation_principles']
    metrics = []
    for theme in response:
        if 'name' in theme and 'definition' in theme:
            name = theme['name']
            definition = theme['definition']
            scoring_criteria = theme.get('scoring_criteria', None)
            metrics.append(Metric(name, definition, scoring_criteria))
    print(metrics)
    return metrics
    

async def get_llm_output_score(gold_resp: str, agent_resp: str,judge_provider, judge_model,  metrics_list: List[Metric]):
    metrics_prompt = get_metrics_instructions(metrics_list)
    
    prompt = """You are an Evaluator Bot, which will compare a bot response to a provided expected response, along with a set of scoring metrics based on which the bot respond needs to be scored, based on this you will return a list of scores corresponding to the scoring metrics given to you
    
    New Bot Response: {agent_resp}
    
    Current Bot Response: {gold_resp}
    
    Scoring Parameters: 
    {metrics_prompt}
    
    You need to return a json response which should contain all the metrics 'name' as the key and the score based on the 'scoring criteria' as the value, For eg: if the metrics are 'user_values' which is scored out of 3 and 'guidelines' which is scored out of 5, you should return {{ "user_values": {{"score":3,"total":3}}, "guidelines": {{"score": 4,"total": 5}} }}
    return a json response only and nothing else
    {{
         (values based on scoring metrics and criteria provided)
    }}
    """
    #
            
    judge = AsyncAgent("judge",prompt=prompt.format(agent_resp=agent_resp, gold_resp=gold_resp, metrics_prompt=metrics_prompt), model=judge_model, provider=judge_provider)
    result = await judge.generate_response(json_resp=True)
    # result = await judge.generate_response(JudgeResponse)
    return result


#### Metrics

relavency = Metric('Relevancy', 'How similar the bot response is compared to the expected response', 'score 3 if the bot response is very similar, score 2 if the response is somewhat similar, score 1 if the response is not very similar, score 0 if the response is not similar at all')
correctness = Metric('Correctness', 'Are all the details mentioned in the expected response present in the bot response', 'score 3 if the bot response contains all correct information, score 0 if any of the present details is incorrect, score 3 if the expected response does not contain any details')
tone = Metric('Tone', 'Does the bot response maintain the same tone as the expected response', 'score 3 if the tone is exactly the same, score 2 if the tone is somewhat the same, score 1 if the tone is not very similar, score 0 if the tone is not similar at all')



