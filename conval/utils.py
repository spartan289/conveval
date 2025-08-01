from openai import OpenAI
import os
from dotenv import load_dotenv
from typing import List, Optional
load_dotenv()

def swap_conversation(conversation_history):
    """
    Swap the conversation history to simulate a conversation between the user and the Bajaj agent.
    """
    a = conversation_history
    for message in a:
        if message['role'] == 'user':
            message['role'] = 'assistant'
        else:
            message['role'] = 'user'
    return a

def convert_to_string(conversation_list):
    conversation = ""
    for message in conversation_list:
            if message['content']=='NA':
                message['content']=='Call End'
            conversation += f"{message['role']}: {message['content'].replace('\n',' ')}\n"
    return conversation

def convert_to_json(conversation: str):
    conversation_list = []
    for line in conversation.split("\n"):
        if line.strip():
            role, content = line.split(": ", 1)
            conversation_list.append({"role": role, "content": content})
    return conversation_list

def generate_embedding(text):
    """
    Generate an embedding for the given text using OpenAI's API.
    Args:
        text: Text to generate an embedding for.
    Returns:
        Embedding for the text.
    """
    embedding = []
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.embeddings.create(input=[text], model="text-embedding-ada-002")
    for j in range(len(response.data)):
        embedding.append(response.data[j].embedding)

    return embedding


def get_metrics_instructions(metrics_list: List):
    metrics_prompt = ""
    
    for metric in metrics_list:
        name = metric.name
        definition = metric.definition
        scoring_criteria = metric.scoring_criteria
        
        metric_compiled = f"""{name}: {definition}
        Scoring Criteria: {scoring_criteria}\n
        """
        
        metrics_prompt += metric_compiled
        
    return metrics_prompt
    