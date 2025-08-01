import random
import json
def build_insight_dictionary(data, cluster_key='cluster_number', transcript_key='transcript', insight_key='insights'):
    """
    Builds structured dictionaries from insight data grouped by cluster.

    Parameters:
    - data (list): List of dicts with call/insight data.
    - cluster_key (str): Key in each dict indicating the cluster number.
    - transcript_key (str): Key in each dict indicating the transcript text.
    - insight_key (str): Key in each dict indicating the insight object.

    Returns:
    - insights_clusters (dict): Dict mapping cluster_number -> list of insights.
    - transcripts_cluster (dict): Dict mapping cluster_number -> list of original data entries.
    """
    clusters = set()
    insights_clusters = {}
    transcripts_cluster = {}
    transcripts = []
    insights = []

    max_cluster = 0
    for item in data:
        cluster = item[cluster_key]
        cluster = int(cluster)
        clusters.add(cluster)
        
        max_cluster = max(max_cluster, cluster)

    cluster_sizes = [0] * (max_cluster + 1)

    for cluster in clusters:
        insights_clusters[cluster] = []
        transcripts_cluster[cluster] = []

    for item in data:
        cluster = item[cluster_key]
        insight = item[insight_key]
        transcript = item[transcript_key]

        cluster_sizes[cluster] += 1
        insights_clusters[cluster].append(insight)
        transcripts_cluster[cluster].append(item)

        transcripts.append(transcript)
        insights.append(insight)

    return (
        insights_clusters,
        transcripts_cluster,
    )


def create_insight_payload(insights_clusters, max_samples=15):
    """
    Generate a formatted payload string from clustered insights.

    Parameters:
    - insights_clusters (dict): A dictionary where keys are cluster numbers and values are lists of insights.
    - max_samples (int): Maximum number of insights to include per cluster (default is 15).

    Returns:
    - str: Formatted payload string.
    """
    payload = ""
    for cluster in insights_clusters:
        payload += f"CLUSTER NUMBER: {cluster}\n\n"
        k_value = min(max_samples, len(insights_clusters[cluster]))
        selected_values = random.choices(insights_clusters[cluster], k=k_value)

        for i, insight in enumerate(selected_values):
            payload += f"{i+1}. {insight}\n"
        payload += "\n"
    return payload


from openai import OpenAI
def get_insights_from_cluster_payloads(cluster_payload: str):
    prompt = """You are a data analysis agent, and you will be given data, which are insights of calls between a user and a bot.
    However these insights have been first categorized via a clustering algorithm, and divided based on their semantic meanings
    You will be given up-to 15 insights for each of these clusters, and in response you need to provide a 'title' for each of these clusters
    
    The 'title' you return must be around 15 words and should represent the best generalization of that cluster, which should cover any important pattern, detail or similarity.
    
    However also remember, that the 'title' you return for one cluster should not overlap with the other clusters, each title should be as non-repetetive and unique as possible, representing what is uniquely different about that cluster.
    
    Keep the words and the title itself, easy to understand and read. 
    
    Given Below is the data of clusters (delimited by <data></data>)
    <data>
    
    {cluster_payload}
    </data>
    
    return a json response, with a key for each cluster number like 1, 2, 3 etc. and the title as value.as_integer_ratio
    
    Example Response: 
    {{
        "0": "(title for cluster 0)",
        "1": "(title for cluster 0)",
        "2": "(title for cluster 0)",
        "3": "(title for cluster 0)",
        
    }}
    """
    
    prompt = prompt.format(cluster_payload=cluster_payload)
    client = OpenAI(
        api_key="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",
    )
    completion = client.chat.completions.create(
            model="o4-mini",
            messages=[
                {
                    "role": "system", "content": prompt
                },
            ],
            temperature=1,
            # top_p=0.9
    )
    
    result = completion.choices[0].message.content
    
    if(result[:8] == "```json\n"):
            result = result[8:]
            
    if result.endswith("```"):
        result = result[:-3]
            
    result = json.loads(result, strict=False)
    # convert keys to integers
    result = {int(k): v for k, v in result.items()}
    return result

def get_label(data, cluster_key='cluster_number', transcript_key='transcript', insight_key='insights'):
    """
    Get labels for each cluster based on insights.

    Parameters:
    - data (list): List of dicts with call/insight data.
    - cluster_key (str): Key in each dict indicating the cluster number.
    - transcript_key (str): Key in each dict indicating the transcript text.
    - insight_key (str): Key in each dict indicating the insight object.

    Returns:
    - dict: A dictionary mapping cluster numbers to their labels.
    """
    insights_clusters, _ = build_insight_dictionary(data, cluster_key, transcript_key, insight_key)
    payload = create_insight_payload(insights_clusters)
    
    return get_insights_from_cluster_payloads(payload)