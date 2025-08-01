from conval.utils import convert_to_string
from conval.agent import Agent, AsyncMultiAgent, AsyncAgent
import json
from typing import List
import asyncio
agent = AsyncMultiAgent(
        name="insights_extractor",
        description="Extracts behavioral insights from conversations",
        model="gemini-2.0-flash",
        provider="google",
        prompt="",
    )

async def get_insights_and_summary(conversation: List):
    conversation_str = convert_to_string(conversation)
    summarisation_prompt = """You are an expert conversation analyst. Analyze the following conversation with respect to bot prompt to extract key insights.

    Instructions:
    - {extra_instruction}

ORIGINAL BOT PROMPT/INSTRUCTIONS(delimited by <prompt></prompt>):
<prompt>
{bot_prompt}
</prompt>

CONVERSATION TRANSCRIPT(delimited by <conversation></conversation>):
<conversation>
{conversation}
</conversation>

Stricly follow the below given analysis requirements
ANALYSIS REQUIREMENTS:
- Ignore scripted/templated bot responses that directly follow the prompt instructions
- Focus on pivotal moments that controlled the conversation direction
- Identify key user behaviors
- Identify bot handling

- Maximum 50-70 words

Provide a concise insight patterns and how the bot handled (or failed to handle, what did bot do) critical moments. Focus on what actually mattered for the conversation outcome.

return json response and nothing else
{{
    "insights": (generated insights according to requirements)
}}
"""
    summarisation_prompt = """You are analyzing sales call transcripts related to credit card debt relief. Each conversation follows a general script but varies based on the customer’s responses.

Summarize each conversation abstractly, focusing on:

    The customer's debt situation

    Their interest level or openness

    Tone or sentiment

    Behavioral signals (e.g., saved number, shared struggles, objections)

    Any implied barriers to conversion

Avoid repeating the script. Be concise but insightful. Write in a structured, high-level style suitable for clustering.

Format your summary like this:

    Customer [summary of financial situation], [summary of interest level or behavior]. [Mention tone or emotional cues]. [Any final action or signal]. [Brief insight into what’s holding them back].

Example:

    Customer self-manages <10K in debt, not actively seeking help. Expressed past financial difficulty, ended call politely, saved number. No objections to program—barrier likely timing or self-reliance.

{transcript}

"""

    prompt = summarisation_prompt.format(transcript=conversation_str)
    agent.prompt = prompt
    result = await agent.generate_response()
    return result

