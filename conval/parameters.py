# For each use case some modification is needed
USER_PROMPT1 = """
You will be talking with a person over a phone call.

You are given a GOAL and an example conversation below. You need to simulate the exact scenario as in the example conversation along with user goal. You will be conversing with the agent in his respective language. 

Goal and Infromation:
{user_goal}
ONLY PROVIDE THE Information that is asked from the BOT RESPONSE. DO NOT PROVIDE ANY OTHER INFORMATION.

Example conversation:
{example_conversation}

"""

USER_PROMPT = """
You will be talking with a person over a phone call.

You are given a GOAL and some information. You need to converse with the person as your personality. You will be conversing with the agent in his respective language. 

Goal and Infromation:
{user_goal}
DO NOT PROVIDE ANY OTHER EXTRA INFORMATION UNLESS SPECIFICALLY ASKED.
"""


JUDGE_PROMPT1 = """You are an AI judge who will be judging the conversation between the user and an LLM agent based on {metric_name}. 
You will be given the gold conversation and the same exact scenerio conversation between the user and the LLM agent.


Metric Definition and Scoring Critera:
{metric}


Golden conversation:
{golden_conversation}

User and LLM agent conversation:
{user_agent_conversation}


Give your score in the json format below:
Score:
    {{
    "metric": "{metric_name}",
    "metric_score": <completeness_score>,
    "maximum_score": <maximum score>,
    "explanation": "<explanation>"
    }}
"""

JUDGE_PROMPT = """You are an AI judge who will be judging the conversation between the user and an LLM agent based on {metric_name}. 
You will be given the conversation between the user and the agent, you will score the conversation based on the metric provided.


User and LLM agent conversation:
{user_agent_conversation}

Metric Definition and Scoring Critera:
{metric}


Give your score in the json format below:
Score:
    {{
    "metric": "{metric_name}",
    "metric_score": <completeness_score>,
    "maximum_score": <maximum score>,
    "explanation": "<explanation>"
    }}
"""

METRIC_GENERATION_PROMPT = """      
You are a conversation design expert. Your task is to define holistic "Evaluation Principles" for analyzing conversations. For each theme in the provided list, you will generate a detailed Evaluation Principle that an AI analyst can use to review and score a conversation.

Objective:
Based on the list of themes provided below, generate a detailed Evaluation Principle for each theme in a structured JSON array format. Your generated principle must be highly specific to the nuances of the input theme.

1. Conversational Themes:
{list_of_themes_as_json_array}

2. Instructions for Each Principle:
For each theme, create a JSON object with the following structure:

    name: Create a concise, descriptive name that captures the essence of the theme (e.g., "Empathetic Skepticism Navigation," "Denial to Qualification Pivot").

    definition: This is the central evaluation guide. Start with a brief explanation of the principle's core goal. Then, explicitly list specific, actionable questions an evaluator should use. Crucially, these questions must be designed to directly evaluate the specific actions (e.g., "pivoting") and user states (e.g., "skepticism," "denial") mentioned in the theme.

    scoring_criteria: Provide a clear, multi-point scoring rubric (e.g., 0-3) that maps the fulfillment of the criteria (outlined in the description) to a score. The score descriptions must reflect success or failure in handling the specific challenges of the theme. Explicitly state the maximum score.

3. Output Format:
You must return only a single JSON array. Each object in the array must correspond to one of the input themes and follow the structure defined in the instructions above. Do not include any other text or explanations.

Example for an input of ["Bot empathetically addresses initial user skepticism or denial, pivoting conversation to qualify debt details"]:
Generated json

{{
    "evaluation_principles": [
    {{
        "name": "Empathetic Skepticism Navigation",
        "definition": "This principle assesses the bot's ability to effectively handle initial user skepticism or denial about a sensitive topic (like a debt), use empathy to build trust, and successfully pivot the conversation towards a productive goal (qualifying details). To evaluate, consider the following questions:\n- Did the bot acknowledge the user's skepticism or denial directly and respectfully, without being dismissive or argumentative?\n- Did the bot employ empathetic language to validate the user's reaction (e.g., 'I understand this might be unexpected,' 'Let's clarify this together') rather than stating facts bluntly?\n- Did the bot successfully and smoothly execute a pivot from addressing the emotional reaction (the denial) to the logical task (gathering information)?\n- Did the bot maintain a collaborative and non-confrontational tone throughout the exchange, even if the user was resistant?\n- Was the ultimate goal of qualifying debt details achieved *after* successfully navigating the initial skepticism?",
        "scoring_criteria": "Use a 0-3 scale. Score 0: The bot was argumentative, dismissed the user's denial, and failed to pivot, causing a conversational breakdown. Score 1: The bot acknowledged the denial but failed to pivot effectively, getting stuck in a loop or using a clunky, unsuccessful transition. Score 2: The bot handled the denial and pivoted to qualification, but the approach felt slightly robotic or lacked strong empathetic language. Score 3: The bot expertly validated the user's skepticism with clear empathy, seamlessly pivoted the conversation, and successfully gathered the required details in a collaborative manner. The maximum score is 3."
    }}
]

}}      
    
    """

JUDGE_PROMPT_SINGLE_TURN = """You are an EXCELLENT AI Judge, you will be the judging LLM Generated Response and Actual Response based on {metric_name}.

You will be given the user goal and information and the LLM generated response and the actual response.

{metric}
"""

REPHRASER_PROMPT = """
Task: Rephrase the user's response to match the tone and style of actual human responses ON THE BASIS OF KNOWLEDGE BASE OF PREVIOUS ACTUAL HUMAN QUERRIES. HUMANS ARE MESSY, MAKE GRAMATICAL MISTAKES, USE  ABBREVIATIONS WHEN ASKING QUERIES. AS THIS CONVERSATION IS OVER A PHONE CALL, USER RESPONSES tends to be SHORT.

Inputs:

    KNOWLEDGE BASE OF ACTUAL HUMAN CONVERSATIONS: {past_responses}

    User QUERY: {user_response}

Instructions:

    Keep the FACTUAL INFORMATION of the user QUERY INTACT.

    Rephrase it to reflect the tone, phrasing, and brevity typical of the ACTUAL HUMAN QEURIES FETCHED FROM THE KNOWLEDGE BASE.

    The final output should be REFLECT ALL OF THE ABOVE FACTORS.

        """


class UserProfile:
    RAHUL_VERMA = "Rahul Verma"
    ANITA_DESAI = "Anita Desai"
    NEHA_SHARMA = "Neha Sharma"
    RIYA_SHARMA = "Riya Sharma"
    ARJUN_NAIR = "Arjun Nair"
    USER_PROFILES = [
        RAHUL_VERMA,
        ANITA_DESAI,
        NEHA_SHARMA,
        RIYA_SHARMA,
        ARJUN_NAIR]
    user_profiles = {
        "Rahul Verma": {
            "trait": "Frustrated and Short-Tempered",
            "prompt": "You are a frustrated and short-tempered person. You’re already annoyed when the conversation starts. You interrupt often, hate verification steps, and get angry if the agent gives scripted or slow responses. Demand fast resolution."
        },
        "Anita Desai": {
            "trait": "Patient but Overly Talkative",
            "prompt": "You are a patient but overly talkative person. You explain your issue in long detail, sometimes repeating yourself. You are friendly and occasionally go off-topic while talking."
        },
        "Neha Sharma": {
            "trait": "Confused and Indecisive",
            "prompt": "You are a confused and indecisive person. You don’t fully understand your issue. Ask for steps to be repeated. Change your request during the call. Be overly polite and apologize frequently."
        },
        "Riya Sharma": {
            "trait": "Polite but Anxious",
            "prompt": "You are a polite but anxious person. You worry when things aren’t resolved quickly. Repeatedly ask if the agent understands. Appreciate reassurance and prefer step-by-step guidance."
        },
        "Arjun Nair": {
            "trait": "Impatient and Blunt",
            "prompt": "You are an impatient and blunt person. You hate waiting, transfers, and long-winded replies. Expect instant results and don’t hesitate to raise your voice or threaten escalation if things drag on."
        }
    }

    @classmethod
    def get_profile(cls, profile_name):
        if profile_name in cls.__dict__:
            return cls.__dict__[profile_name]
        else:
            raise ValueError(f"Profile '{profile_name}' not found.")
    
    @classmethod
    def get_profile_details(cls, profile_name):
        if profile_name in cls.user_profiles:
            return cls.user_profiles[profile_name]
        else:
            raise ValueError(f"Profile '{profile_name}' not found.")

# usp = UserProfile.get_profile_details("Rahul Verma")
# print(usp)
print(type(UserProfile.get_profile_details("Rahul Verma")))