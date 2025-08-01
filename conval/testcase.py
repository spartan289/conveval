from typing import Optional, List
import json
from conval.parameters import JUDGE_PROMPT, USER_PROMPT, USER_PROMPT1, UserProfile
from conval.utils import convert_to_string
from conval.evaluator import Metric
from conval.utils import convert_to_json

class Chat:
    """
    A class to represent a chat conversation.
    Attributes
    ----------
    name : str
        name of the chat
    conversation_path : str
        path to the conversation file
    conversation : str
        conversation string
    messages : str
        conversation messages
    Methods
    -------
    load_conversation(file_path)
        Loads the conversation from a file.
    get_chat()
        Returns the chat messages.
    """

    def __init__(self,name: str, conversation_path: str= None, conversation: str=None):
        self.name = name
        if conversation is not None:
            self.messages = conversation

        else:
            self.messages = self.load_conversation(conversation_path)

    def load_conversation(self, file_path):
        with open(file_path, 'r') as f:
            conversation_dict = json.load(f)
        return convert_to_string(conversation_dict)
    
    def get_chat(self):
        return self.messages
    def to_list(self):
        return convert_to_json(self.messages)
    
    

class UnitTest:
    def __init__(self,conv_id: str, user_goal: str=None, user_profile: str=None, gold_conversation: Optional[Chat]=None, metrics: List[Metric] = [] , user_prompt=None, judge_prompt=None):
        self.conv_id = conv_id
        self.user_profile = user_profile
        self.user_goal = user_goal
        if gold_conversation is None:
            self.gold_conv = None
            self.chat = None
        else:
            self.gold_conv = gold_conversation
            self.chat = self.gold_conv.get_chat()
        self.metrics = metrics
        if user_prompt is not None:
            self.user_prompt = user_prompt.format(example_conversation=self.chat, user_goal=self.user_goal, user_profile=UserProfile.get_profile_details(self.user_profile)['prompt'])
        else:
            self.user_prompt = USER_PROMPT1.format(example_conversation=self.chat, user_goal=self.user_goal, user_profile=UserProfile.get_profile_details(self.user_profile)['prompt'])
        if judge_prompt is not None:
            self.judge_prompt = judge_prompt
        else:
            self.judge_prompt = JUDGE_PROMPT

class GoldenDataset:
    """
    A class to represent a dataset of golden conversations.
    Attributes
    ----------
    dataset : str
        name of the dataset
    unittests : dict
        dictionary of golden conversations
    file_path : str
        path to the dataset file
    Methods
    -------
    add_golden_conversation(conv_id, example_conversation)
        Adds a golden conversation to the dataset.
    save_dataset(file_path)
        Saves the dataset to a file.
    load_dataset(file_path)
        Loads the dataset from a file.
    get_golden_conversation(conv_id)
        Returns the golden conversation with the given ID.
    get_unittests()
        Returns the list of golden conversations.
    """
    def __init__(self, dataset_name: str = None, unitests: Optional[dict]=dict(), file_path: Optional[str]=None):
        self.dataset_name = dataset_name
        self.unittests = {unittest.conv_id: unittest for unittest in unitests}
        if file_path is not None:
            self.load_dataset(file_path)


    def add_test(self,conv_id: str,gold_conversation: Optional[Chat], metrics: List[Metric],user_profile: str|None,user_goal:str|None,user_prompt=None, judge_prompt=None):
        self.unittests[conv_id] = UnitTest(conv_id, user_goal, user_profile, gold_conversation, metrics, user_prompt, judge_prompt)


    def save_dataset(self, file_path: str):
        dataset_dict = {
            "dataset_name": self.dataset_name,
            "unit_tests": []
        }

        for ids in self.unittests:
            dataset_dict["unit_tests"].append({
                "conv_id": self.unittests[ids].conv_id,
                "user_goal": self.unittests[ids].user_goal,
                "user_profile": self.unittests[ids].user_profile,
                "gold_conversation": self.unittests[ids].chat,
                "metric": {metric.name: metric.definition for metric in self.unittests[ids].metrics},
                "user_prompt": self.unittests[ids].user_prompt,
                "judge_prompt": self.unittests[ids].judge_prompt
            })
        with open(file_path, 'w') as f:
            json.dump(dataset_dict, f, indent=4, ensure_ascii=False)

    def load_dataset(self, file_path):
        #load from json format
        import json
        with open(file_path, 'r') as f:
            dataset_dict = json.load(f)
        self.dataset_name = dataset_dict["dataset_name"]
        for golden_conv in dataset_dict["unit_tests"]:
            self.add_test(
                conv_id=golden_conv["conv_id"],
               user_profile= golden_conv["user_profile"],
              user_goal=  golden_conv["user_goal"],
               gold_conversation=  Chat(golden_conv["conv_id"], conversation=golden_conv["gold_conversation"]),
               metrics= [Metric(metric_name, metric_definition, "") for metric_name, metric_definition in golden_conv["metric"].items()],
              user_prompt = golden_conv["user_prompt"],
              judge_prompt= golden_conv["judge_prompt"]
            )
    def get_test(self, conv_id):
        if conv_id in self.unittests:
            return self.unittests[conv_id]
        else:   
            print(f"UnitTest with ID {conv_id} not found.")
    def get_unittests(self):
        return self.unittests

    def __iter__(self):
        return iter(self.unittests.values())
