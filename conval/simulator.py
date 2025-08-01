from typing import Optional, List, Callable
import copy
from conval.agent import Agent
from conval.utils import swap_conversation, generate_embedding
import asyncio
from conval.evaluator import get_llm_output_score, relavency, tone, correctness

class SimulateChat:
    def __init__(self, assistant_agent: Optional[Agent], user_agent: Optional[Agent]=None,endswith=None, conversation_history: Optional[List] = [], max_turns: int =25):
        self.user_agent = user_agent
        self.assistant_agent = assistant_agent
        self.conversation_history = conversation_history
        self.max_turns = max_turns
        self.endswith = endswith
        


    async def simulate(self):
        
        # User starts the conversation
        while len(self.conversation_history) < self.max_turns: 
            # User responds
            a = copy.deepcopy(self.conversation_history)
            reverse_conv = swap_conversation(a)
            user_message = await self.user_agent.generate_response(reverse_conv)
            del reverse_conv
            self.conversation_history.append({"role": "user", "content": user_message})

            # Agent responds
            agent_message = await self.assistant_agent.generate_response(self.conversation_history)
            self.conversation_history.append({"role": "assistant", "content": agent_message})

            if self.endswith is not None and self.endswith in agent_message:
                break

        return self.conversation_history



    async def simulate_stepwisewithgather(self, conversation,judge_provider, judge_model, custom_metrics: List=[], callback: Callable[[List], None] =None):
        """
        Simulate the conversation step by step, allowing for interaction at each step. and return score.
        """
        print(conversation)
        self.conversation_history = [{"role": "assistant", "content": conversation[0]['content']}]
        conversation_history_batch = []
        for index, turn in enumerate(conversation):
            
            if turn['role']=='user':
                if self.endswith is not None and self.endswith in turn['content']:
                    break
                self.conversation_history.append({"role": "user", "content": turn['content']})
                gold_response = conversation[index+1]['content']
                self.conversation_history.append({"role": "assistant", "content": gold_response})            
                a = copy.deepcopy(self.conversation_history)
                conversation_history_batch.append(a)

        group =  await asyncio.gather(*[self.evaluate_response(history, judge_provider, judge_model,custom_metrics) for history in conversation_history_batch])
        return group
    
    async def simulate_stepwise(self, conversation,judge_provider, judge_model, custom_metrics: List=[], callback: Callable[[List], None] =None):
        """
        Simulate the conversation step by step, allowing for interaction at each step.
        """
        self.conversation_history = [{"role": "assistant", "content": conversation[0]['content']}]
        conversation_history_batch = []
        for index, turn in enumerate(conversation):
            if turn['role']=='user':
                if self.endswith is not None and self.endswith in turn['content']:
                    break
                self.conversation_history.append({"role": "user", "content": turn['content']})
                gold_response = conversation[index+1]['content']
                self.conversation_history.append({"role": "assistant", "content": gold_response})
                a = copy.deepcopy(self.conversation_history)
                conversation_history_batch.append(a)

        group =  [await self.evaluate_response(history, judge_provider, judge_model,custom_metrics) for history in conversation_history_batch]
        return group

    async def evaluate_response(self, history,judge_provider, judge_model, custom_metrics):
        """
        Evaluate the response using the provided metrics.
        """

        gold_history = history[:-1]
        gold_response = history[-1]
        agent_message = await self.assistant_agent.generate_response(gold_history)
        score = await get_llm_output_score(gold_response, agent_message,judge_provider, judge_model, custom_metrics)
        return score

    async def simulatewithoutusersim(self, conversation):
        """
        Simulate the conversation without user simulation.
        """
        self.conversation_history = [{"role": "assistant", "content": conversation[0]['content']}]
        for index, turn in enumerate(conversation):
            if turn['role']=='user':
                if self.endswith is not None and self.endswith in turn['content']:
                    break
                self.conversation_history.append({"role": "user", "content": turn['content']})
                agent_message = await self.assistant_agent.generate_response(self.conversation_history)

                self.conversation_history.append({"role": "assistant", "content": agent_message})

        return self.conversation_history