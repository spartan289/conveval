from openai import OpenAI, AsyncOpenAI
import json
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from together import Together, AsyncTogether
from typing import Optional, List, Dict, Any
import asyncio
import time
import random
load_dotenv()

class Agent:
    '''
    A class to represent an AI agent that can generate responses based on a prompt and history.
    It supports different providers such as OpenAI, Google, and Together.
    Attributes:
        name (str): The name of the agent.
        description (str): A brief description of the agent.
        model (str): The model to be used for generating responses.
        prompt (str): The initial prompt for the agent.
        provider (str): The provider of the AI service ('openai', 'google', or 'together').
        client: An instance of the client for the specified provider.
    Methods:
        generate_response(history: list=[], json_resp=False, temperature=0.5, top_p=0.9) -> str:
        Generates a response based on the provided history and parameters.
    Usage:
        agent = Agent(name="ExampleAgent", prompt="You are a helpful assistant.", model="gpt-4o", provider='openai')
        response = agent.generate_response(history=[{"role": "user", "content": "Hello!"}])
        print(response)
    '''
    def __init__(self, name: str, prompt: str,model: str = "gpt-4o", description: str = None, provider='openai'):
        self.name = name
        self.description = description
        self.model = model
        self.prompt = prompt
        self.provider = provider
        if provider == 'openai':
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif provider == 'google':
            self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            
        elif provider == 'together':
            self.client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
        elif provider == 'ollama':
            self.client = OpenAI(base_url=os.getenv("OLLAMA_API_URL"), api_key='OLLAMA')
           
    def generate_response(self,  history: list=[], json_resp=False, temperature=0.5, top_p=0.9) -> str:
        if self.provider == 'openai':
            return self._generate_response_openai(history, json_resp, temperature, top_p)
        elif self.provider == 'google':
            return self._generate_response_google(history, json_resp, temperature, top_p)
        elif self.provider == 'together':
            return self._generate_response_together(history, json_resp, temperature, top_p)
        elif self.provider == 'ollama':
            return self._generate_response_ollama(history, json_resp)
    
    def _generate_response_openai(self, history: list=[], json_resp=False, temperature=0.5, top_p=0.9) -> str:
        if json_resp:
            content = [{"role": "system", "content": self.prompt}] + history
            response = self.client.chat.completions.create(
                model=self.model,
                messages=content,
                response_format={"type": "json_object"},
                temperature=temperature,
                top_p=top_p
            )
            return json.loads(response.choices[0].message.content)
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": self.prompt}] + history,
                temperature=temperature,
                top_p=top_p
            )
            return response.choices[0].message.content
    def convert_to_genai_format(self, message: dict) -> dict:
        return {
            "role": message["role"],
            "parts": [{"text": message["content"]}]
        }
    def _generate_response_google(self, history: list=[], json_resp=False, temperature=0.5, top_p=0.9) -> str:
        converted_messages = [self.convert_to_genai_format(msg) for msg in history]
        if json_resp:
            if converted_messages==[]:
                response =  self.client.models.generate_content(
                contents= self.prompt,
                model=self.model,
                config=types.GenerateContentConfig(
                    response_mime_type='application/json',
                ))
            else:
                self.client.models.embed_content
                response =  self.client.models.generate_content(
                    contents=converted_messages,
                    model=self.model,
                    config=types.GenerateContentConfig(
                        system_instruction=self.prompt,
                        response_mime_type='application/json',
                    ))
            return json.loads(response.text)
        else:
            if converted_messages==[]:
                response =  self.client.models.generate_content(
                    contents= self.prompt,
                    model=self.model, 
                    config=types.GenerateContentConfig(
                    )
                    
                )
            else:

                response =  self.client.models.generate_content(
                    contents=converted_messages,
                    model=self.model, 
                    config=types.GenerateContentConfig(
                            system_instruction=self.prompt, 
                        )
                )
            return response.text

    def _generate_response_together(self, history: list=[], json_resp=False, temperature=0.5, top_p=0.9) -> str:
        if json_resp:
            content = [{"role": "system", "content": self.prompt}] + history
            response = self.client.chat.completions.create(
                model=self.model,
                messages=content,
                response_format={"type": "json_object"},
                temperature=temperature,
                top_p=top_p
            )
            return json.loads(response.choices[0].message.content)
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": self.prompt}] + history,
                temperature=temperature,
                top_p=top_p
            )
            return response.choices[0].message.content

    def _generate_response_ollama(self, history: list=[], json_resp=False):
        """
        Placeholder for Ollama response generation.
        This method should be implemented when Ollama support is added.
        """
        if json_resp:
            content = [{"role": "system", "content": self.prompt}] + history
            response = self.client.chat.completions.create(
                model=self.model,
                messages=content,
                response_format={"type": "json_object"},
            )
            return json.loads(response.choices[0].message.content)
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": self.prompt}] + history,
            )
            return response.choices[0].message.content


    
class AsyncAgent:
    '''
    A class to represent an AI agent that can asynchronous generate responses based on a prompt and history.
    It supports different providers such as OpenAI, Google, and Together.
    Attributes:
        name (str): The name of the agent.
        description (str): A brief description of the agent.
        model (str): The model to be used for generating responses.
        prompt (str): The initial prompt for the agent.
        provider (str): The provider of the AI service ('openai', 'google', or 'together').
        client: An instance of the client for the specified provider.
    Methods:
        generate_response(history: list=[], json_resp=False, temperature=0.5, top_p=0.9) -> str:
        Generates a response based on the provided history and parameters.
    Usage:
        agent = Agent(name="ExampleAgent", prompt="You are a helpful assistant.", model="gpt-4o", provider='openai')
        response = agent.generate_response(history=[{"role": "user", "content": "Hello!"}])
        print(response)
    '''
    def __init__(self, name: str, prompt: str="",model: str = "gpt-4o", description: str = None, provider='openai'):
        self.name = name
        self.description = description
        self.model = model
        self.prompt = prompt
        self.provider = provider
        if provider == 'openai':
            self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif provider == 'google':
            self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            
        elif provider == 'together':
            self.client = AsyncTogether(api_key=os.getenv("TOGETHER_API_KEY"))
        elif provider == 'ollama':
            print(os.getenv("OLLAMA_API_URL"))
            self.client = AsyncOpenAI(api_key='OLLAMA', base_url=os.getenv("OLLAMA_API_URL"))
        elif provider == 'vllm':
            self.client = AsyncOpenAI(api_key='VLLM', base_url=os.getenv("VLLM_ENDPOINT"))
        elif provider=="huggingface":
            from transformers import pipeline
            self.client = pipeline(
                "text-generation",
                model=self.model,
            )
            
            
        
           
    async def generate_response(self,  history: list=[], json_resp=False, temperature=0.5, top_p=0.9, json_schema=None) -> str:
        if self.provider == 'openai':
            return await self._generate_response_openai(history, json_resp, temperature, top_p)
        elif self.provider == 'google':
            return await self._generate_response_google(history, json_resp, temperature, top_p)
        elif self.provider == 'together':
            return await self._generate_response_together(history, json_resp, temperature, top_p)
        elif self.provider == 'ollama':
            return await self._generate_response_ollama(history, json_resp, json_schema=None)
        elif self.provider == 'vllm':
            return await self._generate_response_openai(history, json_resp, temperature, top_p)
        elif self.provider == 'huggingface':
            messages = [{"role": "system", "content": self.prompt}] + history
            response = self.client(messages)
            return response[0]['generated_text'][-1]['content']
    
    async def _generate_response_openai(self, history: list=[], json_resp=False, temperature=0.5, top_p=0.9) -> str:
        if json_resp:
            content = [{"role": "system", "content": self.prompt}] + history
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=content,
                response_format={"type": "json_object"},
                temperature=temperature,
                top_p=top_p
            )
            return json.loads(response.choices[0].message.content)
        else:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": self.prompt}] + history,
                temperature=temperature,
                top_p=top_p
            )
            return response.choices[0].message.content
    def convert_to_genai_format(self, message: dict) -> dict:
        if message["role"]=="assistant":
            return {
                "role": "model",
                "parts": [{"text": message["content"]}]
            }
        return {
            "role": message["role"],
            "parts": [{"text": message["content"]}]
        }
    async def _generate_response_google(self, history: list=[], json_resp=False, temperature=0.5, top_p=0.9) -> str:
        converted_messages = [self.convert_to_genai_format(msg) for msg in history]
        if json_resp:
            if converted_messages==[]:
                response = await self.client.aio.models.generate_content(
                contents= self.prompt,
                model=self.model,
                config=types.GenerateContentConfig(
                    response_mime_type='application/json',

                ))
            else:
                response = await self.client.aio.models.generate_content(
                    contents=converted_messages,
                    model=self.model,
                    config=types.GenerateContentConfig(
                        system_instruction=self.prompt,
                        response_mime_type='application/json',

                    ))
            return json.loads(response.text)
        else:
            if converted_messages==[]:
                response = await self.client.aio.models.generate_content(
                    contents= self.prompt,
                    model=self.model, 
                    config=types.GenerateContentConfig(

                    )
                )
            else:

                response = await self.client.aio.models.generate_content(
                    contents=converted_messages,
                    model=self.model, 
                    config=types.GenerateContentConfig(
                        system_instruction=self.prompt, 
                        )
                )
            return response.text

    async def _generate_response_together(self, history: list=[], json_resp=False, temperature=0.5, top_p=0.9) -> str:
        if json_resp:
            content = [{"role": "system", "content": self.prompt}] + history
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=content,
                response_format={"type": "json_object"},
                temperature=temperature,
                top_p=top_p
            )
            return json.loads(response.choices[0].message.content)
        else:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": self.prompt}] + history,
                temperature=temperature,
                top_p=top_p
            )
            return response.choices[0].message.content
    async def _generate_response_ollama(self, history: list=[], json_resp=False, json_schema=None):
        """
        Placeholder for Ollama response generation.
        This method should be implemented when Ollama support is added.
        """
        if json_resp:
            content = [{"role": "system", "content": self.prompt}] + history
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=content,
                response_format={"type": "json_object"},
                max_completion_tokens=128
            )
            return json.loads(response.choices[0].message.content)
        elif json_schema:
            content = [{"role": "system", "content": self.prompt}] + history
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=content,
                response_format=json_schema,
                max_completion_tokens=128
            )
            return json.loads(response.choices[0].message.content)
        else:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": self.prompt}] + history,
                max_completion_tokens=128,
            )
            return response.choices[0].message.content

class AsyncMultiAgent:
    '''
    A class to represent an AI agent that can asynchronously generate responses based on a prompt and history.
    It supports different providers such as OpenAI, Google, and Together with multiple API keys for rate limit management.
    
    Attributes:
        name (str): The name of the agent.
        description (str): A brief description of the agent.
        model (str): The model to be used for generating responses.
        prompt (str): The initial prompt for the agent.
        provider (str): The provider of the AI service ('openai', 'google', or 'together').
        clients (list): List of client instances for the specified provider using different API keys.
        key_usage (dict): Track usage statistics for each API key.
        
    Methods:
        generate_response(history: list=[], json_resp=False, temperature=0.5, top_p=0.9) -> str:
        Generates a response based on the provided history and parameters.
        
    Usage:
        # Set multiple API keys as environment variables or pass them directly
        agent = AsyncAgent(
            name="ExampleAgent", 
            prompt="You are a helpful assistant.", 
            model="gpt-4o", 
            provider='openai',
            api_keys=['key1', 'key2', 'key3']  # or None to use environment variables
        )
        response = await agent.generate_response(history=[{"role": "user", "content": "Hello!"}])
        print(response)
    '''
    
    def __init__(self, name: str, prompt: str, model: str = "gpt-4o", description: str = None, 
        provider='openai', api_keys: Optional[List[str]] = None, 
        key_rotation_strategy: str = 'round_robin'):
        self.name = name
        self.description = description
        self.model = model
        self.prompt = prompt
        self.provider = provider
        self.key_rotation_strategy = key_rotation_strategy  # 'round_robin', 'random', 'least_used'
        self.current_key_index = 0
        self.key_usage = {}  # Track usage for each key
        self.failed_keys = set()  # Track temporarily failed keys
        
        # Initialize clients with multiple API keys
        self._initialize_clients(api_keys)
    
    def _initialize_clients(self, api_keys: Optional[List[str]]):
        """Initialize multiple clients with different API keys."""
        self.clients = []
        self.api_keys = []
        
        if api_keys:
            self.api_keys = api_keys
        else:
            # Load multiple keys from environment variables
            self.api_keys = self._load_keys_from_env()
        
        if not self.api_keys:
            raise ValueError(f"No API keys found for provider {self.provider}")
        
        # Initialize clients for each key
        for i, key in enumerate(self.api_keys):
            try:
                if self.provider == 'openai':
                    from openai import AsyncOpenAI
                    client = AsyncOpenAI(api_key=key)
                elif self.provider == 'google':
                    client = genai.Client(api_key=key,vertexai=False)
                elif self.provider == 'together':
                    from together import AsyncTogether
                    client = AsyncTogether(api_key=key)
                else:
                    raise ValueError(f"Unsupported provider: {self.provider}")
                
                self.clients.append(client)
                self.key_usage[i] = {'requests': 0, 'errors': 0, 'last_used': 0}
                
            except Exception as e:
                print(f"Failed to initialize client for key {i+1}: {e}")
    
    def _load_keys_from_env(self) -> List[str]:
        """Load multiple API keys from environment variables."""
        keys = []
        
        if self.provider == 'openai':
            # Look for OPENAI_API_KEY, OPENAI_API_KEY_1, OPENAI_API_KEY_2, etc.
            base_key = os.getenv("OPENAI_API_KEY")
            if base_key:
                keys.append(base_key)
            
            i = 1
            while True:
                key = os.getenv(f"OPENAI_API_KEY_{i}")
                if key:
                    keys.append(key)
                    i += 1
                else:
                    break
                    
        elif self.provider == 'google':
            base_key = os.getenv("GEMINI_API_KEY")
            if base_key:
                keys.append(base_key)
            
            i = 1
            while True:
                key = os.getenv(f"GEMINI_API_KEY_{i}")
                if key:
                    keys.append(key)
                    i += 1
                else:
                    break
                    
        elif self.provider == 'together':
            base_key = os.getenv("TOGETHER_API_KEY")
            if base_key:
                keys.append(base_key)
            
            i = 1
            while True:
                key = os.getenv(f"TOGETHER_API_KEY_{i}")
                if key:
                    keys.append(key)
                    i += 1
                else:
                    break
        
        return keys
    
    def _select_client(self) -> tuple:
        """Select the best client based on the rotation strategy."""
        available_indices = [i for i in range(len(self.clients)) if i not in self.failed_keys]
        
        if not available_indices:
            # Reset failed keys if all are failed (they might have recovered)
            self.failed_keys.clear()
            available_indices = list(range(len(self.clients)))
        
        if self.key_rotation_strategy == 'round_robin':
            # Round robin through available keys
            index = available_indices[self.current_key_index % len(available_indices)]
            self.current_key_index = (self.current_key_index + 1) % len(available_indices)
            
        elif self.key_rotation_strategy == 'random':
            # Random selection
            index = random.choice(available_indices)
            
        elif self.key_rotation_strategy == 'least_used':
            # Select the key with least usage
            index = min(available_indices, key=lambda i: self.key_usage[i]['requests'])
            
        else:
            # Default to round robin
            index = available_indices[self.current_key_index % len(available_indices)]
            self.current_key_index = (self.current_key_index + 1) % len(available_indices)
        
        return self.clients[index], index
    
    def _update_key_usage(self, key_index: int, success: bool = True):
        """Update usage statistics for a key."""
        self.key_usage[key_index]['requests'] += 1
        self.key_usage[key_index]['last_used'] = time.time()
        
        if not success:
            self.key_usage[key_index]['errors'] += 1
    
    def _handle_rate_limit_error(self, key_index: int, error: Exception):
        """Handle rate limit errors by temporarily marking key as failed."""
        print(f"Rate limit hit for key {key_index + 1}: {error}")
        self.failed_keys.add(key_index)
        
        # Remove from failed keys after some time (you can adjust this)
        asyncio.create_task(self._recover_key(key_index))
    
    async def _recover_key(self, key_index: int, delay: int = 60):
        """Recover a failed key after a delay."""
        await asyncio.sleep(delay)
        if key_index in self.failed_keys:
            self.failed_keys.remove(key_index)
            print(f"Recovered key {key_index + 1}")
    
    async def generate_response(self, history: list = [], json_resp=False, 
                              temperature=0.5, top_p=0.9, max_retries=3) -> str:
        """Generate response with automatic key rotation and retry logic."""
        last_error = None

        for attempt in range(max_retries):
            try:
                client, key_index = self._select_client()
                
                if self.provider == 'openai':
                    result = await self._generate_response_openai(
                        client, history, json_resp, temperature, top_p
                    )
                elif self.provider == 'google':
                    result = await self._generate_response_google(
                        client, history, json_resp, temperature, top_p
                    )
                elif self.provider == 'together':
                    result = await self._generate_response_together(
                        client, history, json_resp, temperature, top_p
                    )
                
                self._update_key_usage(key_index, success=True)
                return result
                
            except Exception as e:
                last_error = e
                self._update_key_usage(key_index, success=False)
                
                # Check if it's a rate limit error
                error_str = str(e).lower()
                if any(term in error_str for term in ['rate limit', 'quota', 'too many requests', '429']):
                    self._handle_rate_limit_error(key_index, e)
                
                print(f"Attempt {attempt + 1} failed with key {key_index + 1}: {e}")
                
                if attempt < max_retries - 1:
                    # Wait before retry
                    await asyncio.sleep(1)
        
        raise Exception(f"All retry attempts failed. Last error: {last_error}")
    
    async def _generate_response_openai(self, client, history: list = [], json_resp=False, 
                                      temperature=0.5, top_p=0.9) -> str:
        if json_resp:
            content = [{"role": "system", "content": self.prompt}] + history
            response = await client.chat.completions.create(
                model=self.model,
                messages=content,
                response_format={"type": "json_object"},
                temperature=temperature,
                top_p=top_p
            )
            return json.loads(response.choices[0].message.content)
        else:
            response = await client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": self.prompt}] + history,
                temperature=temperature,
                top_p=top_p
            )
            return response.choices[0].message.content

    def convert_to_genai_format(self, message: dict) -> dict:
        if message["role"] == "assistant":
            return {
                "role": "model",
                "parts": [{"text": message["content"]}]
            }
        return {
            "role": message["role"],
            "parts": [{"text": message["content"]}]
        }

    async def _generate_response_google(self, client, history: list = [], json_resp=False, temperature=0.5, top_p=0.9) -> str:
        
        converted_messages = [self.convert_to_genai_format(msg) for msg in history]
        
        if json_resp:
            if converted_messages == []:
                response = await client.aio.models.generate_content(
                    contents=self.prompt,
                    model=self.model,
                    config=types.GenerateContentConfig(
                        response_mime_type='application/json',
                    ))
            else:
                response = await client.aio.models.generate_content(
                    contents=converted_messages,
                    model=self.model,
                    config=types.GenerateContentConfig(
                        system_instruction=self.prompt,
                        response_mime_type='application/json',
                    ))
            return json.loads(response.text)
        else:
            if converted_messages == []:
                response = await client.aio.models.generate_content(
                    contents=self.prompt,
                    model=self.model,
                    config=types.GenerateContentConfig()
                )
            else:
                response = await client.aio.models.generate_content(
                    contents=converted_messages,
                    model=self.model,
                    config=types.GenerateContentConfig(
                        system_instruction=self.prompt,
                    )
                )
            return response.text

    async def _generate_response_together(self, client, history: list = [], json_resp=False, 
                                        temperature=0.5, top_p=0.9) -> str:
        if json_resp:
            content = [{"role": "system", "content": self.prompt}] + history
            response = await client.chat.completions.create(
                model=self.model,
                messages=content,
                response_format={"type": "json_object"},
                temperature=temperature,
                top_p=top_p
            )
            return json.loads(response.choices[0].message.content)
        else:
            response = await client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": self.prompt}] + history,
                temperature=temperature,
                top_p=top_p
            )
            return response.choices[0].message.content
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics for all keys."""
        return {
            'total_keys': len(self.clients),
            'failed_keys': len(self.failed_keys),
            'key_usage': self.key_usage,
            'rotation_strategy': self.key_rotation_strategy
        }
    
    def print_usage_stats(self):
        """Print usage statistics in a readable format."""
        stats = self.get_usage_stats()
        print(f"\n=== Usage Statistics for {self.name} ===")
        print(f"Total Keys: {stats['total_keys']}")
        print(f"Failed Keys: {stats['failed_keys']}")
        print(f"Rotation Strategy: {stats['rotation_strategy']}")
        print("\nPer-Key Statistics:")
        
        for key_idx, usage in stats['key_usage'].items():
            status = "FAILED" if key_idx in self.failed_keys else "ACTIVE"
            print(f"  Key {key_idx + 1}: {usage['requests']} requests, "
                  f"{usage['errors']} errors, Status: {status}")

async def main():
    agent = AsyncAgent(
            name="AsyncExampleAgent",
            prompt="You are a helpful assistant.",
            model="gpt-4.1",
            provider='openai'
    )
    response = await agent.generate_response(history=[{"role": "user", "content": "Hello!, return me hello"}])
    print(response)

async def example_usage():
    """Example of how to use the AsyncAgent with multiple keys."""
    
    agent = AsyncMultiAgent(
        name="MultiKeyAgent",
        prompt="You are a helpful assistant.",
        model="gemma-3n-e4b-it",
        provider='google'

    )
    
    # Generate multiple responses
    tasks = []
    for i in range(50):
        task = await agent.generate_response(
            history=[{"role": "user", "content": f"Hello! This is request {i+1}"}]
        )
        agent.print_usage_stats()

        tasks.append(task)
    
    # Execute all requests concurrently
    try:
        # responses = await asyncio.gather(*tasks)
        responses = tasks
        for i, response in enumerate(responses):
            print(f"Response {i+1}: {response[:100]}...")
            
    except Exception as e:
        print(f"Error: {e}")
    
    # Print usage statistics
    agent.print_usage_stats()

from asyncio import tasks

async def example_run_ollama():
    """Example of how to use the AsyncAgent with Ollama."""
    
    agent = AsyncAgent(
        name="OllamaAgent",
        prompt="You are a helpful assistant.",
        model="llama3.2:latest",
        provider='ollama'
    )
    print("Request Sent")
    response = await agent.generate_response(history=[{"role": "user", "content": "Hello! This is a test"}])
    print(response)
    # response = await tasks.gather(*[agent.generate_response(history=[{"role": "user", "content": "Hello! This is a test"}]) for i in range(100)])
    # print(len(response))

if __name__ == "__main__":
    asyncio.run(example_run_ollama())