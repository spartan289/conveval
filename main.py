from conval.agent import AsyncAgent, AsyncMultiAgent
from conval.parameters import USER_PROMPT, UserProfile
from conval.simulator import SimulateChat
from conval.evaluator import evaluate_metric, evaluate_metric_with_gold, Metric
from conval.testcase import GoldenDataset, UnitTest, Chat
from conval.utils import convert_to_json, convert_to_string
import asyncio
import yaml
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

def load_yaml_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config

def validate_config(config: dict) -> None:
    """Validate required configuration parameters."""
    required_fields = [
        'agent_name', 'user_agent_model', 'assistant_agent_model',
        'user_provider', 'assistant_provider', 'judge_agent_model',
        'judge_provider', 'assistant_prompt', 'evaluate_with_user_simulation',
        'evaluate_with_gold_conversation', 'unit_tests'
    ]
    
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required configuration field: {field}")
    
    # Validate unit_tests structure
    if not isinstance(config['unit_tests'], list) or len(config['unit_tests']) == 0:
        raise ValueError("unit_tests must be a non-empty list")
    
    for i, test in enumerate(config['unit_tests']):
        if 'conv_id' not in test:
            raise ValueError(f"Missing conv_id in unit_tests[{i}]")
        if 'gold_conversation' not in test:
            raise ValueError(f"Missing gold_conversation in unit_tests[{i}]")
        if 'metrics' not in test:
            raise ValueError(f"Missing metrics in unit_tests[{i}]")

def load_prompt_file(prompt_file: str) -> str:
    """Load prompt content from file."""
    try:
        with open(prompt_file, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

def create_dataset_from_yaml(config: dict) -> GoldenDataset:
    """Create a GoldenDataset from YAML configuration."""
    dataset = GoldenDataset(dataset_name=config.get('dataset_name', 'YAML Dataset'))
    
    for test_config in config['unit_tests']:
        # Create Chat object from gold_conversation
        gold_conversation = Chat(
            name=test_config['conv_id'],
            conversation=test_config['gold_conversation']
        )
        
        # Create metrics from YAML config
        metrics = []
        for metric_config in test_config['metrics']:
            metric = Metric(
                name=metric_config['name'],
                definition=metric_config['definition'],
                scoring_criteria=metric_config.get('scoring_criteria', '')
            )
            metrics.append(metric)
        
        # Add test to dataset
        dataset.add_test(
            conv_id=test_config['conv_id'],
            gold_conversation=gold_conversation,
            metrics=metrics,
            user_profile=test_config.get('user_profile'),
            user_goal=test_config.get('user_goal'),
            user_prompt=test_config.get('user_prompt'),
            judge_prompt=test_config.get('judge_prompt')
        )
    
    return dataset

def save_results(results: list, output_path: str, config: dict) -> None:
    """Save results to file with metadata."""
    output_data = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'dataset_name': config.get('dataset_name', 'Unknown'),
            'description': config.get('description', ''),
            'version': config.get('version', '1.0'),
            'created_by': config.get('created_by', ''),
            'created_date': config.get('created_date', ''),
            'agent_name': config.get('agent_name', ''),
            'total_conversations': len(results),
            'config_summary': {
                'user_agent_model': config.get('user_agent_model'),
                'assistant_agent_model': config.get('assistant_agent_model'),
                'judge_agent_model': config.get('judge_agent_model'),
                'evaluate_with_user_simulation': config.get('evaluate_with_user_simulation'),
                'evaluate_with_gold_conversation': config.get('evaluate_with_gold_conversation')
            }
        },
        'results': results
    }
    
    # Determine file format based on extension
    output_path = Path(output_path)
    
    if output_path.suffix.lower() == '.json':
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
    elif output_path.suffix.lower() == '.yaml' or output_path.suffix.lower() == '.yml':
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(output_data, f, default_flow_style=False, allow_unicode=True)
    else:
        # Default to JSON if no extension or unknown extension
        output_path = output_path.with_suffix('.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to: {output_path}")

def generate_default_output_filename(config: dict) -> str:
    """Generate a default output filename based on config and timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = config.get('dataset_name', 'dataset').replace(' ', '_').lower()
    agent_name = config.get('agent_name', 'agent').replace(' ', '_').lower()
    return f"{dataset_name}_{agent_name}_{timestamp}_results.json"

def print_progress(current: int, total: int, conv_id: str):
    """Print progress information."""
    print(f"[{current}/{total}] Processing conversation: {conv_id}")

async def main(config_path: str = "config.yaml", output_path: str = None):
    """
    Main function to run the conversation and judge it.
    
    Args:
        config_path: Path to the YAML configuration file
        output_path: Path to save the results (optional)
    """
    # Load configuration
    config = load_yaml_config(config_path)
    validate_config(config)
    
    # Create dataset from YAML config
    dataset = create_dataset_from_yaml(config)
    
    # Load assistant prompt
    assistant_prompt = load_prompt_file(config['assistant_prompt'])
    
    unit_test_results = []
    total_tests = len(config['unit_tests'])
    
    print(f"Starting evaluation of {total_tests} test cases...")
    print(f"Dataset: {config.get('dataset_name', 'Unknown')}")
    print(f"Agent: {config.get('agent_name', 'Unknown')}")
    print("-" * 60)
    
    # Process each conversation in the dataset
    for i, gold_cov in enumerate(dataset, 1):
        print_progress(i, total_tests, gold_cov.conv_id)
        
        # Create simulator with assistant agent
        simulator = SimulateChat(
            assistant_agent=AsyncAgent(
                name=config['agent_name'],
                prompt=assistant_prompt,
                model=config['assistant_agent_model'],
                provider=config['assistant_provider']
            ),
            conversation_history=config.get('conversation_history', [
                {"role": "assistant", "content": "हेलो मैं काव्य बोल रही हूँ क्या मेरी बात Harsimran से हो रही है"}
            ]),
            endswith=config.get('endswith', '<action>call_end</action>'),
            max_turns=config.get('max_turns', 10)
        )
        
        # Configure user simulation based on config
        if config.get('evaluate_with_user_simulation', False):
            if config.get('using_gold_conversation', False):
                # Simulate with Gold Conversation
                simulator.user_agent = AsyncAgent(
                    name=config.get('user_agent_name', 'User'),
                    prompt=gold_cov.user_prompt,
                    model=config['user_agent_model'],
                    provider=config['user_provider']
                )
            else:
                # Simulate without Gold Conversation
                user_profile_details = ""
                if gold_cov.user_profile:
                    user_profile_details = UserProfile.get_profile_details(gold_cov.user_profile)['prompt']
                
                simulator.user_agent = AsyncAgent(
                    name=config.get('user_agent_name', 'User'),
                    prompt=USER_PROMPT.format(
                        user_profile=user_profile_details, 
                        user_goal=gold_cov.user_goal or ""
                    ),
                    model=config['user_agent_model'],
                    provider=config['user_provider']
                )
            
            simulated_conversation = await simulator.simulate()
        else:
            simulated_conversation = await simulator.simulatewithoutusersim(gold_cov.gold_conv.messages)
        
        # Create judge agent
        judge_agent = AsyncAgent(
            name=config.get('judge_agent_name', 'Judge'),
            model=config['judge_agent_model'],
            provider=config['judge_provider'],
        )
        
        # Evaluate the conversation
        if config.get('evaluate_with_gold_conversation', False):
            resp = await evaluate_metric(
                conversation=simulated_conversation,
                metrics=gold_cov.metrics,
                judge_agent=judge_agent
            )    
        else:
            resp = await evaluate_metric_with_gold(
                gold_conversation=convert_to_string(gold_cov.gold_conv.messages),
                simulated_conversation=simulated_conversation,
                metrics=gold_cov.metrics,
                judge_agent=judge_agent
            )

        # Store results
        unit_test_results.append({
            'conv_id': gold_cov.conv_id,
            'user_goal': gold_cov.user_goal,
            'user_profile': gold_cov.user_profile,
            'gold_conversation': gold_cov.gold_conv.messages,
            'simulated_conversation': simulated_conversation,
            'metrics': [{'name': m.name, 'definition': m.definition} for m in gold_cov.metrics],
            'metrics_results': resp
        })
        
        # Print summary for this conversation
        if config.get('verbose', False):
            print(f"  Simulated conversation: {len(simulated_conversation)} turns")
            print(f"  Metrics evaluated: {len(gold_cov.metrics)}")
            print(f"  Results: {resp}")
        
        print(f"  ✓ Completed conversation {gold_cov.conv_id}")
        print("-" * 40)
    
    # Save results
    if output_path:
        save_results(unit_test_results, output_path, config)
    else:
        # Generate default filename and save
        default_filename = generate_default_output_filename(config)
        save_results(unit_test_results, default_filename, config)
    
    # Print final summary
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETED!")
    print(f"Dataset: {config.get('dataset_name', 'Unknown')}")
    print(f"Total conversations processed: {len(unit_test_results)}")
    print(f"Agent: {config.get('agent_name', 'Unknown')}")
    print(f"Models used: {config.get('assistant_agent_model', 'Unknown')}")
    print("=" * 60)
    
    return unit_test_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run conversation simulation and evaluation with YAML config')
    parser.add_argument('config', nargs='?', default='config.yaml',
                       help='Path to YAML configuration file (default: config.yaml)')
    parser.add_argument('--output', '-o', 
                       help='Path to save results (supports .json, .yaml, .yml). If not provided, generates default filename')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output (overrides config setting)')
    
    args = parser.parse_args()
    
    # Check if config file exists
    if not Path(args.config).exists():
        print(f"Configuration file not found: {args.config}")
        print("Please provide a valid YAML configuration file.")
        exit(1)
    
    # Override verbose setting if provided via command line
    if args.verbose:
        # We'll handle this in the main function
        pass
    
    asyncio.run(main(args.config, args.output))
