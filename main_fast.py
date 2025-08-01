from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from pathlib import Path
import asyncio
import json
from typing import Optional, Dict, Any, List
from datetime import datetime
import uuid
import tempfile
import os

# Import your existing modules
from conval.agent import AsyncAgent, AsyncMultiAgent
from conval.parameters import USER_PROMPT, UserProfile
from conval.simulator import SimulateChat
from conval.evaluator import evaluate_metric, evaluate_metric_with_gold, Metric
from conval.testcase import GoldenDataset, UnitTest, Chat
from conval.utils import convert_to_json, convert_to_string
import yaml

app = FastAPI(title="Conversation Evaluation API", version="1.0.0")

# Import your existing functions
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

# Pydantic models for request/response
class EvaluationRequest(BaseModel):
    config_path: str
    output_path: Optional[str] = None
    verbose: bool = False

class EvaluationResponse(BaseModel):
    job_id: str
    status: str
    message: str
    started_at: datetime

class EvaluationResult(BaseModel):
    job_id: str
    status: str
    total_conversations: int
    results: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    completed_at: Optional[datetime] = None
    error: Optional[str] = None

job_storage: Dict[str, EvaluationResult] = {}

@app.get("/")
async def root():
    return {"message": "Conversation Evaluation API", "version": "1.0.0"}

@app.post("/evaluate", response_model=EvaluationResponse)
async def start_evaluation(request: EvaluationRequest, background_tasks: BackgroundTasks):
    """
    Start a conversation evaluation job with a YAML configuration file.
    
    Args:
        request: Contains the path to YAML config file and optional parameters
        
    Returns:
        EvaluationResponse with job_id and status
    """
    # Validate config file exists
    if not Path(request.config_path).exists():
        raise HTTPException(
            status_code=400, 
            detail=f"Configuration file not found: {request.config_path}"
        )
    
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    
    # Initialize job in storage
    job_storage[job_id] = EvaluationResult(
        job_id=job_id,
        status="started",
        total_conversations=0,
        results=[],
        metadata={}
    )
    
    # Add background task
    background_tasks.add_task(
        run_evaluation_job, 
        job_id, 
        request.config_path, 
        request.output_path, 
        request.verbose
    )
    
    return EvaluationResponse(
        job_id=job_id,
        status="started",
        message="Evaluation job started successfully",
        started_at=datetime.now()
    )

@app.get("/evaluate/{job_id}", response_model=EvaluationResult)
async def get_evaluation_status(job_id: str):
    """
    Get the status and results of an evaluation job.
    
    Args:
        job_id: The unique identifier for the evaluation job
        
    Returns:
        EvaluationResult with current status and results
    """
    if job_id not in job_storage:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return job_storage[job_id]

@app.get("/evaluate/{job_id}/results")
async def get_evaluation_results(job_id: str):
    """
    Get only the results of a completed evaluation job.
    
    Args:
        job_id: The unique identifier for the evaluation job
        
    Returns:
        JSON response with results data
    """
    if job_id not in job_storage:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = job_storage[job_id]
    
    if job.status not in ["completed"]:
        raise HTTPException(
            status_code=400, 
            detail=f"Job is not completed yet. Current status: {job.status}"
        )
    
    return {
        "job_id": job_id,
        "status": job.status,
        "total_conversations": job.total_conversations,
        "results": job.results,
        "metadata": job.metadata,
        "completed_at": job.completed_at
    }

@app.get("/evaluate/{job_id}/download")
async def download_evaluation_results(
    job_id: str, 
    format: str = "json",
    include_metadata: bool = True
):
    """
    Download evaluation results as a file.
    
    Args:
        job_id: The unique identifier for the evaluation job
        format: File format (json, yaml, csv)
        include_metadata: Whether to include metadata in the file
        
    Returns:
        FileResponse with the results file
    """
    if job_id not in job_storage:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = job_storage[job_id]
    
    if job.status != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Job is not completed yet. Current status: {job.status}"
        )
    
    # Prepare data for download
    if include_metadata:
        download_data = {
            'metadata': {
                'job_id': job_id,
                'timestamp': job.completed_at.isoformat() if job.completed_at else datetime.now().isoformat(),
                'total_conversations': job.total_conversations,
                **job.metadata
            },
            'results': job.results
        }
    else:
        download_data = job.results
    
    # Create temporary file
    temp_dir = tempfile.mkdtemp()
    
    if format.lower() == "json":
        filename = f"evaluation_results_{job_id}.json"
        filepath = os.path.join(temp_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(download_data, f, indent=2, ensure_ascii=False, default=str)
        
        return FileResponse(
            filepath, 
            media_type='application/json',
            filename=filename,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    
    elif format.lower() in ["yaml", "yml"]:
        filename = f"evaluation_results_{job_id}.yaml"
        filepath = os.path.join(temp_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(download_data, f, default_flow_style=False, allow_unicode=True)
        
        return FileResponse(
            filepath, 
            media_type='application/x-yaml',
            filename=filename,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    
    elif format.lower() == "csv":
        filename = f"evaluation_results_{job_id}.csv"
        filepath = os.path.join(temp_dir, filename)
        
        # Convert results to CSV format
        import csv
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            if job.results:
                # Extract headers from first result
                headers = ['conv_id', 'user_goal', 'user_profile']
                
                # Add metric columns
                if 'metrics' in job.results[0]:
                    for metric in job.results[0]['metrics']:
                        headers.append(f"metric_{metric['name']}")
                
                # Add metric results columns
                if 'metrics_results' in job.results[0]:
                    for i, result in enumerate(job.results[0]['metrics_results']):
                        headers.append(f"score_{i}")
                
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                
                for result in job.results:
                    row = {
                        'conv_id': result.get('conv_id', ''),
                        'user_goal': result.get('user_goal', ''),
                        'user_profile': result.get('user_profile', '')
                    }
                    
                    # Add metric definitions
                    for metric in result.get('metrics', []):
                        row[f"metric_{metric['name']}"] = metric.get('definition', '')
                    
                    # Add metric scores
                    for i, score in enumerate(result.get('metrics_results', [])):
                        row[f"score_{i}"] = score
                    
                    writer.writerow(row)
        
        return FileResponse(
            filepath, 
            media_type='text/csv',
            filename=filename,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    
    else:
        raise HTTPException(
            status_code=400, 
            detail="Unsupported format. Use 'json', 'yaml', or 'csv'"
        )

@app.get("/jobs")
async def list_jobs():
    """
    List all evaluation jobs and their statuses.
    
    Returns:
        Dictionary of all jobs with their current status
    """
    return {
        job_id: {
            "status": job.status,
            "total_conversations": job.total_conversations,
            "completed_at": job.completed_at
        }
        for job_id, job in job_storage.items()
    }

@app.delete("/evaluate/{job_id}")
async def delete_job(job_id: str):
    """
    Delete a completed evaluation job from storage.
    
    Args:
        job_id: The unique identifier for the evaluation job
        
    Returns:
        Success message
    """
    if job_id not in job_storage:
        raise HTTPException(status_code=404, detail="Job not found")
    
    del job_storage[job_id]
    return {"message": f"Job {job_id} deleted successfully"}

async def run_evaluation_job(job_id: str, config_path: str, output_path: Optional[str], verbose: bool):
    """
    Background task to run the evaluation job.
    
    Args:
        job_id: Unique identifier for the job
        config_path: Path to YAML configuration file
        output_path: Optional path to save results
        verbose: Whether to enable verbose output
    """
    try:
        # Update job status
        job_storage[job_id].status = "running"
        
        # Load and validate configuration
        config = load_yaml_config(config_path)
        validate_config(config)
        
        # Create dataset from YAML config
        dataset = create_dataset_from_yaml(config)
        
        # Load assistant prompt
        assistant_prompt = load_prompt_file(config['assistant_prompt'])
        
        unit_test_results = []
        total_tests = len(config['unit_tests'])
        
        # Update job with total conversations
        job_storage[job_id].total_conversations = total_tests
        job_storage[job_id].metadata = {
            'dataset_name': config.get('dataset_name', 'Unknown'),
            'agent_name': config.get('agent_name', 'Unknown'),
            'config_summary': {
                'user_agent_model': config.get('user_agent_model'),
                'assistant_agent_model': config.get('assistant_agent_model'),
                'judge_agent_model': config.get('judge_agent_model'),
                'evaluate_with_user_simulation': config.get('evaluate_with_user_simulation'),
                'evaluate_with_gold_conversation': config.get('evaluate_with_gold_conversation')
            }
        }
        
        # Process each conversation in the dataset
        for i, gold_cov in enumerate(dataset, 1):
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
                    conversation=simulated_conversation,
                    metrics=gold_cov.metrics,
                    judge_agent=judge_agent
                )

            # Store results
            result = {
                'conv_id': gold_cov.conv_id,
                'user_goal': gold_cov.user_goal,
                'user_profile': gold_cov.user_profile,
                'gold_conversation': gold_cov.gold_conv.messages,
                'simulated_conversation': simulated_conversation,
                'metrics': [{'name': m.name, 'definition': m.definition} for m in gold_cov.metrics],
                'metrics_results': resp
            }
            
            unit_test_results.append(result)
            
            # Update job progress
            job_storage[job_id].results = unit_test_results
            job_storage[job_id].status = f"processing ({i}/{total_tests})"
        
        # Save results if output path provided
        if output_path:
            save_results(unit_test_results, output_path, config)
        else:
            # Generate default filename and save
            default_filename = generate_default_output_filename(config)
            save_results(unit_test_results, default_filename, config)
        
        # Mark job as completed
        job_storage[job_id].status = "completed"
        job_storage[job_id].completed_at = datetime.now()
        job_storage[job_id].results = unit_test_results
        
    except Exception as e:
        # Mark job as failed
        job_storage[job_id].status = "failed"
        job_storage[job_id].error = str(e)
        job_storage[job_id].completed_at = datetime.now()

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)