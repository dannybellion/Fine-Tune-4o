from openai import OpenAI
import os
import json
import tiktoken
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import time
import pandas as pd
import matplotlib.pyplot as plt

class FineTuner:
    """Class to handle OpenAI fine-tuning operations"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize with optional API key, otherwise uses env var"""
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.training_tokens = {}  # Store token counts by file ID
        self.training_examples = {}

    def upload_file(self, file_path: str) -> str:
        """
        Upload training file to OpenAI
        Args:
            file_path: Path to JSONL training file
        Returns:
            File ID from OpenAI
        """
        # Count tokens in the file
        enc = tiktoken.get_encoding("cl100k_base")
        total_tokens = 0
        
        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                for message in data.get('messages', []):
                    total_tokens += len(enc.encode(message.get('content', '')))
        
        with open(file_path, 'rb') as f:
            response = self.client.files.create(
                file=f,
                purpose='fine-tune'
            )
            print(f"File uploaded successfully. File ID: {response.id}")
            print(f"Total tokens in file: {total_tokens}")
            
            # Reset file pointer to beginning and count rows
            f.seek(0)
            total_examples = sum(1 for _ in f)
            print(f"Total examples in file: {total_examples}")
            
            self.training_tokens[response.id] = total_tokens
            self.training_examples[response.id] = total_examples
        return response.id

    def create_job(self, 
                  training_file: str,
                  model: str = "gpt-4o-mini-2024-07-18",
                  validation_file: Optional[str] = None,
                  hyperparameters: Optional[Dict] = None) -> str:
        """
        Create a fine-tuning job
        Args:
            training_file: File ID from upload
            model: Base model to fine-tune
            validation_file: Optional validation file ID
            hyperparameters: Optional dict of hyperparameters
        Returns:
            Job ID
        """
        job_args = {
            "training_file": training_file,
            "model": model
        }
        
        if validation_file:
            job_args["validation_file"] = validation_file
            
        if hyperparameters:
            job_args["hyperparameters"] = hyperparameters

        response = self.client.fine_tuning.jobs.create(**job_args)
        print(f"Job created successfully. Job ID: {response.id}")
        return response.id

    def get_job_status(self, job_id: str) -> Dict:
        """
        Get status of a fine-tuning job
        Args:
            job_id: ID of fine-tuning job
        Returns:
            Job status details
        """
        
        status = self.client.fine_tuning.jobs.retrieve(job_id)
        
        print(f"Status: {status.status}")
        print(f"Model: {status.model}")
        if hasattr(status, 'created_at'):
            start_time = datetime.fromtimestamp(status.created_at)
            print(f"Started at: {start_time}")
        
        if isinstance(status.hyperparameters.n_epochs, int):
            n_epochs = status.hyperparameters.n_epochs
            training_file = status.training_file
            tokens = self.training_tokens[training_file]
            est_minutes = (tokens / 1000) * n_epochs
            est_completion = start_time + timedelta(minutes=est_minutes)
            print(f"Estimated completion: {est_completion}")
        
        # Get latest training step
        step_df = self.get_training_metrics(job_id)
        if not step_df.empty:
            latest_step = step_df['step'].max()
            total_steps = n_epochs * self.training_examples[training_file]
            if total_steps:
                print(f"Training progress: Step {latest_step}/{total_steps}")
                
        print(f"Hyperparameters: {status.hyperparameters}")


    def list_jobs(self, limit: int = 10) -> List[Dict]:
        """
        List fine-tuning jobs
        Args:
            limit: Max number of jobs to return
        Returns:
            List of job details
        """
        return list(self.client.fine_tuning.jobs.list(limit=limit))

    def cancel_job(self, job_id: str) -> Dict:
        """
        Cancel a fine-tuning job
        Args:
            job_id: ID of job to cancel
        Returns:
            Cancelled job details
        """
        return self.client.fine_tuning.jobs.cancel(job_id)

    def get_training_metrics(self, job_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get training metrics for a fine-tuning job and return as DataFrames
        Args:
            job_id: ID of fine-tuning job
        Returns:
            Tuple of (step metrics DataFrame, epoch metrics DataFrame)
        """
        events = self.client.fine_tuning.jobs.list_events(job_id)
        step_metrics = []
        
        for event in events:
            if hasattr(event, 'data') and event.data is not None:
                data = event.data
                if data.get('step') is not None:  # Step metrics
                    step_metrics.append({
                        'step': data.get('step', None),
                        'train_loss': data.get('train_loss', None),
                        'valid_loss': data.get('valid_loss', None),
                        'train_mean_token_accuracy': data.get('train_mean_token_accuracy', None),
                        'valid_mean_token_accuracy': data.get('valid_mean_token_accuracy', None)
                    })
        
        return pd.DataFrame(step_metrics)

    def plot_training_metrics(self, job_id: str, figsize: tuple = (12, 6)) -> None:
        """
        Plot training metrics for a fine-tuning job
        Args:
            job_id: ID of fine-tuning job
            figsize: Size of the figure (width, height)
        """
        step_df, epoch_df = self.get_training_metrics(job_id)
        
        plt.figure(figsize=figsize)
        
        # Plot step-based metrics
        # Loss by step
        plt.subplot(2, 2, 1)
        plt.plot(step_df['step'], step_df['train_loss'], label='Training Loss')
        plt.plot(step_df['step'], step_df['valid_loss'], label='Validation Loss')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title('Loss by Step')
        plt.legend()
        plt.grid(True)
        
        # Accuracy by step
        plt.subplot(2, 2, 2)
        plt.plot(step_df['step'], step_df['train_mean_token_accuracy'], 
                label='Training Accuracy')
        plt.plot(step_df['step'], step_df['valid_mean_token_accuracy'], 
                label='Validation Accuracy')
        plt.xlabel('Step')
        plt.ylabel('Token Accuracy')
        plt.title('Accuracy by Step')
        plt.legend()
        plt.grid(True)
        
        
        plt.tight_layout()
        plt.show()
