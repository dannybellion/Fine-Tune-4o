from openai import OpenAI
import os
import json
from typing import Optional, List, Dict, Any
import time
import pandas as pd
import matplotlib.pyplot as plt

class FineTuner:
    """Class to handle OpenAI fine-tuning operations"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize with optional API key, otherwise uses env var"""
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def prepare_chat_data(self, conversations: List[Dict[str, Any]], output_file: str) -> str:
        """
        Convert chat conversations to JSONL format for fine-tuning
        Args:
            conversations: List of conversation dicts with messages
            output_file: Path to save JSONL file
        Returns:
            Path to created JSONL file
        """
        with open(output_file, 'w') as f:
            for conv in conversations:
                json.dump({"messages": conv["messages"]}, f)
                f.write('\n')
        return output_file

    def upload_training_file(self, file_path: str) -> str:
        """
        Upload training file to OpenAI
        Args:
            file_path: Path to JSONL training file
        Returns:
            File ID from OpenAI
        """
        with open(file_path, 'rb') as f:
            response = self.client.files.create(
                file=f,
                purpose='fine-tune'
            )
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
        return response.id

    def get_job_status(self, job_id: str) -> Dict:
        """
        Get status of a fine-tuning job
        Args:
            job_id: ID of fine-tuning job
        Returns:
            Job status details
        """
        return self.client.fine_tuning.jobs.retrieve(job_id)

    def wait_for_job(self, job_id: str, poll_interval: int = 60) -> Dict:
        """
        Wait for job completion, polling at specified interval
        Args:
            job_id: ID of fine-tuning job
            poll_interval: Seconds between status checks
        Returns:
            Final job status
        """
        while True:
            job = self.get_job_status(job_id)
            if job.status in ["succeeded", "failed"]:
                return job
            time.sleep(poll_interval)

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

    def get_training_metrics(self, job_id: str) -> pd.DataFrame:
        """
        Get training metrics for a fine-tuning job and return as DataFrame
        Args:
            job_id: ID of fine-tuning job
        Returns:
            DataFrame with training metrics
        """
        events = self.client.fine_tuning.jobs.list_events(job_id)
        metrics_data = []
        
        for event in events:
            if hasattr(event, 'data') and event.data is not None:
                metrics_data.append({
                    'step': event.data.get('step', None),
                    'train_loss': event.data.get('train_loss', None),
                    'valid_loss': event.data.get('valid_loss', None),
                    'train_mean_token_accuracy': event.data.get('train_mean_token_accuracy', None),
                    'valid_mean_token_accuracy': event.data.get('valid_mean_token_accuracy', None)
                })
        
        return pd.DataFrame(metrics_data)

    def plot_training_metrics(self, job_id: str, figsize: tuple = (12, 6)) -> None:
        """
        Plot training metrics for a fine-tuning job
        Args:
            job_id: ID of fine-tuning job
            figsize: Size of the figure (width, height)
        """
        df = self.get_training_metrics(job_id)
        
        plt.figure(figsize=figsize)
        
        # Plot losses
        plt.subplot(1, 2, 1)
        plt.plot(df['step'], df['train_loss'], label='Training Loss')
        plt.plot(df['step'], df['valid_loss'], label='Validation Loss')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot accuracies
        plt.subplot(1, 2, 2)
        plt.plot(df['step'], df['train_mean_token_accuracy'], 
                label='Training Accuracy')
        plt.plot(df['step'], df['valid_mean_token_accuracy'], 
                label='Validation Accuracy')
        plt.xlabel('Step')
        plt.ylabel('Token Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
