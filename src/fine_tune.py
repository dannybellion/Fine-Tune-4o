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

    # def prepare_chat_data(self, conversations: List[Dict[str, Any]], output_file: str) -> str:
    #     """
    #     Convert chat conversations to JSONL format for fine-tuning
    #     Args:
    #         conversations: List of conversation dicts with messages
    #         output_file: Path to save JSONL file
    #     Returns:
    #         Path to created JSONL file
    #     """
    #     with open(output_file, 'w') as f:
    #         for conv in conversations:
    #             json.dump({"messages": conv["messages"]}, f)
    #             f.write('\n')
    #     return output_file

    def upload_training_file(self, file_path: str) -> str:
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
        
        print(f"Total tokens in training file: {total_tokens}")
        
        with open(file_path, 'rb') as f:
            response = self.client.files.create(
                file=f,
                purpose='fine-tune'
            )
            print(f"File uploaded successfully. File ID: {response.id}")
            print(f"Total tokens in file: {total_tokens}")
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
        
        # Calculate estimated completion time
        if hasattr(status, 'created_at') and hasattr(status, 'hyperparameters'):
            start_time = datetime.fromtimestamp(status.created_at)
            
            # Handle 'auto' epochs setting
            n_epochs = status.hyperparameters.n_epochs
            if n_epochs == 'auto':
                n_epochs = 3  # Default estimate for auto mode
            else:
                n_epochs = int(n_epochs)
                
            # Estimate: tokens/1000 * epochs = minutes
            if hasattr(status, 'trained_tokens'):
                est_minutes = (status.trained_tokens / 1000) * n_epochs
                est_completion = start_time + timedelta(minutes=est_minutes)
                
                print(f"Status: {status.status}")
                print(f"Model: {status.model}")
                print(f"Started at: {start_time}")
                print(f"Estimated completion: {est_completion}")
            else:
                print(f"Status: {status.status}")
                print(f"Model: {status.model}")
                print(f"Started at: {start_time}")
            print(status.hyperparameters)
        else:
            print(f"Status: {status.status}")
            print(f"Model: {status.model}")
            print(status.hyperparameters)

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
        epoch_metrics = []
        
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
                if data.get('epoch') is not None:  # Epoch metrics
                    epoch_metrics.append({
                        'epoch': data.get('epoch', None),
                        'train_loss': data.get('train_loss', None),
                        'valid_loss': data.get('valid_loss', None),
                        'train_mean_token_accuracy': data.get('train_mean_token_accuracy', None),
                        'valid_mean_token_accuracy': data.get('valid_mean_token_accuracy', None)
                    })
        
        return pd.DataFrame(step_metrics), pd.DataFrame(epoch_metrics)

    def plot_training_metrics(self, job_id: str, figsize: tuple = (12, 12)) -> None:
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
        
        # Plot epoch-based metrics if available
        if not epoch_df.empty:
            # Loss by epoch
            plt.subplot(2, 2, 3)
            plt.plot(epoch_df['epoch'], epoch_df['train_loss'], label='Training Loss')
            plt.plot(epoch_df['epoch'], epoch_df['valid_loss'], label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Loss by Epoch')
            plt.legend()
            plt.grid(True)
            
            # Accuracy by epoch
            plt.subplot(2, 2, 4)
            plt.plot(epoch_df['epoch'], epoch_df['train_mean_token_accuracy'], 
                    label='Training Accuracy')
            plt.plot(epoch_df['epoch'], epoch_df['valid_mean_token_accuracy'], 
                    label='Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Token Accuracy')
            plt.title('Accuracy by Epoch')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.show()
