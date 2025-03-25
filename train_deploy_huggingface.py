import os
import argparse
import subprocess
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback,
)
from datasets import Dataset
import pandas as pd
import torch
import json
import deepspeed
import shutil 

def list_dir_contents(path):
    """
    Recursively lists all files and directories within the given path,
    along with their sizes.
    """
    print(f"\nContents of '{path}':")
    for root, dirs, files in os.walk(path):
        level = root.replace(path, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        sub_indent = ' ' * 4 * (level + 1)
        for f in files:
            file_path = os.path.join(root, f)
            size = os.path.getsize(file_path) / 1e6  # Size in MB
            print(f"{sub_indent}{f} - {size:.2f} MB")

def main():
    parser = argparse.ArgumentParser()

    # Hyperparameters
    parser.add_argument("--batch_size", type=int, default=1, help="Per-device training batch size")
    parser.add_argument("--epochs", type=int, default=4, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--max_length", type=int, default=4000, help="Maximum sequence length")

    # Environment variables set by SageMaker
    parser.add_argument("--model_dir", type=str, default="./model")
    parser.add_argument("--output_data_dir", type=str, default="./output")
    parser.add_argument("--train_dir", type=str, default="./data/train")
    parser.add_argument("--validation_dir", type=str, default="./data/validation")

    # **Add the following line to accept --local_rank**
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")

    args = parser.parse_args()

    #deepspeed.init_distributed()
    #device = torch.device(f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "./models_for_training/gemma2-no-rope-scaling",#"Exaone-3.5-it",
        #"meta-llama/Llama-3.1-8B-Instruct",
        #"meta-llama/Llama-3.2-3B-Instruct",
        use_fast=True,
        add_eos_token=True,
        add_bos_token=True,
        padding_side="left",
        trust_remote_code=True,
        #token='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'  # Replace with your actual Hugging Face token
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        "./models_for_training/gemma2-no-rope-scaling",
        #"meta-llama/Llama-3.1-8B-Instruct",
        #"meta-llama/Llama-3.2-3B-Instruct",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        #token='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',  # Replace with your actual Hugging Face token
    )
    
    
    #model.to(device)
    #model.config.rope_scaling = {"type": "linear", "factor": 2.0}
    #model.config.use_cache = False
    model.gradient_checkpointing_enable()

    
    with open('./deepspeed_config.json', 'r') as config_file:
        deepspeed_config = json.load(config_file)
    


    # Move the model to the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model.to(device)
    print(f"Model device: {next(model.parameters()).device}")
    
    #model=model.to('cuda')
    # Prepare data
    def load_and_tokenize_data(input_path):
        df = pd.read_csv(input_path, sep='\t', header=None)
        texts = df[0].tolist()
        
        # Print the first sample for verification
        print("First sample before tokenization:")
        print(texts[0])
        
        tokenized_data = tokenizer(
            texts,
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
        )
        
        # Print the tokenized version of the first sample
        print("\nTokenized input_ids of the first sample:")
        print(tokenized_data["input_ids"][0])
        
        # Optionally, decode the tokenized input_ids back to text for verification
        decoded_text = tokenizer.decode(tokenized_data["input_ids"][0])
        print("\nDecoded text from tokenized input_ids:")
        print(decoded_text)
        
        tokenized_data["labels"] = tokenized_data["input_ids"].copy()
        
        dataset = Dataset.from_dict(tokenized_data)
        
        # Print an example from the dataset
        print("\nDataset example:")
        print(dataset[0])
        
        return dataset

    print("Training data path:")
    print(os.path.join(args.train_dir, 'exaone_train_set.tsv'))

    train_dataset = load_and_tokenize_data(os.path.join(args.train_dir, 'exaone_train_set.tsv'))
    eval_dataset = load_and_tokenize_data(os.path.join(args.validation_dir, 'exaone_validation_set.tsv'))

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Define separate directory for checkpoints
    checkpoints_dir = os.path.join(args.model_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Training arguments with DeepSpeed integration
    training_args = TrainingArguments(
        output_dir=checkpoints_dir,  # Checkpoints saved here,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=3e-7,
        bf16=True,
        logging_dir=os.path.join(args.output_data_dir, "logs"),
        logging_steps=10,
        log_level='debug',
        #eval_steps=50,
        #save_total_limit=1,
        load_best_model_at_end=False,
        evaluation_strategy="epoch",         # Match with save_strategy
        save_strategy="epoch",               # Ensure it matches evaluation_strategy
        save_total_limit=2,                  # Retain only best and last checkpoints
        metric_for_best_model="eval_loss",    # Specify the metric for best model
        #metric_for_best_model="eval_loss",
        dataloader_num_workers=4,
        deepspeed=deepspeed_config#"deepspeed_config.json",  # Specify DeepSpeed config file
    )

    # Custom callback to log GPU stats
    class GPUStatsCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            gpu = int(os.environ.get("LOCAL_RANK", -1))
            if gpu == -1:
                return  # Skip if not using GPU
            # Synchronize to ensure all computations are done
            torch.cuda.synchronize(gpu)
            allocated = torch.cuda.memory_allocated(gpu) / 1e9  # Convert to GB
            reserved = torch.cuda.memory_reserved(gpu) / 1e9  # Convert to GB

            # Optionally, get GPU utilization using nvidia-smi
            try:
                result = subprocess.check_output(
                    ['nvidia-smi', '--id={}'.format(gpu), '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,nounits,noheader'],
                    encoding='utf-8'
                )
                gpu_util, mem_used, mem_total = result.strip().split(',')
                gpu_util = int(gpu_util)
                mem_used = float(mem_used) / 1e3  # Convert MB to GB
                mem_total = float(mem_total) / 1e3  # Convert MB to GB
            except Exception as e:
                gpu_util = 'N/A'
                mem_used = allocated
                mem_total = reserved
                print(f"Error getting GPU utilization: {e}")

            print(f"After step {state.global_step}: GPU {gpu}, Utilization: {gpu_util}%, Memory Used: {mem_used:.2f} GB / {mem_total:.2f} GB")
            
    # Initialize Trainer with DeepSpeed and callbacks
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        #processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=[GPUStatsCallback()],  # Add the custom callback here
    )

    # Start training
    trainer.train()
    
    
    # Save the model - **all processes must call this**
    print("Model trained successfully, proceeding to save...")
    trainer.save_model(args.model_dir)  # All processes call this

    # Only the main process handles tokenizer saving, model card creation, and cleanup
    if trainer.is_world_process_zero():
        print("Saving tokenizer and creating model card...")
        tokenizer.save_pretrained(args.model_dir)
        trainer.create_model_card()

        print("Saving completed. Verifying saved files...")
        # List the contents of the model directory
        list_dir_contents(args.model_dir)

        # Optionally, remove the checkpoints directory to free up space
        try:
            shutil.rmtree(checkpoints_dir)
            print(f"Removed checkpoints directory: {checkpoints_dir}")
        except Exception as e:
            print(f"Error removing checkpoints directory: {e}")
        
        list_dir_contents(args.model_dir)

        print("Done!")


    """print("Model trained successfully, proceeding to save...")
    print("Saving tokenizer...")
    # Save our tokenizer and create model card
    tokenizer.save_pretrained(args.model_dir)
    trainer.create_model_card()
    # Push the results to the hub
    #if args.repository_id:
    #    trainer.push_to_hub()

    # Saves the model to s3 uses os.environ["SM_MODEL_DIR"] to make sure checkpointing works
    print("Saving model...")
    trainer.save_model(os.environ["SM_MODEL_DIR"])
    tokenizer.save_pretrained(os.environ["SM_MODEL_DIR"])
    print("done!")"""
    
    """# Only the main process should handle saving
    if trainer.is_world_process_zero():
        print("Model trained successfully, proceeding to save...")
        print("Saving tokenizer...")
        tokenizer.save_pretrained(args.model_dir)
        trainer.create_model_card()
        
        print("Saving model...")
        trainer.save_model(args.model_dir)
        tokenizer.save_pretrained(args.model_dir)
        print("done!")"""
    
    # Save the model (only on the main process)
    #print("Model trained successfully, proceeding to save...")
    #if trainer.is_world_process_zero():
    #    print("Saving ongoing...")
    #    print("Files in model_dir:", os.listdir(args.model_dir))
    #    model.save_pretrained(args.model_dir, safe_serialization=False)
    #    #trainer.save_model(args.model_dir)
    #    print("Files in model_dir:", os.listdir(args.model_dir))
    #    print("done!")

if __name__ == "__main__":
    main()
