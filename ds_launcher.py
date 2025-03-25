import subprocess
import argparse
import torch
import os 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_gpus", type=int, default=torch.cuda.device_count(), help="Number of GPUs")
    parser.add_argument("--master_port", type=int, default=29501)
    args, remaining_args = parser.parse_known_args()

    env = {
        **dict(os.environ),
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": str(args.master_port),
        "NCCL_SOCKET_IFNAME": "lo",
        "GLOO_SOCKET_IFNAME": "lo"
    }

    command = f"deepspeed --num_gpus={args.num_gpus} train_deploy_huggingface.py {' '.join(remaining_args)}"
    print(f"Running command: {command}")

    subprocess.run(command, shell=True, check=True, env=env)

if __name__ == "__main__":
    main()
