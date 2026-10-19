import yaml
import argparse
import shutil
import os
import glob
import lightly_train

def run_training():
    # 1. Setup Argument Parser
    parser = argparse.ArgumentParser(description="DINOv2 Trainer")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.yaml", 
        help="Path to the YAML configuration file"
    )
    args = parser.parse_args()

    # 2. Load the YAML config
    print(f"Loading configuration from: {args.config}")
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # 2. Get output directory and create it
    out_dir = config.get("out", "./output")
    os.makedirs(out_dir, exist_ok=True)

    # 3. Copy the YAML config to the output directory immediately
    config_filename = os.path.basename(args.config)
    shutil.copy2(args.config, os.path.join(out_dir, f"used_{config_filename}"))
    print(f"Config copied to: {out_dir}")

    # 4. Launch training
    # The **config syntax unpacks the YAML keys directly into the function
    lightly_train.train(**config)

    # Export all saved checkpoints (supports both save_top_k=1 and save_top_k=-1)
    best_ckpt_pattern = os.path.join(out_dir, "checkpoints", "epoch=*.ckpt")
    all_ckpts = [p for p in glob.glob(best_ckpt_pattern) if "last" not in p]
    if all_ckpts:
        # Sort by epoch number (PL names are 0-indexed: epoch=49 → epoch 50)
        def _epoch_num(path):
            try:
                return int(os.path.basename(path).split("epoch=")[1].split("-")[0])
            except (IndexError, ValueError):
                return -1

        all_ckpts = sorted(all_ckpts, key=_epoch_num)
        exported_dir = os.path.join(out_dir, "exported_models")
        os.makedirs(exported_dir, exist_ok=True)
        for ckpt in all_ckpts:
            epoch_idx = _epoch_num(ckpt)          # 0-indexed
            epoch_label = epoch_idx + 1            # human-readable (1-indexed)
            exported_path = os.path.join(exported_dir, f"epoch_{epoch_label}.pt")
            print(f"Exporting checkpoint (epoch {epoch_label}): {ckpt}")
            lightly_train.export(
                out=exported_path,
                checkpoint=ckpt,
                part="model",
            )
            print(f"  → {exported_path}")
        print(f"Exported {len(all_ckpts)} checkpoint(s) to {exported_dir}")
    else:
        print("Warning: No checkpoints found to export.")
    
    # 5. Attempt to copy the SLURM log file at the end
    # We look for files in logs/ that match the current job ID
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id:
        log_files = glob.glob(f"logs/*{job_id}*.log")
        for log in log_files:
            shutil.copy2(log, out_dir)
            print(f"Log file {log} copied to {out_dir}")

if __name__ == "__main__":
    run_training()

