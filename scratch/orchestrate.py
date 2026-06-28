#!/usr/bin/env python3
import os
import sys
import csv
import random
import subprocess
import argparse
import json

def load_manifest(manifest_path):
    binaries = []
    with open(manifest_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            binaries.append(row)
    return binaries

def get_event_groups(supported_events_path):
    events = []
    with open(supported_events_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['status'] == 'ok':
                events.append(row['event'])
    
    # Chunk into groups of 4
    groups = []
    for i in range(0, len(events), 4):
        group_events = events[i:i+4]
        groups.append({
            'group_id': f"g{len(groups):03d}",
            'events': ",".join(group_events)
        })
    return groups

def main():
    parser = argparse.ArgumentParser(description="HPC-Boost Pilot Data Collection Orchestrator")
    parser.add_argument("--manifest", required=True, help="Path to benign/malware manifest CSV")
    parser.add_argument("--supported-events", required=True, help="Path to supported_events.csv")
    parser.add_argument("--outdir", required=True, help="Output directory for raw traces")
    parser.add_argument("--reps", type=int, default=5, help="Number of repetitions per binary-group pair")
    parser.add_argument("--cpu", type=int, default=1, help="CPU core to pin executions (default: 1)")
    parser.add_argument("--interval", type=int, default=10, help="Interval in ms (default: 10)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for execution order shuffling")
    
    args = parser.parse_args()
    
    # Load inputs
    binaries = load_manifest(args.manifest)
    groups = get_event_groups(args.supported_events)
    
    print(f"Loaded {len(binaries)} binaries from manifest.")
    print(f"Constructed {len(groups)} event groups from supported events.")
    for g in groups:
        print(f"  {g['group_id']}: {g['events']}")
        
    # Generate all run combinations
    runs = []
    for binary in binaries:
        binary_id = binary['binary_id']
        binary_path = binary['path']
        binary_args = binary['args']
        timeout_sec = binary.get('timeout_sec', '30')
        
        for group in groups:
            for rep in range(args.reps):
                runs.append({
                    'binary': binary,
                    'group': group,
                    'rep': rep,
                    'timeout_sec': timeout_sec
                })
                
    # Shuffle runs to prevent thermal drift and temporal correlation
    random.seed(args.seed)
    random.shuffle(runs)
    
    total_runs = len(runs)
    print(f"Total executions scheduled: {total_runs}")
    
    runner_script = os.path.expanduser("~/hpcboost_collect/scripts/run_one_group.sh")
    if not os.path.exists(runner_script):
        runner_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_one_group.sh")
        
    print(f"Using runner script: {runner_script}")
    
    for idx, run in enumerate(runs):
        binary = run['binary']
        group = run['group']
        rep = run['rep']
        timeout_sec = run['timeout_sec']
        
        binary_id = binary['binary_id']
        binary_path = binary['path']
        binary_args = binary['args']
        group_id = group['group_id']
        events_str = group['events']
        
        # Output directory layout: outdir/binary_id/group_id/rep_rep/
        run_outdir = os.path.join(args.outdir, binary_id, group_id, f"rep_{rep}")
        os.makedirs(run_outdir, exist_ok=True)
        
        print(f"[{idx+1}/{total_runs}] Running {binary_id} | group {group_id} | rep {rep}...")
        
        # Call runner script
        cmd = [
            "/usr/bin/env", "bash", runner_script,
            binary_id, binary_path, binary_args,
            group_id, events_str, run_outdir,
            str(args.cpu), str(timeout_sec), str(args.interval)
        ]
        
        try:
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                print(f"  Runner script exited with error code {res.returncode}")
                print(f"  Stderr: {res.stderr}")
        except Exception as e:
            print(f"  Failed to run command {cmd}: {e}")
            
    print("Orchestration complete.")

if __name__ == "__main__":
    main()
