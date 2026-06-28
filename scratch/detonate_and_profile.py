#!/usr/bin/env python3
"""HOST orchestrator for bare-metal namespace-based malware/benign HPC collection.

Safety & Parity Invariants:
  1. Both benign and malware are run inside the ns-malware network namespace.
  2. Both are run as the 'mw_sandbox' user to restrict permissions.
  3. Both are pinned to CPU Core 1 using taskset.
  4. Both are profiled on Core 1 using 'perf stat -C 1'.
  5. The home directory /home/mw_sandbox is cleaned/seeded before every run.
  6. Egress to physical network is blocked; dns/http requests go to local INetSim.
"""
import argparse
import csv
import os
import random
import subprocess
import sys
import time

# Host execution parameters
CPU_CORE = 1
SANDBOX_USER = "mw_sandbox"
NAMESPACE = "ns-malware"
SCRIPTS_DIR = "/home/iiitd/hpcboost_collect/scripts/mw"

def run_cmd(cmd, check=True, capture=True):
    res = subprocess.run(cmd, shell=True, capture_output=capture, text=True)
    if check and res.returncode != 0:
        print(f"Error executing: {cmd}")
        print(f"Stdout: {res.stdout}")
        print(f"Stderr: {res.stderr}")
        raise subprocess.CalledProcessError(res.returncode, cmd, res.stdout, res.stderr)
    return res

def clean_and_seed_env():
    # Re-run seed_environment.sh to wipe state and seed decoy files
    run_cmd(f"sudo bash {SCRIPTS_DIR}/seed_environment.sh", capture=True)

def terminate_sandbox_processes():
    # Force kill any lingering processes running as the sandbox user
    run_cmd(f"sudo pkill -9 -u {SANDBOX_USER}", check=False, capture=True)
    import time
    time.sleep(0.5)

def predict_detonation_strategy(binary_path, family_label):
    # Static classification of detonation strategy
    fam = family_label.lower()
    if any(x in fam for x in ["ransomware", "encoder", "erebus", "crypt"]):
        return "S3_SEEDED"  # Needs decoy files
    elif any(x in fam for x in ["mirai", "gafgyt", "bashlite", "tsunami", "xorddos", "kaiji", "bot", "miner"]):
        return "S2_FAKEC2"  # Needs INetSim C2 redirection
    
    # Fallback to check strings
    try:
        strings_res = run_cmd(f"strings {binary_path}", check=False)
        strings = strings_res.stdout.lower()
        if "connect" in strings or "socket" in strings:
            return "S2_FAKEC2"
        if "encrypt" in strings or "directory" in strings:
            return "S3_SEEDED"
    except Exception:
        pass
        
    return "S1_DEFAULT"

def execute_run(binary_id, binary_path, binary_args, is_zip, group_id, events, outdir, rep, duration, interval, family):
    os.makedirs(outdir, exist_ok=True)
    perf_file = os.path.join(outdir, f"{group_id}_perf.csv")
    log_file = os.path.join(outdir, f"{group_id}_run.log")
    
    print(f"  [Rep {rep}] Detonating {binary_id} under group {group_id}...")
    
    # 1. Cleanup and seed
    terminate_sandbox_processes()
    clean_and_seed_env()
    
    # 2. Extract binary if it's zipped malware
    exec_path = binary_path
    if is_zip:
        extract_dir = f"/home/{SANDBOX_USER}/extracted"
        run_cmd(f"sudo mkdir -p {extract_dir}")
        # Unzip as root (to read the source zip owned by iiitd), output to sandbox directory
        run_cmd(f"sudo 7z x -y -pinfected -o{extract_dir} {binary_path}")
        # Transfer ownership of extracted files to sandbox user
        run_cmd(f"sudo chown -R {SANDBOX_USER}:{SANDBOX_USER} {extract_dir}")
        # Find the largest executable file
        find_cmd = f"find {extract_dir} -type f -exec file {{}} \\; | grep -E 'ELF|executable' | cut -d: -f1 | head -1"
        res = run_cmd(find_cmd)
        exec_path = res.stdout.strip()
        if not exec_path:
            raise RuntimeError("No ELF executable found in ZIP payload")
        run_cmd(f"sudo chmod +x {exec_path}")
        
    # 3. Predict/set up strategy
    strategy = predict_detonation_strategy(exec_path, family)
    print(f"    Strategy: {strategy} | Executable: {exec_path}")
    
    # Write environment log
    with open(log_file, "w") as f:
        f.write(f"Binary ID: {binary_id}\n")
        f.write(f"Strategy: {strategy}\n")
        f.write(f"Events: {events}\n")
        f.write(f"Timestamp: {time.time()}\n")
        
    # 4. Detonate binary inside the network namespace
    # We run the command detached in its own process group (setsid) as the sandbox user
    detonate_cmd = (
        f"sudo ip netns exec {NAMESPACE} "
        f"sudo -u {SANDBOX_USER} "
        f"taskset -c {CPU_CORE} "
        f"setsid {exec_path} {binary_args} >/dev/null 2>&1 </dev/null &"
    )
    
    # 5. Profile Core 1
    # perf stat counts only Core 1 (-C 1), time-series interval mode (-I), comma-separated (-x,)
    raw_perf = os.path.join(outdir, f"{group_id}_raw.csv")
    perf_cmd = (
        f"sudo taskset -c {CPU_CORE} "
        f"perf stat -C {CPU_CORE} -x, -I {interval} -e '{events}' "
        f"-o {raw_perf} -- sleep {duration}"
    )
    
    # Launch malware
    run_cmd(detonate_cmd)
    
    # Let malware launch, then profile for the fixed duration
    time.sleep(0.5)
    run_cmd(perf_cmd)
    
    # 6. Cleanup processes immediately after profiling window
    terminate_sandbox_processes()
    
    # 7. Post-process perf output into time-series format
    if os.path.exists(raw_perf) and os.path.getsize(raw_perf) > 0:
        # Format raw perf stats: ts_ms,group,event,value
        # input: <time>,<value>,<unit>,<event>,<pct>,...
        with open(raw_perf, "r") as r, open(perf_file, "w") as w:
            w.write("ts_ms,group,event,value\n")
            for line in r:
                # Remove any null bytes that might occur due to filesystem sync races
                line = line.replace('\x00', '').strip()
                if line.startswith("#") or not line:
                    continue
                parts = line.split(",")
                if len(parts) >= 4:
                    try:
                        ts = int(float(parts[0].strip()) * 1000)
                        val = parts[1].strip()
                        ev = parts[3].strip()
                        # Skip '<not counted>' or empty values
                        if val == "<not counted>" or not ev:
                            continue
                        w.write(f"{ts},{group_id},{ev},{val}\n")
                    except ValueError:
                        # Skip malformed/corrupted lines gracefully
                        continue
        try:
            os.unlink(raw_perf)
        except Exception:
            pass
        return True
    else:
        print(f"    WARNING: No perf trace captured for {binary_id}")
        return False

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, help="Path to benign/malware manifest CSV")
    p.add_argument("--samples-dir", required=True, help="Directory containing the binaries/zips")
    p.add_argument("--groups-file", required=True, help="File containing event groups (one group per line)")
    p.add_argument("--outdir", required=True, help="Output directory for traces")
    p.add_argument("--reps", type=int, default=5, help="Number of repetitions")
    p.add_argument("--duration", type=int, default=30, help="Detonation duration in seconds")
    p.add_argument("--interval", type=int, default=10, help="HPC sampling interval in ms")
    p.add_argument("--seed", type=int, default=42, help="Random seed for execution order shuffling")
    p.add_argument("--is-malware", action="store_true", help="Flag if manifest contains malware")
    a = p.parse_args()
    
    # Load manifest
    with open(os.path.expanduser(a.manifest)) as f:
        rows = list(csv.DictReader(f))
        
    # Load event groups
    with open(os.path.expanduser(a.groups_file)) as f:
        groups = [ln.strip() for ln in f if ln.strip()]
        
    # Generate all run combinations
    runs = []
    for r in rows:
        binary_id = r.get("binary_id") or r.get("sha256")
        binary_path = os.path.join(os.path.expanduser(a.samples_dir), r.get("path") or r.get("zip_file"))
        binary_args = r.get("args", "")
        family = r.get("family", "benign")
        
        for rep in range(a.reps):
            for gi, gl in enumerate(groups):
                gid = f"g{gi:03d}"
                runs.append({
                    "binary_id": binary_id,
                    "binary_path": binary_path,
                    "binary_args": binary_args,
                    "is_zip": a.is_malware,
                    "group_id": gid,
                    "events": gl,
                    "rep": rep,
                    "family": family
                })
                
    # Shuffle runs to prevent thermal drift and temporal correlation
    random.seed(a.seed)
    random.shuffle(runs)
    
    total_runs = len(runs)
    print(f"Scheduled {total_runs} executions on the bare-metal namespace.")
    
    success_count = 0
    for idx, run in enumerate(runs, 1):
        binary_id = run["binary_id"]
        group_id = run["group_id"]
        rep = run["rep"]
        family = run["family"]
        
        run_outdir = os.path.join(os.path.expanduser(a.outdir), binary_id, f"rep_{rep}")
        perf_file = os.path.join(run_outdir, f"{group_id}_perf.csv")
        
        # Resume mode: skip if already collected
        if os.path.exists(perf_file) and os.path.getsize(perf_file) > 0:
            print(f"[{idx}/{total_runs}] Skip {binary_id} | {group_id} | rep {rep} (exists)")
            success_count += 1
            continue
            
        print(f"[{idx}/{total_runs}] Running {binary_id} | {group_id} | rep {rep}...")
        try:
            ok = execute_run(
                binary_id=binary_id,
                binary_path=run["binary_path"],
                binary_args=run["binary_args"],
                is_zip=run["is_zip"],
                group_id=group_id,
                events=run["events"],
                outdir=run_outdir,
                rep=rep,
                duration=a.duration,
                interval=a.interval,
                family=family
            )
            if ok:
                success_count += 1
        except Exception as e:
            print(f"    ERROR: Execution failed: {e}")
            
    print(f"\nCollection complete: {success_count}/{total_runs} successful runs.")

if __name__ == "__main__":
    main()
