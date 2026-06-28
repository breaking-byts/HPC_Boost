#!/usr/bin/env bash
# Wrapper to run the entire bare-metal namespace trace collection pipeline.
# Must be run with sudo:  sudo bash run_collection_pipeline.sh
set -euo pipefail

cd /home/iiitd/hpcboost_collect
export PYTHONUNBUFFERED=1

LOG="logs/collection_pipeline.log"
mkdir -p logs results/benign_ns results/malware_ns

log() {
    echo "[$(date '+%F %T')] $*" | tee -a "$LOG"
}

log "=== STARTING HPC TRACE COLLECTION PIPELINE ==="

log "Stage 1: Running Benign Collection (3 binaries, reps=5, dur=30s)"
python3 scripts/mw/detonate_and_profile.py \
    --manifest manifests/benign_manifest.csv \
    --samples-dir / \
    --groups-file event_catalog/guest_groups.txt \
    --outdir results/benign_ns \
    --reps 5 \
    --duration 30 \
    --interval 10 \
    >> "$LOG" 2>&1 || log "WARNING: Benign collection returned status $?"

log "Stage 2: Running Malware Collection (16 binaries, reps=5, dur=30s)"
python3 scripts/mw/detonate_and_profile.py \
    --manifest manifests/malware_manifest_ns.csv \
    --samples-dir payloads/malware_encrypted \
    --groups-file event_catalog/guest_groups.txt \
    --outdir results/malware_ns \
    --reps 5 \
    --duration 30 \
    --interval 10 \
    --is-malware \
    >> "$LOG" 2>&1 || log "WARNING: Malware collection returned status $?"

log "Stage 3: Building Merged Wide Labeled Dataset"
python3 scripts/mw/build_dataset.py \
    --out results/labeled_dataset_track1.csv \
    --malware results/malware_ns \
    --benign results/benign_ns \
    --manifests manifests/malware_manifest_ns.csv manifests/benign_manifest.csv \
    >> "$LOG" 2>&1 || log "WARNING: build_dataset returned status $?"

log "=== PIPELINE COMPLETE: Built results/labeled_dataset_track1.csv ==="
