#!/usr/bin/env python3
import os
import sys
import json
import csv
import glob

def parse_perf_csv(perf_path, expected_events):
    if not os.path.exists(perf_path) or os.path.getsize(perf_path) == 0:
        return None, "empty_file"
        
    intervals = {}
    total_rows = 0
    not_counted_count = 0
    multiplexed_count = 0
    
    with open(perf_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split(',')
            if len(parts) < 5:
                continue
                
            total_rows += 1
            timestamp_str = parts[0].strip()
            val_str = parts[1].strip()
            event = parts[3].strip()
            
            run_pct = 100.0
            if len(parts) >= 6:
                try:
                    run_pct = float(parts[5].strip())
                except ValueError:
                    pass
            
            if val_str == "<not counted>":
                not_counted_count += 1
                val = 0
            else:
                try:
                    val = int(val_str)
                except ValueError:
                    try:
                        val = float(val_str)
                    except ValueError:
                        val = 0
            
            if run_pct < 100.0:
                multiplexed_count += 1
                
            try:
                timestamp_ms = int(float(timestamp_str) * 1000)
            except ValueError:
                continue
                
            if timestamp_ms not in intervals:
                intervals[timestamp_ms] = {}
            intervals[timestamp_ms][event] = val

    if total_rows == 0:
        return None, "no_data_rows"
        
    if len(intervals) < 5:
        return None, "too_few_intervals"
        
    found_events = set()
    for ts in intervals:
        for ev in intervals[ts]:
            found_events.add(ev)
            
    missing_events = [ev for ev in expected_events if ev not in found_events]
    if missing_events:
        return None, f"missing_events:{','.join(missing_events)}"
        
    not_counted_ratio = not_counted_count / total_rows
    if not_counted_ratio > 0.5:
        return None, "high_not_counted_ratio"
        
    multiplex_ratio = multiplexed_count / total_rows
    if multiplex_ratio > 0.1:
        return None, "multiplexed_events"
        
    all_zeros = True
    for ts in intervals:
        for ev in intervals[ts]:
            if intervals[ts][ev] != 0:
                all_zeros = False
                break
        if not all_zeros:
            break
            
    if all_zeros:
        return None, "all_constant_zeros"
        
    return intervals, "ok"

def main():
    import argparse
    parser = argparse.ArgumentParser(description="HPC-Boost Parse and Validate Traces")
    parser.add_argument("--raw-dir", required=True, help="Directory containing raw traces")
    parser.add_argument("--parsed-dir", required=True, help="Directory to save parsed CSVs")
    parser.add_argument("--report-path", required=True, help="Path to write validation report JSON")
    
    args = parser.parse_args()
    
    os.makedirs(args.parsed_dir, exist_ok=True)
    
    metadata_files = glob.glob(os.path.join(args.raw_dir, "**", "metadata.json"), recursive=True)
    
    report = {
        "summary": {
            "total_runs": len(metadata_files),
            "ok": 0,
            "failed": 0,
            "failures_by_reason": {}
        },
        "runs": []
    }
    
    parsed_data_by_binary = {}
    
    for meta_path in metadata_files:
        run_dir = os.path.dirname(meta_path)
        perf_path = os.path.join(run_dir, "perf.csv")
        
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
        except Exception as e:
            report["summary"]["failed"] += 1
            report["summary"]["failures_by_reason"]["corrupt_metadata"] = report["summary"]["failures_by_reason"].get("corrupt_metadata", 0) + 1
            report["runs"].append({
                "dir": run_dir,
                "status": "failed",
                "reason": "corrupt_metadata"
            })
            continue
            
        binary_id = meta["binary_id"]
        group_id = meta["group_id"]
        rep = meta["rep"] if "rep" in meta else run_dir.split(os.sep)[-1].replace("rep_", "")
        expected_events = meta["events"].split(",")
        
        intervals, status = parse_perf_csv(perf_path, expected_events)
        
        run_info = {
            "binary_id": binary_id,
            "group_id": group_id,
            "rep": rep,
            "dir": run_dir,
            "status": status
        }
        
        if status == "ok":
            report["summary"]["ok"] += 1
            trace_id = f"{binary_id}_{group_id}_rep{rep}"
            
            if binary_id not in parsed_data_by_binary:
                parsed_data_by_binary[binary_id] = []
                
            for ts_ms, events_val in sorted(intervals.items()):
                row = {
                    "trace_id": trace_id,
                    "timestamp_ms": ts_ms
                }
                for ev in expected_events:
                    row[ev] = events_val.get(ev, 0)
                parsed_data_by_binary[binary_id].append(row)
        else:
            report["summary"]["failed"] += 1
            report["summary"]["failures_by_reason"][status] = report["summary"]["failures_by_reason"].get(status, 0) + 1
            run_info["reason"] = status
            
        report["runs"].append(run_info)
        
    for binary_id, rows in parsed_data_by_binary.items():
        if not rows:
            continue
            
        all_cols = set()
        for r in rows:
            all_cols.update(r.keys())
            
        ordered_cols = ["trace_id", "timestamp_ms"]
        event_cols = sorted(list(all_cols - {"trace_id", "timestamp_ms"}))
        ordered_cols.extend(event_cols)
        
        csv_path = os.path.join(args.parsed_dir, f"{binary_id}_wide.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=ordered_cols)
            writer.writeheader()
            for r in rows:
                row_to_write = {col: r.get(col, 0) for col in ordered_cols}
                writer.writerow(row_to_write)
                
    with open(args.report_path, 'w') as f:
        json.dump(report, f, indent=2)
        
    print(f"Validation complete. OK: {report['summary']['ok']}, Failed: {report['summary']['failed']}")
    if report["summary"]["failed"] > 0:
        print("Failures by reason:")
        for reason, count in report["summary"]["failures_by_reason"].items():
            print(f"  {reason}: {count}")

if __name__ == "__main__":
    main()
