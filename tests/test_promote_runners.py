import csv
import json
import subprocess
import sys
from pathlib import Path


def read_csv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def test_promote_runners_splits_runner_burst_and_reject(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "sha256,zip_file,family,arch,file_type,source,download_date\n"
        "aaa,aaa.zip,Tsunami,x86-64,elf,MalwareBazaar,2026-06-27\n"
        "bbb,bbb.zip,Mirai,x86-64,elf,MalwareBazaar,2026-06-27\n"
        "ccc,ccc.zip,Gafgyt,x86-64,elf,MalwareBazaar,2026-06-27\n"
    )
    triage = tmp_path / "triage.jsonl"
    triage.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"sha256": "aaa", "verdict": "runner", "entrypoint": "/run/a", "late_instr": 3_000_000},
                {"sha256": "bbb", "verdict": "burst", "entrypoint": "/run/b", "late_instr": 100_000},
                {"sha256": "ccc", "verdict": "silent_exit", "entrypoint": "/run/c", "late_instr": 100_000},
            ]
        )
        + "\n"
    )

    runners = tmp_path / "runners.csv"
    burst = tmp_path / "burst.csv"
    rejects = tmp_path / "rejects.csv"
    audit = tmp_path / "audit.csv"

    result = subprocess.run(
        [
            sys.executable,
            "scratch/malware_collect/promote_runners.py",
            "--triage-jsonl",
            str(triage),
            "--src-manifest",
            str(manifest),
            "--runners-out",
            str(runners),
            "--burst-out",
            str(burst),
            "--reject-out",
            str(rejects),
            "--audit-out",
            str(audit),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=True,
    )

    assert "promoted=1 burst=1 rejected=1" in result.stdout
    assert [r["sha256"] for r in read_csv(runners)] == ["aaa"]
    assert [r["sha256"] for r in read_csv(burst)] == ["bbb"]
    assert [r["sha256"] for r in read_csv(rejects)] == ["ccc"]
    audit_rows = read_csv(audit)
    assert [(r["sha256"], r["decision"]) for r in audit_rows] == [
        ("aaa", "promote"),
        ("bbb", "hold_burst"),
        ("ccc", "reject"),
    ]
