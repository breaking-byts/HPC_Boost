# HPC-Boost Data Collection Plan

## Scope and Safety Position

This plan is for defensive research: collecting hardware performance counter
traces to train and evaluate an HPC-Boost recommender under a four-PMU-event
constraint. The machine is:

- Ubuntu 24.04
- Linux kernel 6.8.0-41-generic
- Intel Core i5-7500U
- NVIDIA GeForce 940MX

The i5-7500U is a Kaby Lake mobile CPU with two physical cores and four logical
threads. It is not a hybrid P-core/E-core processor, which simplifies core
pinning compared with newer Intel CPUs. Confirm the exact SKU with `lscpu` before
relying on the counter and topology assumptions in this plan; the number of
hardware counters and the sibling layout depend on the precise model.

Do not run live malware on a daily-use bare-metal host. Use a VM sandbox for
malware pipeline development and only consider bare-metal malware execution on a
dedicated sacrificial machine that can be reimaged after each batch. VM-based
malware data is safer but may not be publication-grade PMU data unless virtual
PMU support is verified and the limitation is disclosed.

## High-Level Architecture

The collection system has five layers:

1. Host hardening and measurement stabilization.
2. Malware acquisition and quarantine.
3. Sandbox execution environment.
4. PMU collection scheduler.
5. Dataset validation and recommender-target construction.

The intended flow is:

```text
sample manifest
  -> quarantine/encrypted storage
  -> per-sample sandbox snapshot
  -> execute binary under one 4-event PMU group
  -> collect interval traces + metadata
  -> revert snapshot
  -> repeat for all event groups and repetitions
  -> validate traces
  -> compute per-binary utility targets
```

## Phase 0: Research Hygiene and Legal Controls

1. Get written approval from your advisor/lab for live malware handling.
2. Keep malware on a dedicated encrypted partition.
3. Do not sync malware directories to iCloud, Google Drive, GitHub, Dropbox, or
   university backup agents.
4. Never email samples.
5. Keep a sample ledger with source, hash, family, tags, download date, and
   handling status.
6. Use only samples you are allowed to access for research.
7. Prefer MalwareBazaar for current samples and metadata. MalwareBazaar provides
   an API and distributes downloaded samples as password-protected ZIPs with the
   password `infected`; its API currently has download limits, so use manifests
   and avoid repeated downloads. The MalwareBazaar API now requires
   authentication: register a free abuse.ch account, generate an `Auth-Key`, and
   send it with every request via the `Auth-Key` HTTP header (the unauthenticated
   `get_file` calls shown in older guides now fail). Store the key outside the
   repository, for example in an environment variable such as `MB_AUTH_KEY`, and
   never commit it. TheZoo is a live malware repository and should be treated as
   high-risk and mainly useful for small controlled tests.

Sources:

- MalwareBazaar API: https://bazaar.abuse.ch/api/
- MalwareBazaar FAQ: https://bazaar.abuse.ch/faq/
- theZoo GitHub repository: https://github.com/ytisf/theZoo
- Linux perf tutorial: https://perfwiki.github.io/main/tutorial/
- perf stat man page: https://man7.org/linux/man-pages/man1/perf-stat.1.html

## Phase 1: BIOS and Firmware Setup

Enter BIOS/UEFI setup. Exact option names vary by laptop vendor.

Required:

1. Enable Intel Virtualization Technology / VT-x.
2. Enable VT-d if available.
3. Disable Secure Boot only if it blocks required kernel tools or unsigned
   modules. Otherwise leave it enabled.
4. Set SATA/NVMe mode normally; do not change storage mode after OS install.

Strongly recommended for measurement stability:

1. Disable Intel Turbo Boost if BIOS exposes it.
2. Disable Intel SpeedStep / Speed Shift if BIOS exposes it.
3. Disable C-states / deep sleep states if BIOS exposes it.
4. Disable Hyper-Threading if BIOS exposes it. Note the consequence for Phase 2:
   if you disable it, this CPU presents only two logical CPUs (one per physical
   core) and the sibling-pinning steps in Phase 2 no longer apply - just pin
   measurement to one core and leave the other for the OS. If you leave
   Hyper-Threading enabled, you must follow the sibling-isolation steps in Phase
   2 so the measured thread does not share a core with active work. Pick one
   strategy and keep it identical across the entire dataset.
5. Disable Wake-on-LAN and network boot.
6. Disable the NVIDIA/discrete GPU if BIOS exposes a reliable iGPU-only mode.
   This is optional; the GPU is not part of PMU measurement, but it can affect
   thermals and power behavior.

If BIOS does not expose these controls, use OS-level controls in Phase 2.

## Phase 2: Host OS Stabilization

Install required packages:

```bash
sudo apt update
sudo apt install -y \
  linux-tools-$(uname -r) linux-tools-generic linux-tools-common \
  cpufrequtils msr-tools \
  qemu-kvm libvirt-daemon-system libvirt-clients virt-manager virtinst \
  python3 python3-venv python3-pip jq unzip p7zip-full \
  git curl rsync tmux htop sysstat
```

Add your user to libvirt groups:

```bash
sudo usermod -aG libvirt,kvm "$USER"
newgrp libvirt
```

Set permissive perf access for collection:

```bash
echo 'kernel.perf_event_paranoid = -1' | sudo tee /etc/sysctl.d/99-hpcboost-perf.conf
echo 'kernel.kptr_restrict = 0' | sudo tee -a /etc/sysctl.d/99-hpcboost-perf.conf
echo 'kernel.nmi_watchdog = 0' | sudo tee -a /etc/sysctl.d/99-hpcboost-perf.conf
sudo sysctl --system
```

Set CPU governor to performance:

```bash
sudo cpupower frequency-set -g performance
```

Disable turbo at OS level if supported:

```bash
if [ -f /sys/devices/system/cpu/intel_pstate/no_turbo ]; then
  echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo
fi
```

Stop noisy background services during measurement windows:

```bash
sudo systemctl stop apt-daily.timer apt-daily-upgrade.timer || true
sudo systemctl stop unattended-upgrades.service || true
sudo systemctl stop irqbalance.service || true
```

Check CPU topology:

```bash
lscpu
lscpu -e=CPU,CORE,SOCKET,NODE,ONLINE,MAXMHZ,MINMHZ
cat /sys/devices/system/cpu/cpu*/topology/thread_siblings_list
```

Choose one logical CPU for the measured program and keep its sibling idle. On
many two-core/four-thread Intel laptops, sibling pairs look like `0,2` and
`1,3`; verify with the command above. If CPU `1` and `3` are siblings, use CPU
`1` for measurement and avoid scheduling anything on CPU `3`. This sibling step
only applies if Hyper-Threading is enabled; if you disabled it in Phase 1 there
are no siblings to idle, so simply pin to one core (for example CPU `1`) and
reserve it. Whichever topology you choose, record it in the trace metadata so
every run in the dataset uses the same configuration.

Optional but useful: boot-time CPU isolation. Edit `/etc/default/grub`:

```bash
sudo cp /etc/default/grub /etc/default/grub.bak
sudo nano /etc/default/grub
```

Append to `GRUB_CMDLINE_LINUX_DEFAULT`:

```text
isolcpus=1 nohz_full=1 rcu_nocbs=1
```

Then:

```bash
sudo update-grub
sudo reboot
```

Use CPU isolation only after confirming which CPU you want to reserve.

## Phase 3: Project Directory Layout

Create a strict directory layout:

```bash
mkdir -p ~/hpcboost_collect/{manifests,samples_encrypted,samples_quarantine,benign,vm,results,logs,scripts,event_catalog,tmp}
chmod 700 ~/hpcboost_collect
chmod 700 ~/hpcboost_collect/samples_encrypted ~/hpcboost_collect/samples_quarantine
```

Recommended layout:

```text
~/hpcboost_collect/
  manifests/
    malwarebazaar_manifest.csv
    benign_manifest.csv
    run_manifest.csv
  samples_encrypted/
    malwarebazaar/
  samples_quarantine/
    extracted_only_inside_sandbox/
  benign/
    binaries/
    inputs/
  vm/
    images/
    snapshots/
    iso_payloads/
  event_catalog/
    perf_list.txt
    supported_events.csv
    event_groups.csv
  results/
    raw/
    parsed/
    validation/
  logs/
  scripts/
```

## Phase 4: Event Catalog Construction

Dump available events:

```bash
perf list > ~/hpcboost_collect/event_catalog/perf_list.txt
perf list hw cache pmu > ~/hpcboost_collect/event_catalog/perf_list_core.txt
```

Start with generic events and add PMU-specific events only after validation.

The RaDaR oracle's discriminative signal was dominated by memory-hierarchy and
pipeline-stall events (L3/LLC latency and misses, L2 requests, load-hit
behavior, resource stalls). Prioritize raw memory-hierarchy and stall events for
this CPU early, alongside the generic events. Be aware that these generic perf
aliases will not map one-to-one onto RaDaR's raw micro-architectural event names,
so results on this machine are not directly comparable to the RaDaR numbers. That
is expected for a new machine and event universe, but disclose it.

Initial generic event candidates:

```text
cycles
instructions
branches
branch-misses
cache-references
cache-misses
L1-dcache-loads
L1-dcache-load-misses
L1-dcache-stores
LLC-loads
LLC-load-misses
dTLB-loads
dTLB-load-misses
iTLB-loads
iTLB-load-misses
```

Validate each event:

```bash
cat > ~/hpcboost_collect/scripts/validate_events.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
EVENT_FILE="$1"
OUT="$2"
echo "event,status" > "$OUT"
while read -r ev; do
  [ -z "$ev" ] && continue
  if perf stat -x, -e "$ev" -- sleep 0.1 >/tmp/perf_test.out 2>/tmp/perf_test.err; then
    echo "$ev,ok" >> "$OUT"
  else
    echo "$ev,fail" >> "$OUT"
  fi
done < "$EVENT_FILE"
EOF
chmod +x ~/hpcboost_collect/scripts/validate_events.sh
```

Create a candidate event list:

```bash
cat > ~/hpcboost_collect/event_catalog/candidate_events.txt <<'EOF'
cycles
instructions
branches
branch-misses
cache-references
cache-misses
L1-dcache-loads
L1-dcache-load-misses
L1-dcache-stores
LLC-loads
LLC-load-misses
dTLB-loads
dTLB-load-misses
iTLB-loads
iTLB-load-misses
EOF
```

Validate:

```bash
~/hpcboost_collect/scripts/validate_events.sh \
  ~/hpcboost_collect/event_catalog/candidate_events.txt \
  ~/hpcboost_collect/event_catalog/supported_events.csv
```

Create groups of at most four events. Two distinct limits are at play here, and
they are not the same number. The four-event budget is HPC-Boost's *deployment*
constraint - the recommender is designed to pick four events to monitor at
runtime - and it is what the event groups must reflect. Separately, the hardware
has a fixed number of physical counters; Linux perf multiplexes (time-shares)
events when a group requests more than the hardware can count at once, which
corrupts interval traces. On this Kaby Lake CPU there are typically four
general-purpose counters plus three fixed-function counters, and
`cycles`/`instructions` use the fixed counters. A group such as
`{cycles, instructions, branches, branch-misses}` therefore consumes only two
general-purpose counters and will not multiplex; with Hyper-Threading disabled
some Intel parts expose more general-purpose counters still. Do not assume the
four-event budget equals the multiplexing threshold: design groups around the
deployment constraint, and verify empirically (below) that each group counts
without multiplexing.

Use perf event groups to make scheduling failure visible:

```bash
perf stat -e '{cycles,instructions,branches,branch-misses}' -- sleep 1
```

Reject a group if perf reports:

- `<not counted>`
- high multiplex scaling
- unsupported event
- inconsistent repeated measurements

## Phase 5: Malware Acquisition

### 5.1 MalwareBazaar

Use MalwareBazaar metadata first. Do not download binaries until the sandbox is
ready.

Example manifest fields:

```text
sha256,source,signature,tags,file_type,first_seen,download_status,notes
```

Download only into encrypted/quarantine storage. MalwareBazaar samples are
password-protected ZIPs using password `infected`.

Example API pattern:

```bash
curl -X POST https://mb-api.abuse.ch/api/v1/ \
  -H "Auth-Key: $MB_AUTH_KEY" \
  -d "query=get_file&sha256_hash=<SHA256>" \
  --output ~/hpcboost_collect/samples_encrypted/malwarebazaar/<SHA256>.zip
```

The `Auth-Key` header is mandatory: abuse.ch enabled authentication on the
MalwareBazaar API, so `get_file` requests without a valid key now fail. The data
endpoint is `https://mb-api.abuse.ch/api/v1/`. Export your key once per shell,
for example `export MB_AUTH_KEY=...`, and keep it out of the repository.

Do not extract on the host. Extraction should happen inside the sandbox or
inside an offline analysis VM.

### 5.2 theZoo

Use theZoo only for small controlled tests because it is a live malware
repository with older samples and inconsistent metadata. Clone it only inside a
quarantine VM or an encrypted research partition:

```bash
cd ~/hpcboost_collect/samples_encrypted
git clone https://github.com/ytisf/theZoo.git theZoo
```

Do not execute anything from theZoo on the host.

## Phase 6: Benign Dataset Collection

Collect benign binaries first. This validates the PMU pipeline without live
malware.

Recommended benign sources:

1. Coreutils: `ls`, `cat`, `sort`, `sha256sum`, `grep`, `find`.
2. Compression: `gzip`, `bzip2`, `xz`, `zstd`.
3. Crypto/hash: `openssl speed`, `sha256sum`, `gpg` if available.
4. Build tools: `gcc`, `make`, `ld`.
5. Scripting: Python workloads.
6. Phoronix/MiBench/SPEC if available.

Create deterministic inputs:

```bash
mkdir -p ~/hpcboost_collect/benign/inputs
dd if=/dev/urandom of=~/hpcboost_collect/benign/inputs/random_100mb.bin bs=1M count=100
seq 1 1000000 > ~/hpcboost_collect/benign/inputs/nums.txt
```

Create `benign_manifest.csv`:

```text
binary_id,path,args,input_id,category,timeout_sec
coreutils_sort,/usr/bin/sort,"~/hpcboost_collect/benign/inputs/nums.txt",nums,coreutils,30
gzip_random,/usr/bin/gzip,"-c ~/hpcboost_collect/benign/inputs/random_100mb.bin",random_100mb,compression,30
openssl_sha256,/usr/bin/openssl,"dgst -sha256 ~/hpcboost_collect/benign/inputs/random_100mb.bin",random_100mb,crypto,30
```

## Phase 7: Sandbox Strategy

### Option A: KVM VM sandbox, safety-first

Use this for live malware pipeline development.

Pros:

- Snapshot/revert per sample.
- Malware containment is much safer.
- Automation is easier.

Cons:

- PMU readings may be virtualized or reflect QEMU/guest overhead.
- Host-side perf on `qemu-system-*` is not equivalent to process-level malware
  PMU behavior.
- Publication claims must disclose VM-based measurement.

### Option B: Bare-metal sacrificial machine, fidelity-first

Use only if you have a dedicated machine that can be reimaged.

Pros:

- Best PMU fidelity.
- Closest to runtime deployment.

Cons:

- Higher containment risk.
- Requires network isolation and restore automation.

For your current laptop, use Option A for live malware and Option B only for
benign/pilot workloads or if the laptop is dedicated and reimageable.

## Phase 8: KVM Sandbox Setup

Install an Ubuntu guest VM. Prefer a minimal Ubuntu 24.04 guest.

Create a base image:

```bash
mkdir -p ~/hpcboost_collect/vm/images
qemu-img create -f qcow2 ~/hpcboost_collect/vm/images/malware_base.qcow2 40G
```

Install with virt-install, initially with NAT only for package installation:

```bash
virt-install \
  --name hpcboost-malware-base \
  --memory 4096 \
  --vcpus 1 \
  --cpu host-passthrough \
  --disk path=$HOME/hpcboost_collect/vm/images/malware_base.qcow2,format=qcow2 \
  --cdrom /path/to/ubuntu-24.04.iso \
  --os-variant ubuntu24.04 \
  --network network=default \
  --graphics spice
```

Inside the guest:

```bash
sudo apt update
sudo apt install -y linux-tools-$(uname -r) linux-tools-generic python3 jq unzip p7zip-full
echo 'kernel.perf_event_paranoid = -1' | sudo tee /etc/sysctl.d/99-perf.conf
sudo sysctl --system
```

Check guest PMU availability:

```bash
grep -E 'arch_perfmon|pmu' /proc/cpuinfo | head
perf stat -e cycles,instructions -- sleep 1
```

If this fails, the VM cannot collect guest-level PMU traces. In that case, VM
malware traces are useful only for pipeline testing, not final PMU claims.

After installing tools, shut down the guest and remove network:

```bash
virsh shutdown hpcboost-malware-base
virsh domiflist hpcboost-malware-base
```

Use `virt-manager` or `virsh edit hpcboost-malware-base` to remove the NIC for
final malware runs, or attach it only to an isolated libvirt network with no
route to the internet.

Create an isolated libvirt network:

```bash
cat > /tmp/hpcboost-isolated.xml <<'EOF'
<network>
  <name>hpcboost-isolated</name>
  <bridge name='virbr-hpcboost' stp='on' delay='0'/>
</network>
EOF
virsh net-define /tmp/hpcboost-isolated.xml
virsh net-start hpcboost-isolated
virsh net-autostart hpcboost-isolated
```

For maximum safety, run with no NIC at all unless the experiment explicitly
requires controlled network behavior.

Create a clean snapshot:

```bash
virsh snapshot-create-as hpcboost-malware-base clean-base \
  "Clean base before malware execution"
```

Disable clipboard, shared folders, drag-and-drop, and guest-to-host integration
features in the VM UI. Do not mount host directories read-write inside the
guest.

## Phase 9: Safe Sample Transfer Into the VM

Preferred method: per-run ISO payload.

For each sample:

1. Create a temporary directory with:
   - encrypted sample ZIP,
   - runner script,
   - event group file,
   - metadata JSON.
2. Build a read-only ISO.
3. Attach ISO to VM.
4. Boot VM with no network.
5. Inside VM, copy payload to a temp directory and extract.
6. Run collection.
7. Shut down VM.
8. Extract result disk or use `virt-copy-out` while VM is powered off.
9. Revert snapshot.

Create ISO:

```bash
mkdir -p ~/hpcboost_collect/tmp/payload
cp sample.zip runner.sh event_groups.csv metadata.json ~/hpcboost_collect/tmp/payload/
xorriso -as mkisofs -o ~/hpcboost_collect/vm/iso_payloads/payload.iso \
  ~/hpcboost_collect/tmp/payload
```

Attach:

```bash
virsh attach-disk hpcboost-malware-base \
  ~/hpcboost_collect/vm/iso_payloads/payload.iso sdb \
  --type cdrom --mode readonly
```

Detach after shutdown:

```bash
virsh detach-disk hpcboost-malware-base sdb
```

## Phase 10: PMU Collection Command

Use `perf stat` interval mode for time-series-like traces:

```bash
perf stat -o perf.csv -x, -I 10 -e '{cycles,instructions,branches,branch-misses}' \
  -- timeout 30s taskset -c 1 /path/to/binary arg1 arg2 \
  1> program.stdout \
  2> program.stderr
```

Notes:

- `-I 10` records every 10 ms. Increase to 50 or 100 ms if overhead is too
  high.
- `-x,` gives CSV-like output.
- Braces request an event group, making scheduling failures more visible.
- Use `timeout` to avoid hanging samples.
- Use `taskset` for CPU pinning. Pin to the isolated CPU chosen in Phase 2 (CPU
  `1` in these examples) and keep it identical across the entire dataset; do not
  mix `-c 0` and `-c 1` between runs.
- Keep perf's event run/enabled ratio (the trailing percentage column in `-x,`
  output) in the parsed data. Phase 14 validation needs it to detect
  multiplexing; do not discard it during parsing.
- Store stdout/stderr separately.
- Record exit code and timeout status.

For benign host-side collection:

```bash
sudo chrt -f 80 taskset -c 1 \
  perf stat -o run.perf.csv -x, -I 10 -e '{cycles,instructions,branches,branch-misses}' \
  -- timeout 30s /usr/bin/sort ~/hpcboost_collect/benign/inputs/nums.txt \
  1> run.stdout \
  2> run.stderr
```

For VM guest-side collection, run the same pattern inside the guest.

## Phase 11: Event Group Scheduling

If you have 55 supported events and can collect four at a time, create:

```text
group_id,event_1,event_2,event_3,event_4
g000,cycles,instructions,branches,branch-misses
g001,cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses
...
```

Number of executions per binary:

```text
num_groups = ceil(num_events / 4)
num_executions = num_groups x repetitions x conditions
```

For 55 events, 14 groups, 10 repetitions, and 2 conditions:

```text
14 x 10 x 2 = 280 executions per binary
```

This is expensive. Start with 15-20 events and 5 repetitions for the pilot.
Scale after validation.

Randomize execution order:

```text
binary_id,condition,group_id,rep,random_order_seed
```

Randomization prevents thermal drift or time-of-day effects from correlating
with event groups.

## Phase 12: Runner Script Skeleton

Create `run_one_group.sh`:

```bash
cat > ~/hpcboost_collect/scripts/run_one_group.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

BINARY_ID="$1"
BINARY_PATH="$2"
ARGS="$3"
GROUP_ID="$4"
EVENTS="$5"
OUTDIR="$6"
CPU="${7:-1}"
TIMEOUT_SEC="${8:-30}"
INTERVAL_MS="${9:-10}"

mkdir -p "$OUTDIR"

META="$OUTDIR/metadata.json"
PERF="$OUTDIR/perf.csv"
STDOUT="$OUTDIR/stdout.txt"
STDERR="$OUTDIR/stderr.txt"

START_NS=$(date +%s%N)
set +e
sudo chrt -f 80 taskset -c "$CPU" \
  perf stat -o "$PERF" -x, -I "$INTERVAL_MS" -e "{$EVENTS}" \
  -- timeout "${TIMEOUT_SEC}s" bash -lc "\"$BINARY_PATH\" $ARGS" \
  >"$STDOUT" 2>"$STDERR"
STATUS=$?
set -e
END_NS=$(date +%s%N)

cat > "$META" <<JSON
{
  "binary_id": "$BINARY_ID",
  "binary_path": "$BINARY_PATH",
  "args": "$ARGS",
  "group_id": "$GROUP_ID",
  "events": "$EVENTS",
  "cpu": "$CPU",
  "timeout_sec": "$TIMEOUT_SEC",
  "interval_ms": "$INTERVAL_MS",
  "exit_status": "$STATUS",
  "start_ns": "$START_NS",
  "end_ns": "$END_NS",
  "kernel": "$(uname -r)",
  "hostname": "$(hostname)"
}
JSON

exit 0
EOF
chmod +x ~/hpcboost_collect/scripts/run_one_group.sh
```

Do not use this script for live malware on the host. Use it inside the VM or on
a sacrificial machine.

## Phase 13: Orchestrator Design

Write a Python orchestrator with these functions:

1. Load `run_manifest.csv`.
2. Validate binary/sample existence.
3. Randomize run order.
4. For benign host runs, call `run_one_group.sh`.
5. For malware VM runs:
   - revert snapshot,
   - create payload ISO,
   - attach ISO,
   - boot VM,
   - run guest runner,
   - shut down or forcibly destroy after timeout,
   - extract results,
   - revert snapshot,
   - detach ISO.
6. Record status in SQLite or CSV.
7. Resume failed/incomplete runs.

Use a state machine:

```text
PENDING -> PREPARED -> RUNNING -> COLLECTED -> VALIDATED
                         \-> TIMEOUT
                         \-> FAILED
                         \-> QUARANTINED
```

## Phase 14: Validation After Each Batch

For each run, validate:

1. `perf.csv` exists and is non-empty.
2. All four requested events appear.
3. No `<not counted>` rows.
4. No high multiplex scaling.
5. Runtime is within expected bounds.
6. Exit status is recorded.
7. Trace has enough intervals.
8. Repeated runs are not constant zeros.
9. Malware VM was reverted after the run.

Reject or quarantine traces that fail validation.

## Phase 15: Dataset Schema

Use one metadata JSON per trace:

```json
{
  "trace_id": "sha256_or_binaryid_condition_group_rep",
  "binary_id": "sample sha256 or benign id",
  "class_label": "benign or malware",
  "family": "malware family if known",
  "source": "MalwareBazaar/theZoo/benign_suite",
  "event_group": ["cycles", "instructions", "branches", "branch-misses"],
  "cpu": 1,
  "interval_ms": 10,
  "timeout_sec": 30,
  "environment": "host_baremetal or kvm_guest",
  "vm_snapshot": "clean-base",
  "kernel": "6.8.0-41-generic",
  "governor": "performance",
  "turbo_disabled": true,
  "network": "none",
  "exit_status": 0,
  "notes": ""
}
```

Store parsed traces in columnar form:

```text
trace_id,timestamp_ms,event_name,value
```

or wide form:

```text
trace_id,timestamp_ms,cycles,instructions,branches,branch-misses
```

Wide form is easier for model training; long form is easier for validation.

## Phase 16: Pilot Plan

Do not begin with hundreds of malware samples. Start with:

```text
40-50 benign binaries (maximize program/source diversity)
30 malware samples
15-20 PMU events
5 repetitions
10-30 second timeout
10-50 ms interval
```

Bias the pilot toward benign diversity. The detection-aware oracle showed that
the entire conditional-selection opportunity is benign false-positive reduction:
every malware sample had at least one correct candidate subset, while all
unrecoverable cases were benign. False-positive rate is therefore the metric that
decides go/no-go, and an FPR estimated from only ~30 benign binaries is
high-variance. Prefer 40-50 benign programs drawn from many distinct source
packages so the benign side is the rich side of the dataset, inverting RaDaR's
malware-heavy imbalance.

Pilot goals:

1. Confirm PMU traces are stable.
2. Confirm event groups do not multiplex.
3. Confirm VM PMU works if using guest collection.
4. Confirm snapshot revert is reliable.
5. Estimate runtime per sample.
6. Compute initial per-binary utility targets.
7. Test whether static features predict capped-oracle choices.
8. Most important: measure whether per-binary event-subset utility is actually
   differentiated and stable across repeated runs. The RaDaR candidate pools were
   nearly flat under the training objective (many near-equivalent subsets, inner
   AUCPR spread under ~0.015). If per-binary utility is also flat, no recommender
   can learn an exact subset, and the target must change (see Phase 17).

Only scale after the pilot passes.

## Phase 17: Recommender Target Construction

Do not train on one hard top-4 label from a single run.

For each binary:

1. Collect repeated traces.
2. Evaluate candidate event subsets.
3. Compute utility per subset:

```text
utility = alpha * TPR - beta * FPR + eta * AUCPR - gamma * run_variance
```

4. Convert utilities into a soft target distribution.
5. Train recommender with listwise or KL-divergence loss.
6. Evaluate regret:

```text
regret = oracle_utility - recommended_subset_utility
```

Use capped candidate pools first:

```text
P = 25 or P = 50
```

The full-pool oracle is useful as a ceiling but too optimistic as the primary
training target.

If the pilot finds per-binary utility is flat (top subsets statistically
indistinguishable across repeated runs), do not force the recommender to imitate
one hard top-4 subset. Switch the target to a coarser, learnable objective:
predict an FPR-safe candidate band or class - the set of subsets that keep FPR
low at high TPR - rather than an exact ranked subset. Report regret against both
the capped oracle (P = 25 or P = 50) and the full-pool ceiling.

## Phase 18: Publication-Grade Evaluation Splits

Use grouped splits:

1. Benign: group by program name/source package.
2. Malware: group by malware family and, where possible, campaign/source.
3. Repeated runs of the same binary must never cross train/test boundaries.
4. Collection sessions should be split if possible.

Avoid plain StratifiedKFold over traces. It can leak program identity across
folds.

Recommended splits:

1. Held-out binaries.
2. Held-out malware families.
3. Held-out collection sessions.
4. If possible, held-out machine or reboot-day.

## Phase 19: Metrics

Report:

1. F1.
2. Precision.
3. Recall.
4. False positive rate.
5. Balanced accuracy.
6. AUCPR.
7. TPR at fixed FPR.
8. Recommender top-k regret.
9. Event-subset stability across repeated runs.
10. Collection overhead and inference overhead.

For malware detection, do not report only F1. The RaDaR oracle showed that F1
can hide severe benign false-positive rates under malware-heavy imbalance.

## Phase 20: Go/No-Go Gates

Proceed from pilot to full collection only if:

1. PMU group validation passes with no multiplexing.
2. Repeated-run variance is acceptable.
3. Snapshot revert is reliable.
4. Static feature extraction works for most binaries.
5. Capped per-binary oracle beats best global subset by at least 3-5 F1 points
   or a comparable balanced-accuracy/FPR gain.
6. A simple recommender beats global routing on held-out binaries.
7. False-positive rate at a fixed high TPR (for example TPR >= 0.95) is reported,
   and the per-binary oracle reduces FPR at that operating point. Because the
   dataset is malware-heavy and F1 hides benign false positives, treat
   FPR-at-fixed-TPR as a first-class gate in its own right, not merely as an
   alternative to the F1 criterion in gate 5.

Stop and redesign if:

1. VM PMU is unavailable or too noisy.
2. Per-binary optimal subsets are unstable across repeated runs.
3. Gains vanish under grouped splits.
4. The recommender cannot beat global selection.
5. Data collection creates safety risks that cannot be controlled.

## Immediate Next Commands

Run these first:

```bash
lscpu
lscpu -e=CPU,CORE,SOCKET,NODE,ONLINE,MAXMHZ,MINMHZ
cat /sys/devices/system/cpu/cpu*/topology/thread_siblings_list
perf stat -e cycles,instructions,branches,branch-misses -- sleep 1
sudo cpupower frequency-info
```

Then install and configure tools:

```bash
sudo apt update
sudo apt install -y linux-tools-$(uname -r) linux-tools-generic linux-tools-common cpufrequtils qemu-kvm libvirt-daemon-system libvirt-clients virt-manager virtinst jq unzip p7zip-full xorriso tmux htop
sudo usermod -aG libvirt,kvm "$USER"
```

After reboot/re-login:

```bash
sudo cpupower frequency-set -g performance
echo 'kernel.perf_event_paranoid = -1' | sudo tee /etc/sysctl.d/99-hpcboost-perf.conf
echo 'kernel.nmi_watchdog = 0' | sudo tee -a /etc/sysctl.d/99-hpcboost-perf.conf
sudo sysctl --system
```

Then validate a four-event group:

```bash
perf stat -x, -I 100 -e '{cycles,instructions,branches,branch-misses}' -- sleep 5
```

Do not download or execute malware until:

1. VM sandbox is created.
2. Network isolation is verified.
3. Snapshot revert is tested.
4. Benign collection pipeline works end to end.
