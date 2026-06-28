#!/usr/bin/env bash
set -euo pipefail

# Create metadata directory
mkdir -p /tmp/cidata

# Generate user-data
cat > /tmp/cidata/user-data <<'EOF'
#cloud-config
users:
  - name: ubuntu
    sudo: ALL=(ALL) NOPASSWD:ALL
    groups: users, admin, sudo
    shell: /bin/bash
    ssh_authorized_keys:
      - ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOCSpTK4NVO0dqIx7eLszrWmHQpVyyB8xfzQW7QXKElg
      - ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIGoldUTdo0cpEy6qi1AIHX3GyzglJWsYyJyP+nEAtVoY iiitd@iiitd-ThinkPad-E470

chpasswd:
  list: |
    ubuntu:ubuntu
  expire: False

ssh_pwauth: True

packages:
  - linux-tools-generic
  - linux-tools-common
  - python3
  - jq
  - unzip
  - p7zip-full

runcmd:
  - [ sh, -c, "echo 'kernel.perf_event_paranoid = -1' > /etc/sysctl.d/99-perf.conf" ]
  - [ sysctl, --system ]
EOF

# Generate meta-data
cat > /tmp/cidata/meta-data <<'EOF'
instance-id: hpcboost-malware-base
local-hostname: hpcboost-malware-base
EOF

# Build ISO
echo "Building cloud-init ISO..."
xorriso -as mkisofs -o "$HOME/hpcboost_collect/vm/iso_payloads/cidata.iso" -V cidata -J -r /tmp/cidata

# Prepare guest image
echo "Preparing VM disk image..."
cp "$HOME/hpcboost_collect/vm/images/ubuntu-24.04-minimal-cloudimg-amd64.img" "$HOME/hpcboost_collect/vm/images/hpcboost-malware-base.qcow2"
qemu-img resize "$HOME/hpcboost_collect/vm/images/hpcboost-malware-base.qcow2" 20G

# Clean up
rm -rf /tmp/cidata

# Destroy previous VM if exists
if virsh -c qemu:///system dominfo hpcboost-malware-base &>/dev/null; then
    echo "Destroying existing hpcboost-malware-base VM..."
    virsh -c qemu:///system destroy hpcboost-malware-base || true
    virsh -c qemu:///system undefine hpcboost-malware-base --remove-all-storage || true
fi

# Define and start VM
echo "Starting VM installation..."
virt-install \
  --connect qemu:///system \
  --name hpcboost-malware-base \
  --memory 2048 \
  --vcpus 1 \
  --cpu host-passthrough \
  --disk path="$HOME/hpcboost_collect/vm/images/hpcboost-malware-base.qcow2",format=qcow2 \
  --disk path="$HOME/hpcboost_collect/vm/iso_payloads/cidata.iso",device=cdrom \
  --os-variant ubuntu22.04 \
  --network network=default \
  --graphics none \
  --import \
  --noautoconsole

echo "VM creation command sent. Checking VM status..."
virsh -c qemu:///system list --all
