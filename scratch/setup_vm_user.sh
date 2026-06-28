#!/usr/bin/env bash
set -euo pipefail

# Destroy previous VM if exists in user session (move to beginning to release file locks)
if virsh -c qemu:///session dominfo hpcboost-malware-base &>/dev/null; then
    echo "Destroying existing hpcboost-malware-base VM in user session..."
    virsh -c qemu:///session destroy hpcboost-malware-base || true
    virsh -c qemu:///session undefine hpcboost-malware-base --nvram || true
fi

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
xorriso -as mkisofs -o "$HOME/hpcboost_collect/vm/iso_payloads/cidata.iso" -V CIDATA -J -r /tmp/cidata

# Prepare guest image
echo "Preparing VM disk image..."
cp "$HOME/hpcboost_collect/vm/images/ubuntu-24.04-minimal-cloudimg-amd64.img" "$HOME/hpcboost_collect/vm/images/hpcboost-malware-base.qcow2"
qemu-img resize "$HOME/hpcboost_collect/vm/images/hpcboost-malware-base.qcow2" 20G

# Clean up
rm -rf /tmp/cidata

# Define and start VM with serial console redirection to log file
echo "Starting VM installation in user session..."
virt-install \
  --connect qemu:///session \
  --name hpcboost-malware-base \
  --memory 2048 \
  --vcpus 1 \
  --boot uefi \
  --disk path="$HOME/hpcboost_collect/vm/images/hpcboost-malware-base.qcow2",format=qcow2 \
  --disk path="$HOME/hpcboost_collect/vm/iso_payloads/cidata.iso",device=disk,bus=virtio,readonly=on,format=raw \
  --os-variant ubuntu22.04 \
  --network none \
  --qemu-commandline='-netdev user,id=hostnet0,hostfwd=tcp::2222-:22 -device virtio-net-pci,netdev=hostnet0,bus=pcie.0,addr=0x10' \
  --serial pty \
  --graphics none \
  --import \
  --noautoconsole

echo "VM creation command sent. Checking VM status..."
virsh -c qemu:///session list --all
