#!/usr/bin/env bash
# Script to seed a clean, realistic environment for malware execution (ransomware, info-stealers).
# Must be run with sudo:  sudo bash seed_environment.sh
set -euo pipefail

SANDBOX_USER="mw_sandbox"
SANDBOX_HOME="/home/$SANDBOX_USER"

echo "=== Ensuring sandbox user exists ==="
if ! id "$SANDBOX_USER" &>/dev/null; then
    useradd -m -s /bin/bash "$SANDBOX_USER"
    echo "Created user $SANDBOX_USER"
fi

echo "=== Cleaning sandbox home directory ==="
# Remove everything except standard bash configs
find "$SANDBOX_HOME" -mindepth 1 -not -name ".bash*" -not -name ".profile" -exec rm -rf {} +

echo "=== Seeding directory structure ==="
for dir in Documents Desktop Downloads Pictures Projects .ssh .gnupg; do
    mkdir -p "$SANDBOX_HOME/$dir"
done

echo "=== Seeding decoy files ==="
# Create some text files with dummy data
cat > "$SANDBOX_HOME/Documents/financial_report_2025.csv" <<EOF
Year,Quarter,Revenue,Profit,Tax
2025,Q1,120500,45000,9000
2025,Q2,135000,52000,10400
2025,Q3,141000,55000,11000
2025,Q4,168000,72000,14400
EOF

cat > "$SANDBOX_HOME/Documents/memo.txt" <<EOF
To: All Staff
From: Management
Date: June 15, 2026
Subject: Security Awareness Training

Please ensure you do not open any suspicious email attachments.
EOF

# Create a few random documents of varying sizes to mimic files to encrypt
for i in {1..15}; do
    # Generate random text content using base64
    head -c "$((RANDOM % 200000 + 5000))" /dev/urandom | base64 > "$SANDBOX_HOME/Documents/doc_${i}.docx"
    head -c "$((RANDOM % 100000 + 5000))" /dev/urandom | base64 > "$SANDBOX_HOME/Desktop/notes_${i}.txt"
    head -c "$((RANDOM % 300000 + 5000))" /dev/urandom | base64 > "$SANDBOX_HOME/Downloads/download_${i}.pdf"
done

# Seed fake SSH keys
rm -f "$SANDBOX_HOME/.ssh/id_rsa" "$SANDBOX_HOME/.ssh/id_rsa.pub"
ssh-keygen -t rsa -b 2048 -N "" -f "$SANDBOX_HOME/.ssh/id_rsa" -q
cat > "$SANDBOX_HOME/.ssh/config" <<EOF
Host backup-server
    HostName 192.168.1.50
    User admin
    Port 22
    IdentityFile ~/.ssh/id_rsa
EOF

# Set ownership and permissions
chown -R "$SANDBOX_USER:$SANDBOX_USER" "$SANDBOX_HOME"
chmod 700 "$SANDBOX_HOME/.ssh"
chmod 600 "$SANDBOX_HOME/.ssh/id_rsa" "$SANDBOX_HOME/.ssh/config"
chmod 644 "$SANDBOX_HOME/.ssh/id_rsa.pub"

echo "=== Environment seeded successfully under $SANDBOX_HOME ==="
ls -la "$SANDBOX_HOME/Documents"
