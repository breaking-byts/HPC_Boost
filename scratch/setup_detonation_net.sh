#!/usr/bin/env bash
# Script to set up isolated network namespaces for HPC malware collection.
# Must be run with sudo:  sudo bash setup_detonation_net.sh
set -euo pipefail

# Namespaces and bridge names
NS_MAL="ns-malware"
NS_SRV="ns-services"
BR="br-hpc"

echo "=== Cleaning up existing setup if any ==="
ip netns del "$NS_MAL" 2>/dev/null || true
ip netns del "$NS_SRV" 2>/dev/null || true
ip link del "$BR" 2>/dev/null || true
rm -rf "/etc/netns/$NS_MAL" 2>/dev/null || true

echo "=== Creating namespaces and bridge ==="
ip netns add "$NS_MAL"
ip netns add "$NS_SRV"
ip link add name "$BR" type bridge
ip link set dev "$BR" up

echo "=== Creating and configuring veth-mal ==="
# veth pair between host (attached to bridge) and ns-malware
ip link add veth-mal type veth peer name veth-mal-ns
ip link set veth-mal master "$BR"
ip link set veth-mal up
ip link set veth-mal-ns netns "$NS_MAL"

# Configure ns-malware interface
ip netns exec "$NS_MAL" ip link set dev lo up
ip netns exec "$NS_MAL" ip link set dev veth-mal-ns up
ip netns exec "$NS_MAL" ip addr add 10.0.0.2/24 dev veth-mal-ns
ip netns exec "$NS_MAL" ip route add default via 10.0.0.1

echo "=== Creating and configuring veth-srv ==="
# veth pair between host (attached to bridge) and ns-services
ip link add veth-srv type veth peer name veth-srv-ns
ip link set veth-srv master "$BR"
ip link set veth-srv up
ip link set veth-srv-ns netns "$NS_SRV"

# Configure ns-services interface
ip netns exec "$NS_SRV" ip link set dev lo up
ip netns exec "$NS_SRV" ip link set dev veth-srv-ns up
ip netns exec "$NS_SRV" ip addr add 10.0.0.1/24 dev veth-srv-ns

echo "=== Creating DNS configuration for ns-malware ==="
# Using /etc/netns/<ns>/resolv.conf so processes in ns-malware resolve via ns-services (10.0.0.1)
mkdir -p "/etc/netns/$NS_MAL"
echo "nameserver 10.0.0.1" > "/etc/netns/$NS_MAL/resolv.conf"

echo "=== Ensuring complete isolation (no forwarding to external interfaces) ==="
# Prevent any packet forwarding from the bridge to physical interfaces (egress block)
sysctl -w net.ipv4.conf.br-hpc.forwarding=0 || true

echo "=== Verification ==="
echo "Namespaces:"
ip netns list
echo "Bridge status:"
ip link show dev "$BR"
echo "ns-malware routes:"
ip netns exec "$NS_MAL" ip route
echo "ns-services routes:"
ip netns exec "$NS_SRV" ip route
echo "DNS configuration for ns-malware:"
cat "/etc/netns/$NS_MAL/resolv.conf"

echo "=== Network namespace setup complete successfully ==="
