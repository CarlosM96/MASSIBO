#!/bin/bash

# Exit on error
set -e

echo "=== MASSIBO Software Installer (Aggressive Fix) ==="
echo "Detected OS: Ubuntu 24.04 (Noble)"

# 0. CLEANUP FIRST
# This is critical: we must remove the broken file before running 'apt-get update'
echo "[0/4] Removing broken repository configurations..."
sudo rm -f /etc/apt/sources.list.d/influxdata.list
sudo rm -f /etc/apt/sources.list.d/influxdata.list.save
sudo rm -f /etc/apt/keyrings/influxdata-archive_compat.gpg

# 1. Update and Install prerequisites
echo "[1/4] Updating Package Lists (Clean) and Installing dependencies..."
sudo apt-get update
sudo apt-get install -y curl gnupg2 software-properties-common wget

# 2. Install InfluxDB
echo "[2/4] Downloading InfluxDB Key and Installing..."

# Create keyrings directory
sudo mkdir -p /etc/apt/keyrings

# Download key directly from InfluxData (official source)
curl -fsSL https://repos.influxdata.com/influxdata-archive_compat.key | sudo gpg --dearmor -o /etc/apt/keyrings/influxdata-archive_compat.gpg

# Add Repo pointing to that specific keyring
echo 'deb [signed-by=/etc/apt/keyrings/influxdata-archive_compat.gpg] https://repos.influxdata.com/debian stable main' | sudo tee /etc/apt/sources.list.d/influxdata.list

sudo apt-get update
sudo apt-get install -y influxdb2 influxdb2-cli

# Start InfluxDB
sudo systemctl unmask influxdb
sudo systemctl enable influxdb
sudo systemctl start influxdb

# 3. Install Grafana
echo "[3/4] Installing Grafana..."
sudo rm -f /etc/apt/sources.list.d/grafana.list

# Add GPG key
wget -q -O - https://apt.grafana.com/gpg.key | gpg --dearmor | sudo tee /etc/apt/keyrings/grafana.gpg > /dev/null

# Add Repo
echo "deb [signed-by=/etc/apt/keyrings/grafana.gpg] https://apt.grafana.com stable main" | sudo tee /etc/apt/sources.list.d/grafana.list

sudo apt-get update
sudo apt-get install -y grafana

# Start Grafana
sudo systemctl enable grafana-server
sudo systemctl start grafana-server

# 4. Install Python Dependencies
echo "[4/4] Installing Python libraries..."
pip3 install influxdb-client requests --break-system-packages --user || pip3 install influxdb-client requests --break-system-packages

echo "=== Installation Complete! ==="
