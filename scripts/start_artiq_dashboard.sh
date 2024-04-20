#! /usr/bin/env nix-shell
#! nix-shell ../shell.nix -i bash -j4

# This will start ONLY the ARTIQ Dashboard.

# Settings
# SERVER_IP=192.168.154.72
SERVER_IP=192.168.78.152

# On exit, kill any processes started by this script & all EURIQAfrontend processes (i.e. applets)
trap 'pkill -P $$ || pkill -f euriqafrontend' EXIT
echo "Starting Dashboard connection to $SERVER_IP"
artiq_dashboard --server=$SERVER_IP

# Close
echo "Finishing & closing all processes"
