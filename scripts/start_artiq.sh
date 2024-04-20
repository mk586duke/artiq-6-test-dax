#! /usr/bin/env nix-shell
#! nix-shell ../shell.nix -i bash -j4

# Starts a full ARTIQ session. Meant for running on a Linux Control PC.
# This will start the ARTIQ master, dashboard, and controller manager.
# The InfluxDB connector is run as a systemd service in the background.
# See ./linux-setup/artiq_influxdb.service for instructions on how to set that up.
PYTHONPATH=/home/euriqa/git/local-lib/:$PYTHONPATH
export PYTHONPATH
# Settings
EURIQA_DIR=/home/euriqa/git/euriqa-artiq
MY_IP=192.168.78.152
# Make the NAS path accessible to subprograms.
export EURIQA_NAS_DIR=/media/euriqa-nas
declare -a MASTER_ARGS_ARR=(
	"--bind=$MY_IP"
	"--name=EURIQALinuxControl"
	"--repository=$EURIQA_DIR/euriqafrontend/repository"
	"--device-db=$EURIQA_DIR/euriqabackend/databases/device_db_main_box.py"
	"--dataset-db=$EURIQA_DIR/dataset_db_23_cooling.pyon"
	)
# Process settings
SESSION_MASTER_ARGS = ""
for a in "${MASTER_ARGS_ARR[@]}"
do
	echo "Adding arg $a"
	SESSION_MASTER_ARGS+="-m="${a}" "
done

# Start ARTIQ components: first artiq_influxdb in background, then artiq_session
# artiq_influxdb has been moved to a system startup script, but leaving this here for future reference/use.
# Uncomment the trap & artiq_influxdb lines to use
# Kill any background jobs when this script exits
# Add euriqafrontend & aqctl_ to autokill applets & controllers in the background
trap 'pkill -P $$ || pkill -f euriqafrontend || pkill -f aqctl_' EXIT
# artiq_influxdb --server-master=$MY_IP --baseurl-db=http://$INFLUX_SERVER_IP:8086 --user-db=editor --password-db=LogiQYbIons --database=artiq --pattern-file=$EURIQA_DIR/influx_db_patterns.cfg &

echo -e "\n\n***About to start ARTIQ.***"
echo "Make sure to start ARTIQ Controller Managers on any other PCs:"
echo -e "\tartiq_ctlmgr --server=$MY_IP"
read -n 1 -p "Once you've started the Controller managers, press any key to continue:"
echo ""

pushd $EURIQA_DIR
artiq_session $SESSION_MASTER_ARGS -c="--server=$MY_IP"
popd

# Close
echo "Finishing & closing all processes"
