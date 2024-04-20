#! /usr/bin/env nix-shell
#! nix-shell ../shell.nix -i bash -j4

# Settings
# EURIQA_DIR=/home/tianyi/euriqa-artiq
# MY_IP=192.168.154.72
EURIQA_DIR=/home/euriqa/git/euriqa-artiq
MY_IP=192.168.78.152
# Make the NAS path accessible to subprograms.
export EURIQA_NAS_DIR=/media/euriqa-nas
declare -a MASTER_ARGS_ARR=(
	"--bind=$MY_IP"
	"--name=EURIQALinuxControl"
	"--repository=$EURIQA_DIR/euriqafrontend/repository"
	"--device-db=$EURIQA_DIR/euriqabackend/databases/device_db_main_box.py"
	)


# Start ARTIQ Master
# TODO: move this command to systemd to autostart??
# Kill any background jobs when this script exits
trap 'kill $(jobs -p)' EXIT

pushd $EURIQA_DIR
echo "ARTIQ Master args: ${MASTER_ARGS_ARR[*]}"
artiq_master "${MASTER_ARGS_ARR[@]}"
popd

# Close
echo "Finishing & closing all processes"
