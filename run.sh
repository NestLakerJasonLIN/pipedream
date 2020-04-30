#!/bin/bash

ROOT=/home/ubuntu/pipedream/runtime
YAML_FILE=vgg16_2mp_exp.yml


sed -i "s/HOST0/$LOCAL_IP/g" $ROOT/image_classification/driver_configs/$YAML_FILE
sed -i "s/HOST1/$REMOTE_IP/g" $ROOT/image_classification/driver_configs/$YAML_FILE

# start
if [ "$1" = "start" ]; then
    cd $ROOT

    python driver.py --config_file image_classification/driver_configs/$YAML_FILE --launch_single_container --mount_directories /home/ubuntu /home/ubuntu/pipedream \
       | tee .log

    log_dirname=$(grep -o  -P '/home/ubuntu/pipedream/output_logs/[^;]*?\s' .log | tail -1 | tr -d '[:space:]')

    eval 'echo "get log dirname: $log_dirname"'

    export LOG_DIRNAME=$log_dirname

    rm .log
fi

# stop
if [ "$1" = "stop" ]; then
    eval 'echo "stop running for $LOG_DIRNAME"'

    cd $ROOT
    python scripts/terminate_runtime.py "$LOG_DIRNAME/machinefile"
fi

# check local log
if [ "$1" = "llog" ]; then
    cd $LOG_DIRNAME
    tail -f output.log.0
fi

# check remote log
if [ "$1" = "rlog" ]; then
    cd $LOG_DIRNAME
    ssh -n $REMOTE_IP -o StrictHostKeyChecking=no "cd $LOG_DIRNAME && tail -f output.log.1"
fi

# retrieve remote log
if [ "$1" = "get_rlog" ]; then
    cd $LOG_DIRNAME
    scp -o StrictHostKeyChecking=no $REMOTE_IP:$LOG_DIRNAME/output.log.1 $LOG_DIRNAME
fi

sed -i "s/$LOCAL_IP/HOST0/g" $ROOT/image_classification/driver_configs/$YAML_FILE
sed -i "s/$REMOTE_IP/HOST1/g" $ROOT/image_classification/driver_configs/$YAML_FILE
