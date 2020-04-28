#!/bin/bash

ROOT=/home/ubuntu/pipedream/runtime
YAML_FILE=vgg16_2mp_exp.yml


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
    ssh -n 172.31.74.148 -o StrictHostKeyChecking=no "cd $LOG_DIRNAME && tail -f output.log.1"
fi

