#!/bin/bash

if (( $(ifconfig | grep $LOCAL_IP | wc -l) == 1 )); then
    echo "local"
else
    echo "remote"
fi

