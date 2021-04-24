#!/bin/bash
touch timestamps.txt
echo start h2human $(date) >> timestamps.txt
while read row; do
    user_session=(${row//,/ })
    xvfb-run -a -s '-screen 0 1400x900x24' python run.py -e h2human -u ${user_session[0]} -s ${user_session[1]} -r 0 &
done < h1_human_sessions.txt
wait
echo end h2human $(date) >> timestamps.txt