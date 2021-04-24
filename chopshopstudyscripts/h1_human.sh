#!/bin/bash
touch timestamps.txt
echo start h1human $(date) >> timestamps.txt
while read row; do
    user_session=(${row//,/ })
    xvfb-run -a -s '-screen 0 1400x900x24' python run.py -e h1human -u ${user_session[0]} -s ${user_session[1]} &
done < h1_human_sessions.txt
wait
echo end h1human $(date) >> timestamps.txt