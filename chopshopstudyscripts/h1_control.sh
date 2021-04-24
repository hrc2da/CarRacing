#!/bin/bash
touch timestamps.txt
echo start h1control $(date) >> timestamps.txt
for i in {0..9}
do
    # echo $i
    xvfb-run -a -s '-screen 0 1400x900x24' python run.py -e h1control &
done
wait
echo end h1control $(date) >> timestamps.txt