#!/bin/bash
touch timestamps.txt
echo start h2control $(date) >> timestamps.txt
for i in {0..4}
do
    # echo $i
    xvfb-run -a -s '-screen 0 1400x900x24' python run.py -e h2control -r $i &
done
wait
echo end h2control $(date) >> timestamps.txt