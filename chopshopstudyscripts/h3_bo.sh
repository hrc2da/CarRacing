#!/bin/bash
touch timestamps.txt
echo start h3bo $(date) >> timestamps.txt
for i in {0..11}
do
    # echo $i
    xvfb-run -a -s '-screen 0 1400x900x24' python run.py -e h3bo &
done
wait
echo end h3bo $(date) >> timestamps.txt
