#!/bin/bash
echo "starting xvfb with GL and rendering, noreset is to workaround memory leak bug in xvfb"
Xvfb :1 -screen 0 1400x900x24 -ac +extension GLX +render -noreset &
export DISPLAY=:1
echo "starting gunicorn with 2 workers; IMPORTANT timeout needs to be long enough to let the test drive finish."
gunicorn -w 2 -b 0.0.0.0:5000 -t 300 --log-level debug flaskapp.app:app
