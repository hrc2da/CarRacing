from flask import Flask
from flask import jsonify
from flask_socketio import SocketIO,send,emit
import sys
sys.path.append('/share/sandbox/')
from carracing.agents.nsgaii import nsgaii_agent
import time
import threading
import json
app = Flask(__name__)
socketio = SocketIO(app)

max_reward = 0

@app.route("/")
def hello():
        return "Hello World!"
@app.route("/about")
def about():
    data = {"msg":"This is a simulated workspace where you can design racecars for OpenAI Gym's CarRacing environment."}
    return jsonify(data)
        
@socketio.on('connect')
def handle_connect():
    sess = str(time.time())
    emit('session_id', sess)

@socketio.on('start_ga')
def start_ga(sess):
    agent = nsgaii_agent(sess)
    t = threading.Thread(target=agent.run)
    t.start()


@socketio.on('evaluated_car')
def handle_evaluated_car(evaluation):
    global max_reward
    emit('ga_car', evaluation, json=True, broadcast=True)
    if(evaluation['car']['reward'] > max_reward):
        max_reward = evaluation['car']['reward']
        with open("best_configs.json", 'a+') as outfile:
            outfile.write(json.dumps(evaluation['car'])+"\n")
    print('evaluated car: ' + str(evaluation))


if __name__ == "__main__":
        socketio.run(app, host='0.0.0.0')
