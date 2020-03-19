from flask import Flask, request, jsonify
from flask_socketio import SocketIO,send,emit
from flask_cors import CORS
import os,sys
# sys.path.append('/share/sandbox/')
sys.path.append('/home/hrc2/hrcd/cars/carracing')
sys.path.append('/home/zhilong/Documents/HRC/CarRacing')
sys.path.append('/home/dev/scratch/cars/carracing')
#from carracing.agents.nsgaii import nsgaii_agent
from agents.nsgaii import nsgaii_agent
#from carracing.keras_trainer.run_car import run_unparsed
# from keras_trainer.run_car import run_unparsed
from keras_trainer.run_dqn_car import run_unparsed
from keras_trainer.dqn import DQNAgent
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import json
from pyvirtualdisplay import Display
import uuid
app = Flask(__name__, static_url_path='/static')
CORS(app) #not secure; set up a whitelist?
socketio = SocketIO(app)



max_reward = 0

@app.route("/")
def hello():
        return "Hello World!"
@app.route("/about")
def about():
    data = {"msg":"This is a simulated workspace where you can design racecars for OpenAI Gym's CarRacing environment."}
    return jsonify(data)

@app.route("/traindriver", methods=['POST'])
def traindriver():
    return testdrive(train=True)

@app.route("/testdrive", methods=['POST'])
def testdrive(train=False):
    carConfig = request.get_json()
    print("TEST DRIVING")
    #create a unique filename.mp4
    filename = uuid.uuid4().hex+'.mp4'
    #run the car with it
    #t = threading.Thread(target=run_unparsed, args=(carConfig,os.path.join('../static',filename))) #this is a hack to write outside the monitor-files dir
    #t.start()
    #t.join()
    # want to train it for a few episodes first
    trained_model_name = os.path.join(os.getcwd(),"keras_trainer/dqn_train_car_500_retrain.h5")
    if train == True:        
        trainer = DQNAgent(10, trained_model_name, carConfig, 0)
        trainer.train()
    with ThreadPoolExecutor(max_workers=4) as e:
        simulation = e.submit(run_unparsed, carConfig, filename, display, trained_model_name) #pass true if display is enabled
    #let's try this for now, but blocking isn't going to work if >1 user.
    #potential solution: redis+celery (blog.miguelgrinberg.com/post/using-celery-with-flask
        return jsonify({"video":filename, "result":simulation.result()})
#    result = run_unparsed(carConfig, os.path.join('../static',filename),display)
#    return jsonify({"video":filename,"result":result})
@socketio.on('connect')
def handle_connect():
    sess = str(time.time())
    emit('session_id', sess)

@socketio.on('start_ga')
def start_ga(sess):
    agent = nsgaii_agent(sess)
    t = threading.Thread(target=agent.run)
    # t.start()


@socketio.on('evaluated_car')
def handle_evaluated_car(evaluation):
    global max_reward
    # changed 2/21/20
    emit('ga_car', {"config":evaluation['car']["config"],"result":{k:v for k,v in evaluation['car'] if k is not 'config'}}, json=True, broadcast=True)
    if(evaluation['car']['reward'] > max_reward):
        max_reward = evaluation['car']['reward']
        with open("best_configs.json", 'a+') as outfile:
            outfile.write(json.dumps(evaluation['car'])+"\n")
    print('evaluated car: ' + str(evaluation))


if __name__ == "__main__":
        #display = None
        display = Display(visible=0, size=(1400,900))
        display.start()
        socketio.run(app,  host='0.0.0.0')
