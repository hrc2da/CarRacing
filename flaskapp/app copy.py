from flask import Flask, request, jsonify
from flask_socketio import SocketIO,send,emit
from flask_cors import CORS
import os,sys
# sys.path.append('/share/sandbox/')
sys.path.append('/home/zhilong/Documents/HRC/CarRacing')
sys.path.append('/home/dev/scratch/cars/carracing_clean')
#from carracing.agents.nsgaii import nsgaii_agent
# from agents.nsgaii import nsgaii_agent
#from carracing.keras_trainer.run_car import run_unparsed
# from keras_trainer.run_car import run_unparsed
# from keras_trainer.run_dqn_car import run_unparsed
from keras_trainer.avg_dqn import DQNAgent
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
    request_params = request.get_json()
    car_config = request_params['config']
    num_episodes = int(request_params['numEpisodes'])
    print("TEST DRIVING for {} episodes".format(num_episodes))
    #create a unique filename.mp4
    filename = uuid.uuid4().hex+'.mp4'
    #run the car with it
    #t = threading.Thread(target=run_unparsed, args=(carConfig,os.path.join('../static',filename))) #this is a hack to write outside the monitor-files dir
    #t.start()
    #t.join()
    # want to train it for a few episodes first
    # trained_model_name = os.path.join(os.getcwd(),"flask_model/avg_dqn_4_seq_model_every50_3_500_flask.h5")
    trained_model_name = "/home/dev/scratch/cars/carracing_clean/agents/pretrained_drivers/avg_dqn_scratch_driver0.h5"
    # trained_model_name = os.path.join(os.getcwd(),"flask_model/avg_dqn_retraining_100.h5") ## PUT THE NAME OF YOUR (RE)TRAINING MODEL HERE!!!
    
    num_episodes -= 1 # train for n-1 and then call play_once to get the video
    driver = DQNAgent(num_episodes, trained_model_name, car_config, replay_freq=50, freeze_hidden=False)
    if num_episodes > 0:
        training = True       
        driver.train()
    # with ThreadPoolExecutor(max_workers=4) as e:
    #     simulation = e.submit(run_unparsed, car_config, filename, display, trained_model_name) #pass true if display is enabled
    #let's try this for now, but blocking isn't going to work if >1 user.
    #potential solution: redis+celery (blog.miguelgrinberg.com/post/using-celery-with-flask
    else:
        training = False
    result = driver.play_one(train=training,video_path=filename,eps=0)
    driver.memory.save(os.path.join(os.getcwd(),"dumped_memory.pkl"))
    driver.env.close()
    import pdb; pdb.set_trace()
    return jsonify({"video":filename, "result": result})
#    result = run_unparsed(carConfig, os.path.join('../static',filename),display)
#    return jsonify({"video":filename,"result":result})
#     carConfig = request.get_json()
#     print("TEST DRIVING")
#     #create a unique filename.mp4
#     filename = uuid.uuid4().hex+'.mp4'
#     #run the car with it
#     #t = threading.Thread(target=run_unparsed, args=(carConfig,os.path.join('../static',filename))) #this is a hack to write outside the monitor-files dir
#     #t.start()
#     #t.join()
#     # want to train it for a few episodes first
#     trained_model_name = os.path.join(os.getcwd(),"flask_model/bigdqn_trained_model_10000.h5")
#     if train == True:        
#         trainer = DQNAgent(10, trained_model_name, carConfig, 0)
#         trainer.train()
#     with ThreadPoolExecutor(max_workers=4) as e:
#         simulation = e.submit(run_unparsed, carConfig, filename, display, trained_model_name) #pass true if display is enabled
#     #let's try this for now, but blocking isn't going to work if >1 user.
#     #potential solution: redis+celery (blog.miguelgrinberg.com/post/using-celery-with-flask
#         return jsonify({"video":filename, "result":simulation.result()})
# #    result = run_unparsed(carConfig, os.path.join('../static',filename),display)
# #    return jsonify({"video":filename,"result":result})
@socketio.on('connect')
def handle_connect():
    sess = str(time.time())
    emit('session_id', sess)

# @socketio.on('start_ga')
# def start_ga(sess):
#     agent = nsgaii_agent(sess)
#     t = threading.Thread(target=agent.run)
#     # t.start()


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
