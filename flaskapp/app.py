from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO,send,emit
from flask_cors import CORS
import os,sys
import pymongo
from flask_pymongo import PyMongo
from bson import ObjectId, json_util
import uuid
# sys.path.append('/share/sandbox/')
sys.path.append('/home/zhilong/Documents/HRC/CarRacing')
sys.path.append('/home/dev/scratch/cars/carracing_clean')
sys.path.append('/home/ubuntu/chopshop/carracing')
#from carracing.agents.nsgaii import nsgaii_agent
# from agents.nsgaii import nsgaii_agent
#from carracing.keras_trainer.run_car import run_unparsed
# from keras_trainer.run_car import run_unparsed
# from keras_trainer.run_dqn_car import run_unparsed
from keras_trainer.avg_dqn_updated_gym import DQNAgent
import time
import datetime
import threading
from concurrent.futures import ThreadPoolExecutor
import json
from pyvirtualdisplay import Display
import uuid
import glob
from .config import Config
app = Flask(__name__, static_url_path='/static')
app.config.from_object(Config)
print(app.config)
mongo = PyMongo(app)
CORS(app) #not secure; set up a whitelist?
socketio = SocketIO(app)
# app.config["MONGO_URI"] = "mongodb://localhost:27017/carRacingTest"



max_reward = 0

@app.route("/")
def hello():
    return "Hello World!"
@app.route("/help")
def help():
    return render_template("help.html")
@app.route("/success")
def success():
    return render_template("success.html",finished=False)
@app.route("/error", methods=['GET'])
def error():
    status = request.args.get('status')
    return render_template("error.html",status=status)

@app.route("/about")
def about():
    data = {"msg":"This is a simulated workspace where you can design racecars for OpenAI Gym's CarRacing environment."}
    return jsonify(data)

@app.route("/session", methods=["POST"])
def session():
    request_params = request.get_json()
    user_id = request_params["user_id"]
    if user_id == "":
        # create a new session
        # load the default config (json)
        # get the default pretrained agent path
        # get the default test drive video paths
        default_file_path = os.path.join(app.config['FILE_PATH'], app.config['DEFAULT_PATH'])

        videos = [s.replace(f'{app.config["FILE_PATH"]}/','') for s in glob.glob(f'{str(default_file_path)}/*.mp4')]
        print(f'GLOB: {str(default_file_path)}/*.h5')
        agent = glob.glob(f'{str(default_file_path)}/*.h5')[0]
        configs = glob.glob(f'{str(default_file_path)}/*config.json')
        reward_plot_path = f'{app.config["DEFAULT_PATH"]}/reward_plot.png'
        configs.sort()
        with open(configs[0], 'r') as infile:
            initial_config = json.load(infile)
        
        user_id = uuid.uuid4().hex
        current_time = datetime.datetime.utcnow()
        session = mongo.db.sessions.insert_one({ "user_id":user_id,
                                                 "time_created": current_time,
                                                 "last_modified": current_time,
                                                 "status":"designing",
                                                 "agent": agent,
                                                 "tested_designs":[], 
                                                 "tested_videos":[],
                                                 "tested_results":[],
                                                 "initial_design": initial_config,
                                                 "final_design": {},
                                                 "question_answers": {},
                                                 "initial_test_drives":videos,
                                                 "reward_plot": reward_plot_path,
                                                 "n_training_episodes":app.config['N_TRAINING_EPISODES']}) # use the auto ObjectId as the session id
        session_id = session.inserted_id
        # make a directory to store the videos and agents
        file_path = os.path.join(app.config['FILE_PATH'],user_id,str(session_id)) # session_id is an ObjectID, not str
        os.makedirs(file_path)
        session = mongo.db.sessions.find_one_and_update({"_id":session_id},{"$set":{"file_path":file_path}},return_document=pymongo.ReturnDocument.AFTER)
    else:
        # otherwise, retrieve the session
        session = mongo.db.sessions.find_one({"user_id": user_id},sort=[("_id",pymongo.DESCENDING)])
        # if it doesn't exist, then send an error
        if session == None:
            raise InvalidSession('There are no sessions for this user id.', status_code=404)
        # if the session is waiting on a job, then send an error
        if session["status"] != "designing":
            raise InvalidSession('The current session is inacessible as it is waiting for a job to finish.', status_code=403)
    # return the session id, video paths, and configuration
    return json.dumps(session, default=json_util.default)


@app.route("/submitdesign", methods=["POST"])
def submitdesign():
    request_params = request.get_json()
    print(request_params)
    try:
        user_id = request_params["userId"]
    except:
        user_id = None
    try:
        car_config = request_params['config']
    except:
        car_config = None
    try:
        question_answers = request_params['questions']
    except:
        question_answers = None
    try:
        selected_features = request_params['selectedFeatures']
    except:
        selected_features = None
    if user_id is None:
        return jsonify({'success':False,'error':'The user is not logged in.'})
    if car_config is None:
        return jsonify({'success':False, 'error':'The car design was empty or missing.'})

    # get the session
    session = mongo.db.sessions.find_one({"user_id": user_id},sort=[("_id",pymongo.DESCENDING)])
    if session == None:
        return jsonify({'success':False,'error':'There are no sessions for this user id.'})
    # if the session is waiting on a job, then send an error
    if session["status"] != "designing":
        return jsonify({'success':False,'error':'The current session is inacessible as it is waiting for a job to finish.'})

    # set the session to pending and add the config as final config
    try:
        session = mongo.db.sessions.find_one_and_update({"_id":session["_id"]},{"$set":{"last_modified": datetime.datetime.utcnow(),"time_submitted":datetime.datetime.utcnow(),"status":"pending","final_design":car_config,"question_answers":question_answers, "selected_features":selected_features}},return_document=pymongo.ReturnDocument.AFTER)
    except Exception as e:
        print(e)
        return jsonify({'success':False,'error':'There was a problem updating the design.'})
    
    return jsonify({'success':True})

@app.route("/traindriver", methods=['POST'])
def traindriver():
    return testdrive(train=True)

@app.route("/testdrive", methods=['POST'])
def testdrive(train=False):
    request_params = request.get_json()
    try:
        user_id = request_params["userId"]
    except:
        user_id = None
    try:
        car_config = request_params['config']
    except:
        car_config = None
    if user_id is None:
        return jsonify({'success':False,'error':'The user is not logged in.'})
    if car_config is None:
        return jsonify({'success':False, 'error':'The car design was empty or missing.'})

    # get the session
    session = mongo.db.sessions.find_one({"user_id": user_id},sort=[("_id",pymongo.DESCENDING)])
    if session == None:
        return jsonify({'success':False,'error':'There are no sessions for this user id.'})
    # if the session is waiting on a job, then send an error
    if session["status"] != "designing":
        return jsonify({'success':False,'error':'The current session is inacessible as it is waiting for a job to finish.'})
    trained_model_name = session['agent']
    num_episodes = int(request_params['numEpisodes'])
    print("TEST DRIVING for {} episodes".format(num_episodes))
    
    
    #create a unique filename.mp4
    video_filename = uuid.uuid4().hex+'.mp4'
    base_path = session['file_path']
    video_path = str(os.path.join(base_path,video_filename)).replace(f'{app.config["FILE_PATH"]}/','')
    #run the car with it
    #t = threading.Thread(target=run_unparsed, args=(carConfig,os.path.join('../static',filename))) #this is a hack to write outside the monitor-files dir
    #t.start()
    #t.join()
    # want to train it for a few episodes first
    # trained_model_name = os.path.join(os.getcwd(),"flask_model/avg_dqn_4_seq_model_every50_3_500_flask.h5")
    # trained_model_name = "/home/dev/scratch/cars/carracing_clean/agents/pretrained_drivers/avg_dqn_scratch_driver0.h5"
    # trained_model_name = os.path.join(os.getcwd(),"flask_model/avg_dqn_retraining_100.h5") ## PUT THE NAME OF YOUR (RE)TRAINING MODEL HERE!!!
    
    num_episodes -= 1 # train for n-1 and then call play_once to get the video
    driver = DQNAgent(num_episodes, trained_model_name, car_config, replay_freq=50, freeze_hidden=False)
    # if num_episodes > 0:
    #     training = True       
    #     driver.train()
    
    
    # with ThreadPoolExecutor(max_workers=4) as e:
    #     simulation = e.submit(run_unparsed, car_config, filename, display, trained_model_name) #pass true if display is enabled
    #let's try this for now, but blocking isn't going to work if >1 user.
    #potential solution: redis+celery (blog.miguelgrinberg.com/post/using-celery-with-flask
    # else:
    training = False
    result = driver.play_one(train=training,video_path=video_path,eps=0.01)
    driver.memory.save(os.path.join(os.getcwd(),"dumped_memory.pkl"))
    driver.env.env.close()
    driver.env.close()
    session = mongo.db.sessions.find_one_and_update({"_id":session["_id"]},
                                                    {"$set":{"last_modified":datetime.datetime.utcnow()},"$push":{"tested_designs":car_config,
                                                              "tested_results":result,
                                                              "tested_videos":video_path}},return_document=pymongo.ReturnDocument.AFTER)
    # return jsonify({"success": True, "video":filename, "result": result})
    return json.dumps({"session":session,"success":True}, default=json_util.default)
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

class InvalidSession(Exception):
    status_code = 400

    def __init__(self, message, status_code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv['message'] = self.message
        return rv

@app.errorhandler(InvalidSession)
def handle_invalid_session(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response




if __name__ == "__main__":
        #display = None
        display = Display(visible=0, size=(1400,900))
        display.start()
        socketio.run(app,  host='0.0.0.0')
