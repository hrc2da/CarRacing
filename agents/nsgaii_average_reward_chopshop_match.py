import sys, traceback
import random
sys.path.append('/home/dev/scratch/cars/carracing_clean')
#from schwimmbad import MultiPool
from platypus import GeneticAlgorithm, NSGAII, Problem, Type, Real, Binary, Integer,ProcessPoolEvaluator, PoolEvaluator,CompoundOperator,SBX,HUX,MultiprocessingEvaluator, run_job, Generator, Solution, InjectedPopulation
from platypus.config import default_variator
from collections import namedtuple
# from keras_trainer.run_car import init_buffer, kill_buffer
# from keras_trainer.run_car import run as car_run
from keras_trainer.avg_dqn import DQNAgent
from pyvirtualdisplay import Display
from copy import deepcopy
import time
import json
'''
Car Config Format:
self.eng_power = int
self.moment = float
self.friction_lim = float 
self.wheel_r = int
self.wheel_w = int
self.wheel_pos = [(),(),(),()]
self.hull_poly1 = {w1:int,w2:int,l:int,d:float}
self.hull_poly2 = {w1:int,w2:int,l:int,d:float}
self.bumper = int (width)
self.spoiler = int (width)


Flattened Car Config Format
--range max is an order of magnitude above the baseline
eng_power: Integer(0,1e9)
wheel_moment: Real(0,4e4)
friction_lim: Real(0,)
wheel_rad: 
wheel_width:
wheel1_x:
wheel1_y:
wheel2_x:
wheel2_y:
wheel3_x:
wheel3_y:
wheel4_x:
wheel4_y:
drive_train:
bumper_width:
hull1_width1:
hull1_width2:
hull1_length:
hull2_width1:
hull2_width2:
hull2_length:
spoiler_width:

Objectives:

gym_reward:
fuel_consumption:
num_grass_tiles:
'''
def init_buffer():
    #testing os level display fix
    global orig
    #testing os level display fix
    display = Display(visible=0,size=(1400,900))
    display.start()
    #orig = os.environ["DISPLAY"]
    return display

def kill_buffer(display):
    display.sendstop()

def log_result(algorithm_object):
    print("CALLBACKING!!!!")
    outcomes = [r.objectives[0] for r in algorithm_object.result]
    outcomes.sort(reverse=True)
    algorithm_object.generations.append(outcomes)
    print(f'Generation {len(algorithm_object.generations)-1}: {outcomes}')

class nsgaii_agent:
    def __init__(self,session_id=None):
        self.n_iters=50 #00000
        self.population_size = 50
        self.offspring_size = 50
        self.session_id=session_id
        self.problem = Problem(15,1) # 15 variables, 1 objectives, 0 constraints (we need constraints more for wheelpos)
        
        # we fix the wheel pos, since chopshop doesn't allow user to change it
        # for the body shape, we are fixing the y vals
        # allowing the GA to expand horizontally on each axis
        # but symmetrically
        # this matches what chopshop allows
        # thus there are two variables for all polys except hull_poly3
        # which is an octagon and has four.

        self.problem.types = [Integer(1e3,1e6), #eng_power          0
                                NormReal(0,5), #wheel_moment        1
                                NormReal(0,1e4), #friction_lim      2
                                Integer(10,100), #wheel_rad         3
                                Integer(10,100), #wheel_width       4
                                Integer(5,300), #bumper_width1      5
                                Integer(5,300), #bumper_width2      6
                                Integer(10,250), #hull2_width1      7
                                Integer(10,250), #hull2_width2      8
                                Integer(10,250), #hull3_width1      9
                                Integer(10,250), #hull3_width2      10
                                Integer(10,250), #hull3_width3      11
                                Integer(10,250), #hull3_width4      12
                                Integer(5,300), #spoiler_width1     13
                                Integer(5,300)] #spoiler_width2     14
        # self.problem.constraints[:] = "<=0" #constraints returned in the second half of evaluate tuple
        self.problem.function = self.evaluate
        self.problem.directions = [self.problem.MAXIMIZE]#,self.problem.MINIMIZE,self.problem.MINIMIZE]
        dummy_problem = namedtuple('dummy_problem','types')
        variators = [default_variator(dummy_problem(types=[Real(0,1) if type(t)==Integer else t])) for t in self.problem.types]
        print("VARIATOR",variators[0])
        self.variator = CompoundOperator(*list(set(variators)))

        # setup an initial population (leave empty if you don't want one)
        seed_car = Solution(self.problem)
        eng_power = 40000
        wheel_moment = 1.6
        friction_lim = 400
        wheel_rad = 27
        wheel_width = 14
        wheel_pos = [(-55,+80), (+55,+80),(-55,-82), (+55,-82)]
        hull_poly1 = [(-60,+130), (+60,+130), (+60,+110), (-60,+110)]
        hull_poly2 = [(-15,+120), (+15,+120),(+20, +20), (-20,  20)]
        hull_poly3 = [  (+25, +20),(+50, -10),(+50, -40),(+20, -90),(-20, -90),(-50, -40),(-50, -10),(-25, +20)]
        hull_poly4 = [(-50,-120), (+50,-120),(+50,-90),  (-50,-90)]
        drive_train = [0,0,1,1]
        hull_densities = [1,1,1,1]
        seed_car_config =  {
            "eng_power": eng_power,
            "wheel_moment": wheel_moment,
            "friction_lim": friction_lim,
            "wheel_rad": wheel_rad,
            "wheel_width": wheel_width,
            "wheel_pos": wheel_pos,
            "hull_poly1": hull_poly1,
            "hull_poly2": hull_poly2,
            "hull_poly3": hull_poly3,
            "hull_poly4": hull_poly4,
            "drive_train": drive_train,
            "hull_densities": hull_densities
        }
        seed_car_config_arr = self.unpack(self.unparse_config(seed_car_config))
        seed_car.variables = seed_car_config_arr
        self.init_pop = [deepcopy(seed_car) for i in range(self.population_size//10)]
        self.driver = DQNAgent(num_episodes=1, model_name='/home/dev/scratch/cars/carracing_clean/agents/pretrained_drivers/avg_dqn_ep_300.h5')
    def evaluate(self, config):
        return self.simulate(config)

    def pack(self,config):

        '''
        "eng_power": eng_power,
        "wheel_moment": wheel_moment,
        "friction_lim": friction_lim,
        "wheel_rad": wheel_rad,
        "wheel_width": wheel_width,
        "wheel_pos": wheel_pos,
        "hull_poly1": hull_poly1,
        "hull_poly2": hull_poly2,
        "hull_poly3": hull_poly3,
        "hull_poly4": hull_poly4,
        "drive_train": drive_train,
        "hull_densities": hull_densities
        '''
        packed_config = {}
        packed_config['eng_power'] = config[0]
        packed_config['wheel_moment'] = config[1]
        packed_config['friction_lim'] = config[2]
        packed_config['wheel_rad'] = config[3]
        packed_config['wheel_width'] = config[4]
        packed_config['bumper'] = {'w1':config[5], 'w2':config[6]}
        packed_config['hull_poly2'] = {'w1':config[7],'w2':config[8]}
        packed_config['hull_poly3'] = {'w1':config[9],'w2':config[10],'w3':config[11],'w4':config[12]}
        packed_config['spoiler'] = {'w1':config[13], 'w2':config[14]}
        return packed_config

    def unpack(self,config):
        '''
            return an array version of an unparsed config
        '''
        unpacked_config = [0 for i in range(20)]
        unpacked_config[0] = config['eng_power']    
        unpacked_config[1] = config['wheel_moment']
        unpacked_config[2] = config['friction_lim']
        unpacked_config[3] = config['wheel_rad']
        unpacked_config[4] = config['wheel_width']
        unpacked_config[5] = config['bumper']['w1']
        unpacked_config[6] = config['bumper']['w2']
        unpacked_config[7] = config['hull_poly2']['w1']
        unpacked_config[8] = config['hull_poly2']['w2']
        unpacked_config[9] = config['hull_poly3']['w1']
        unpacked_config[10] = config['hull_poly3']['w2']
        unpacked_config[11] = config['hull_poly3']['w3']
        unpacked_config[12] = config['hull_poly3']['w4']
        unpacked_config[13] = config['spoiler']['w1']
        unpacked_config[14] = config['spoiler']['w2']
        return [self.problem.types[i].encode(unpacked_config[i]) for i in range(self.problem.nvars)]

    def parse_config(self,config):
        # l1 = 0
        # l2 = 0
        # spoiler_d = 35 #TODO: check that these match the poly's on the baseline car
        # bumper_d = 25
        # densities =[1,1,1,1]
        # for reference, from the baseline car: 
        # bumper is hull_poly1, spoiler is hull_poly4
        # hull_poly1 = [(-60,+130), (+60,+130), (+60,+110), (-60,+110)]
        # hull_poly2 = [(-15,+120), (+15,+120),(+20, +20), (-20,  20)]
        # hull_poly3 = [  (+25, +20),(+50, -10),(+50, -40),(+20, -90),(-20, -90),(-50, -40),(-50, -10),(-25, +20)]
        # hull_poly4 = [(-50,-120), (+50,-120),(+50,-90),  (-50,-90)]
        bumper_y1 = 130
        bumper_y2 = 110
        h2_y1 = 120
        h2_y2 = 20
        h3_y1 = 20
        h3_y2 = -10
        h3_y3 = -40
        h3_y4 = -90
        spoiler_y1 = -120
        spoiler_y2 = -90
        if(config == {}):
            return config
        else:
            if("bumper" in config.keys()):
                bumper = config["bumper"]
                config["hull_poly1"] = [(-bumper['w1']/2, bumper_y1),(bumper["w1"]/2, bumper_y1),( bumper["w2"]/2, bumper_y2),(-bumper["w2"]/2, bumper_y2)]
                # densities[0] = bumper["d"]
                del config["bumper"]

            if("hull_poly2" in config.keys()):
                hull2 = config["hull_poly2"]
                config["hull_poly2"] = [(-hull2['w1']/2, h2_y1),(hull2['w1']/2, h2_y1),(hull2['w2']/2, h2_y2),(-hull2['w2']/2, h2_y2)]
                # densities[1] = hull2['d']

            if("hull_poly3" in config.keys()):
                hull3 = config["hull_poly3"]
                config["hull_poly3"] = [(hull3['w1']/2, h3_y1),(hull3['w2']/2, h3_y2),(hull3['w3']/2, h3_y3),(hull3['w4']/2, h3_y4), \
                                        (-hull3['w4']/2, h3_y4),(-hull3['w3']/2, h3_y3),(-hull3['w2']/2, h3_y2),(-hull3['w1']/2, h3_y1)]
                # densities[2] = hull3['d']

            if("spoiler" in config.keys()):
                spoiler = config["spoiler"]
                config["hull_poly4"] = [(-spoiler['w1']/2, spoiler_y1),(spoiler['w1']/2, spoiler_y1),(spoiler['w2']/2, spoiler_y2),(-spoiler['w2']/2, spoiler_y2)]
                # densities[3] = spoiler['d']
                del config["spoiler"]
            config["drive_train"] = [0,0,1,1] # rwd fixed for now
            config["hull_densities"] = [1,1,1,1] # fixed densities
            config["wheel_pos"] = [(-55,+80), (+55,+80),(-55,-82), (+55,-82)]
        return config

    def unparse_config(self,config):
        '''
            translate a config in the carracing format back into the ga format (packed)
            This returns a dict, for the GA we need an array.
            to get all the way to the ga format, call unpack(unparse_config(config))
            This is reverse-engineered from parse_config
        '''
        if(config == {}):
            return config
        else:
            if("hull_poly1" in config.keys()):
                coords = config["hull_poly1"]
                config['bumper'] = {'w1':coords[1][0]*2, 'w2':coords[2][0]*2}
            if("hull_poly2" in config.keys()):
                coords = config["hull_poly2"]
                config['hull_poly2'] = {'w1':coords[1][0]*2,'w2':coords[2][0]*2}
            if("hull_poly3" in config.keys()):
                coords = config["hull_poly3"]
                config['hull_poly3'] = {'w1':coords[0][0]*2,'w2':coords[1][0]*2,'w3':coords[2][0]*2,'w4':coords[3][0]*2}
            if("hull_poly4" in config.keys()):
                coords = config["hull_poly4"]
                config["spoiler"] = {'w1':coords[1][0]*2,'w2':coords[2][0]*2}
            del config['hull_poly1']
            del config['hull_poly4']
        return config
    


    def simulate(self,config):
            #print("about to import keras for simulate")
            #print("evaluating a car")
            
            # result = car_run(self.pack(config),session_id=self.session_id)
            self.driver.carConfig = self.parse_config(self.pack(config))
            result = 0
            for i in range(5):
                result += self.driver.play_one(eps=0.1,train=False)[0] # For now, only return the reward (single-objective)
            return result/5 
            #return [1,2,3]
        

    def run(self):
        display = init_buffer()
        print("starting the pool now")
        # with errorBlindMultiprocessingEvaluator(3) as evaluator:
            # print("got an evaluator")
            # algorithm = NSGAII(self.problem, evaluator=evaluator, population_size=20,variator=self.variator)
            # print("initialized algorithm")
            # algorithm.run(self.n_iters)
            # results = algorithm.result # should be the pareto-sorted population
            # import pdb; pdb.set_trace() # let's see what we can do with the results object
        algorithm = GeneticAlgorithm(self.problem, population_size=self.population_size,offspring_size=self.offspring_size,variator=self.variator, generator=InjectedPopulation(self.init_pop))
        print("initialized algorithm")
        algorithm.generations = [] # make this a queue for multithreading
        # first eval plus each iteration of offspring generation
        algorithm.run(self.population_size + self.n_iters*(self.offspring_size + self.population_size), callback=log_result) # the termination condition is the # max evals, so iters x pop_size
        results = algorithm.result # should be the pareto-sorted population
        results.sort(key=lambda x: x.objectives[0])
        outcomes = [r.objectives[0] for r in results]
        configs = [self.parse_config(self.pack([self.problem.types[i].decode(v) for i,v in enumerate(r.variables)])) for r in results]    
        print(configs)
        print(outcomes)
        import pdb; pdb.set_trace()
        with open('/home/dev/scratch/cars/carracing_clean/agents/ga_results_300_driver_no_drivetrain_or_densities_avg_5.json','w') as outfile:
            json.dump([outcomes,configs,algorithm.generations], outfile)

        kill_buffer(display)


class NormReal(Real):
    """Represents a floating-point value with min and max bounds.
    Attributes
    ----------
    min_value : int
        The minimum value (inclusive)
    max_value: int
        The maximum value (inclusive)
    """
    
    def __init__(self, min_value, max_value):
        super(NormReal, self).__init__(min_value, max_value)
        self.min_value = float(min_value)
        self.max_value = float(max_value)
        
    def rand(self):
        return random.normalvariate(self.min_value+self.max_value/2, self.min_value+self.max_value/8)
        
    def __str__(self):
        return "Real(%f, %f)" % (self.min_value, self.max_value)


class errorBlindMultiprocessingEvaluator(MultiprocessingEvaluator):
    def __init__(self,processes=None):
        super(errorBlindMultiprocessingEvaluator,self).__init__(processes)

    def evaluate_all(self, jobs, **kwargs):
        MAX_ATTEMPTS=3
        log_frequency = kwargs.get("log_frequency", None)
        if log_frequency is None:
            for attempts in range(MAX_ATTEMPTS):
                try:
                    result = list(self.map_func(run_job, jobs))
                    return result
                    break
                except Exception as e:
                    exception = e
                    traceback.print_exc(file=sys.stdout)
            raise exception
        else:
            result = []
            job_name = kwargs.get("job_name", "Batch Jobs")
            start_time = time.time()

            for chunk in _chunks(jobs, log_frequency):
                for attempts in range(MAX_ATTEMPTS):
                    try:
                        result.extend(self.map_func(run_job, chunk))
                        LOGGER.log(logging.INFO,
                                "%s running; Jobs Complete: %d, Elapsed Time: %s",
                                job_name,
                                len(result),
                                datetime.timedelta(seconds=time.time()-start_time))
                        break
                    except Exception as e:
                        exception = e
                        traceback.print_exc(file=sys.stdout)    
                    raise exception #this might not be necessary since it's only a chunk
            return result

if __name__ == "__main__":
    agent = nsgaii_agent()
    try:
        agent.run()
    except Exception as e:
        traceback.print_exc(file=sys.stdout)    
        print("problem", e)
