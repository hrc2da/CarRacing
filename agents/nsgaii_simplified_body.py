import sys, traceback
import random
sys.path.append('/home/dev/scratch/cars/carracing_clean')
#from schwimmbad import MultiPool
from platypus import GeneticAlgorithm, NSGAII, Problem, Type, Real, Binary, Integer,ProcessPoolEvaluator, PoolEvaluator,CompoundOperator,SBX,HUX,MultiprocessingEvaluator, run_job, Generator, Solution
from platypus.config import default_variator
from collections import namedtuple
from keras_trainer.run_car import init_buffer, kill_buffer
from keras_trainer.run_car import run as car_run
from keras_trainer.avg_dqn import DQNAgent
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


def log_result(algorithm_object):
    print("CALLBACKING!!!!")
    outcomes = [r.objectives[0] for r in algorithm_object.result]
    outcomes.sort(reverse=True)
    algorithm_object.generations.append(outcomes)
    print(f'Generation {len(algorithm_object.generations)-1}: {outcomes}')

class nsgaii_agent:
        def __init__(self,session_id=None):
                self.n_iters=100 #00000
                self.population_size = 20
                self.offspring_size = 20
                self.session_id=session_id
                self.problem = Problem(24,1,6) # 24 variables, 3 objectives, 2 constraints
                self.problem.types = [Integer(1e3,1e7), #eng_power
                                        NormReal(0,5), #wheel_moment
                                        NormReal(0,1e4), #friction_lim
                                        Integer(15,100), #wheel_rad
                                        Integer(15,100), #wheel_width
                                        Integer(-250,-10), #wheel1_x
                                        Integer(10,125), #wheel1_y
                                        Integer(10,125), #wheel2_x
                                        #Integer(10,500), #wheel2_y
                                        Integer(-125,-10), #wheel3_x 
                                        Integer(-250,-10), #wheel3_y
                                        Integer(10,125), #wheel4_x
                                        #Integer(-500,-10), #wheel4_y
                                        Binary(4), #drive_train]
                                        Integer(5,300), #bumper_width
                                        NormReal(0.1,2), #spoiler_density
                                        Integer(10,250), #hull1_width1
                                        Integer(10,250), #hull1_width2
                                        Integer(10,250), #hull1_length
                                        NormReal(0.1,2), #hull1_density
                                        Integer(10,250), #hull2_width1
                                        Integer(10,250), #hull2_width2
                                        Integer(10,250), #hull2_length
                                        NormReal(0.1,2), #hull_density
                                        Integer(5,300), #spoiler_width
                                        NormReal(0.1,2)] #spoiler_density
                self.problem.constraints[:] = "<=0" #constraints returned in the second half of evaluate tuple
                self.problem.function = self.evaluate
                self.problem.directions = [self.problem.MAXIMIZE,self.problem.MINIMIZE,self.problem.MINIMIZE]
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
                
                self.driver = DQNAgent(num_episodes=1, model_name='/home/dev/scratch/cars/carracing_clean/agents/pretrained_drivers/avg_dqn_trained_model_5000.h5')
        def evaluate(self, config):
                m1 = max(config[14],config[15])
                m2 = max(config[18], config[19])
                l1 = config[16]
                l2 = config[20]
                x1 = config[5]
                y1 = config[6]
                y2 = config[9]
                x2 = config[7]
                x3 = config[8]
                x4 = config[10]
                return self.simulate(config), [-m1/2 - x1,-m1/2+x2 ,-m2/2-x4 ,-m2/2+x3,-l1 + y1 ,-l2 - y2]
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
                packed_config['wheel_pos'] = [(config[5],config[6]),
                                                (config[7],config[6]),
                                                (config[8],config[9]),
                                                (config[10],config[9])]
                packed_config['drive_train'] = config[11]
                packed_config['bumper'] = {'w':config[12],'d':config[13]}
                packed_config['hull_poly1'] = {'w1':config[14],'w2':config[15],'l':config[16],'d':config[17]}
                packed_config['hull_poly2'] = {'w1':config[18],'w2':config[19],'l':config[20],'d':config[21]}
                packed_config['spoiler'] = {'w':config[22],'d':config[23]}
                return packed_config

        def unpack(self,config):
            '''
                return an array version of an unparsed config
            '''
            unpacked_config = [0 for i in range(24)]
            unpacked_config[0] = config['eng_power']    
            unpacked_config[1] = config['wheel_moment']
            unpacked_config[2] = config['friction_lim']
            unpacked_config[3] = config['wheel_rad']
            unpacked_config[4] = config['wheel_width']
            unpacked_config[5] = config['wheel_pos'][0][0]
            unpacked_config[6] = config['wheel_pos'][0][1]
            unpacked_config[7] = config['wheel_pos'][1][0]
            unpacked_config[8] = config['wheel_pos'][2][0]
            unpacked_config[9] = config['wheel_pos'][2][1]
            unpacked_config[10] = config['wheel_pos'][3][0]
            unpacked_config[11] = config['drive_train']
            unpacked_config[12] = config['bumper']['w']
            unpacked_config[13] = config['bumper']['d']
            unpacked_config[14] = config['hull_poly1']['w1']
            unpacked_config[15] = config['hull_poly1']['w2']
            unpacked_config[16] = config['hull_poly1']['l']
            unpacked_config[17] = config['hull_poly1']['d']
            unpacked_config[18] = config['hull_poly2']['w1']
            unpacked_config[19] = config['hull_poly2']['w2']
            unpacked_config[20] = config['hull_poly2']['l']
            unpacked_config[21] = config['hull_poly2']['d']
            unpacked_config[22] = config['spoiler']['w']
            unpacked_config[23] = config['spoiler']['d']
            return unpacked_config

        def parse_config(self,config):
            l1 = 0
            l2 = 0
            spoiler_d = 35 #TODO: check that these match the poly's on the baseline car
            bumper_d = 25
            densities =[0,0,0,0]
            if(config == {}):
                return config
            else:
                if("hull_poly1" in config.keys()):
                    hull1_vars = config["hull_poly1"]
                    l1 = hull1_vars['l']
                    config["hull_poly1"] = [(-hull1_vars['w2']/2, 0),(hull1_vars['w2']/2, 0),(-hull1_vars['w1']/2, l1),(hull1_vars['w1']/2, l1)]
                    densities[0] = hull1_vars['d']

                if("hull_poly2" in config.keys()):
                    hull2_vars = config["hull_poly2"]
                    l2 = hull2_vars['l']
                    config["hull_poly2"] = [(-hull2_vars['w1']/2, 0),(hull2_vars['w1']/2, 0),(-hull2_vars['w2']/2, -l2),(hull2_vars['w2']/2, -l2)]
                    densities[1] = hull2_vars['d']

                if("spoiler" in config.keys()):
                    spoiler_w = config["spoiler"]['w']
                    config["hull_poly3"] = [(-spoiler_w, -l2),(spoiler_w, -l2),(-spoiler_w, -l2-spoiler_d),(spoiler_w, -l2-spoiler_d)]
                    densities[2] = config["spoiler"]['d']
                    del config["spoiler"]

                if("bumper" in config.keys()):
                    bumper_w = config["bumper"]['w']
                    config["hull_poly4"] = [(-bumper_w, l1),(bumper_w, l1),( -bumper_w, l1+bumper_d),(bumper_w, l1+bumper_d)]
                    densities[3] = bumper_w = config["bumper"]['d']
                    del config["bumper"]

                config["hull_densities"] = densities

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
                    config['hull_poly1'] = {'w1':coords[3][0]*2,'w2':coords[1][0]*2,'l':coords[3][1],'d':config['hull_densities'][0]}
                if("hull_poly2" in config.keys()):
                    coords = config["hull_poly2"]
                    config['hull_poly2'] = {'w1':coords[1][0]*2,'w2':coords[3][0]*2,'l':-coords[3][1],'d':config['hull_densities'][1]}
                if("hull_poly3" in config.keys()):
                    coords = config["hull_poly3"]
                    config['spoiler'] = {'w':coords[1][0],'d':config['hull_densities'][2]}
                if("hull_poly4" in config.keys()):
                    coords = config["hull_poly4"]
                    config["bumper"] = {'w':coords[1][0],'d':config['hull_densities'][3]}
                config.pop('hull_poly3')
                config.pop('hull_poly4')
            return config
        


        def simulate(self,config):
                #print("about to import keras for simulate")
                #print("evaluating a car")
                
                # result = car_run(self.pack(config),session_id=self.session_id)
                self.driver.carConfig = self.parse_config(self.pack(config))
                result = self.driver.play_one(eps=0.1,train=False)
                
                return result[0] # For now, only return the reward (single-objective)
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
            algorithm = GeneticAlgorithm(self.problem, population_size=self.population_size,offspring_size=self.offspring_size,variator=self.variator)
            print("initialized algorithm")
            algorithm.generations = [] # make this a queue for multithreading
            # first eval plus each iteration of offspring generation
            algorithm.run(self.n_iters + self.n_iters*(self.offspring_size + self.population_size), callback=log_result) # the termination condition is the # max evals, so iters x pop_size
            results = algorithm.result # should be the pareto-sorted population
            results.sort(key=lambda x: x.objectives[0])
            outcomes = [r.objectives[0] for r in results]
            configs = [self.parse_config(self.pack([self.problem.types[i].decode(v) for i,v in enumerate(r.variables)])) for r in results]    
            print(configs)
            print(outcomes)
            import pdb; pdb.set_trace()
            with open('/home/dev/scratch/cars/carracing_clean/agents/ga_results.json','w') as outfile:
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
