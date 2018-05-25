import sys, traceback
sys.path.append('/share/sandbox/')
#from schwimmbad import MultiPool
from platypus import NSGAII, Problem, Real, Binary, Integer,ProcessPoolEvaluator, PoolEvaluator,CompoundOperator,SBX,HUX,MultiprocessingEvaluator, run_job
from platypus.config import default_variator
from collections import namedtuple
from carracing.keras_trainer.run_car import run, init_buffer, kill_buffer
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
class nsgaii_agent:
        def __init__(self,session_id=None):
                self.n_iters=100000
                self.session_id=session_id
                self.problem = Problem(24,3,6)
                self.problem.types = [Integer(1e6,1e9), #eng_power
                                        Real(4e2,4e4), #wheel_moment
                                        Real(1e3,1e6), #friction_lim
                                        Integer(15,100), #wheel_rad
                                        Integer(15,100), #wheel_width
                                        Integer(-250,-10), #wheel1_x
                                        Integer(10,500), #wheel1_y
                                        Integer(10,250), #wheel2_x
                                        #Integer(10,500), #wheel2_y
                                        Integer(-250,-10), #wheel3_x 
                                        Integer(-500,-10), #wheel3_y
                                        Integer(10,250), #wheel4_x
                                        #Integer(-500,-10), #wheel4_y
                                        Binary(4), #drive_train]
                                        Integer(5,600), #bumper_width
                                        Real(0.1,2), #spoiler_density
                                        Integer(10,500), #hull1_width1
                                        Integer(10,500), #hull1_width2
                                        Integer(10,500), #hull1_length
                                        Real(0.1,2), #hull1_density
                                        Integer(10,500), #hull2_width1
                                        Integer(10,500), #hull2_width2
                                        Integer(10,500), #hull2_length
                                        Real(0.1,2), #hull_density
                                        Integer(5,600), #spoiler_width
                                        Real(0.1,2)] #spoiler_density
                self.problem.constraints[:] = "<=0"
                self.problem.function = self.evaluate
                self.problem.directions = [self.problem.MAXIMIZE,self.problem.MINIMIZE,self.problem.MINIMIZE]
                dummy_problem = namedtuple('dummy_problem','types')
                variators = [default_variator(dummy_problem(types=[Real(0,1) if type(t)==Integer else t])) for t in self.problem.types]
                print("VARIATOR",variators[0])
                self.variator = CompoundOperator(*list(set(variators)))
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
        def simulate(self,config):
                #print("about to import keras for simulate")
                #print("evaluating a car")
                return run(self.pack(config),session_id=self.session_id)
                #return [1,2,3]
        def run(self):
            display = init_buffer()
            print("starting the pool now")
            with errorBlindMultiprocessingEvaluator(3) as evaluator:
                print("got an evaluator")
                algorithm = NSGAII(self.problem, evaluator=evaluator, population_size=20,variator=self.variator)
                print("initialized algorithm")
                algorithm.run(self.n_iters)
            kill_buffer(display)


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
