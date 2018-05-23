import sys
sys.path.append('/share/sandbox/')
from schwimmbad import MultiPool
from platypus import NSGAII, Problem, Real, Binary, Integer, ProcessPoolEvaluator, PoolEvaluator, CompoundOperator,SBX,HUX,MultiprocessingEvaluator
from platypus.config import default_variator
from collections import namedtuple
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
        def __init__(self):
                self.n_iters=20
                self.problem = Problem(24,3)
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
                                        Integer(0,600), #bumper_width
                                        Real(0,2), #spoiler_density
                                        Integer(10,500), #hull1_width1
                                        Integer(10,500), #hull1_width2
                                        Integer(10,500), #hull1_length
                                        Real(0,2), #hull1_density
                                        Integer(10,500), #hull2_width1
                                        Integer(10,500), #hull2_width2
                                        Integer(10,500), #hull2_length
                                        Real(0,2), #hull_density
                                        Integer(0,600), #spoiler_width
                                        Real(0,2)] #spoiler_density
                self.problem.function = self.simulate
                self.problem.directions = [self.problem.MAXIMIZE]*3
                dummy_problem = namedtuple('dummy_problem','types')
                variators = [default_variator(dummy_problem(types=[Real(0,1) if type(t)==Integer else t])) for t in self.problem.types]
                print("VARIATOR",variators[0])
                self.variator = CompoundOperator(*list(set(variators)))
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
                from carracing.keras_trainer.run_car import run
                #print("evaluating a car")
                return run(self.pack(config))
                #return [1,2,3]
        def run(self):
                print("starting the pool now")
                with MultiprocessingEvaluator(4) as evaluator:
                        print("got an evaluator")
                        algorithm = NSGAII(self.problem,
                                evaluator=evaluator, population_size=4,variator=self.variator)
                        print("initialized algorithm")
                        algorithm.run(self.n_iters)





agent = nsgaii_agent()
agent.run()
