import sys
sys.path.append('/share/sandbox/')
from schwimmbad import MultiPool
from platypus import NSGAII, Problem, Real, Binary, Integer, ProcessPoolEvaluator, PoolEvaluator, CompoundOperator,SBX,HUX
from carracing.keras_trainer.run_car import run
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
	def __init__(self,config=None):
		self.n_iters=20
		self.config = self.init_car(config)
		self.problem = Problem(26,3)
		self.problem.types = [Integer(0,1e9), #eng_power
					Real(0,4e4), #wheel_moment
					Real(0,1e6), #friction_lim
					Integer(0,270), #wheel_rad
					Integer(0,1000), #wheel_width
					Integer(-30,0), #wheel1_x
					Integer(0,160), #wheel1_y
					Integer(0,30), #wheel2_x
					Integer(0,160), #wheel2_y
					Integer(-30,0), #wheel3_x 
					Integer(-160,0), #wheel3_y
					Integer(0,30), #wheel4_x
					Integer(0,160), #wheel4_y
					Binary(4), #drive_train]
					Integer(0,1200), #bumper_width
					Real(0,200), #spoiler_density
					Integer(0,1000), #hull1_width1
					Integer(0,1000), #hull1_width2
					Integer(0,1000), #hull1_length
					Real(0,200), #hull1_density
					Integer(0,1000), #hull2_width1
					Integer(0,1000), #hull2_width2
					Integer(0,1000), #hull2_length
					Real(0,200), #hull_density
					Integer(0,1200), #spoiler_width
					Real(0,200)] #spoiler_density
		self.problem.function = self.simulate
		variators = [HUX() if type(t) == Binary else  SBX() for t in self.problem.types]
		print("VARIATOR",variators[0])
		self.variator = CompoundOperator(*variators)
	def init_car(self,config=None):
		if(config):
			self.config = config
		else:
			self.config = self.random_car()
	def random_car(self):
		return None
	def pack(self,config):
		packed_config = {}
		packed_config['eng_power'] = config[0]
		packed_config['wheel_moment'] = config[1]
		packed_config['friction_lim'] = config[2]
		packed_config['wheel_rad'] = config[3]
		packed_config['wheel_width'] = config[4]
		packed_config['wheel_pos'] = [(config[5],config[6]),
						(config[7],config[8]),
						(config[9],config[10]),
						(config[11],config[12])]
		packed_config['drive_train'] = config[13]
		packed_config['bumper'] = {'w':config[14],'d':config[15]}
		packed_config['hull_poly1'] = {'w1':config[16],'w2':config[17],'l':config[18],'d':config[19]}
		packed_config['hull_poly2'] = {'w1':config[20],'w2':config[21],'l':config[22],'d':config[23]}
		packed_config['spoiler'] = {'w':config[24],'d':config[25]}
		return packed_config
	def simulate(self,config):
		return run(self.pack(config))
	def run(self):
		with ProcessPoolEvaluator(2) as evaluator:
			self.algorithm = NSGAII(self.problem, population_size=5,variator=self.variator)
			self.algorithm.run(self.n_iters)
agent = nsgaii_agent()
agent.run()
