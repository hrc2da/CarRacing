import numpy as np
import math
import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener, shape)

# Top-down car dynamics simulation.
#
# Some ideas are taken from this great tutorial http://www.iforce2d.net/b2dtut/top-down-car by Chris Campbell.
# This simulation is a bit more detailed, with wheels rotation.
#
# Created by Oleg Klimov. Licensed on the same terms as the rest of OpenAI Gym.

# SIZE = 0.02
# ENGINE_POWER            = 100000000*SIZE*SIZE
# WHEEL_MOMENT_OF_INERTIA = 4000*SIZE*SIZE
# FRICTION_LIMIT          = 1000000*SIZE*SIZE     # friction ~= mass ~= size^2 (calculated implicitly using density)
# WHEEL_R  = 27
# WHEEL_W  = 100
# WHEELPOS = [
#     (-15,+80), (+15,+80),
#     (-15,-82), (+15,-82)
#     ]
# HULL_POLY1 =[
#     (-10,+130), (+10,+130),
#     (+10,+110), (-10,+110)
#     ]
# HULL_POLY2 =[
#     (-15,+120), (+15,+120),
#     (+10, +20), (-10,  20)
#     ]
# HULL_POLY3 =[
#     (+15, +20),
#     (+10, -10),
#     (+10, -40),
#     (+10, -90),
#     (-10, -90),
#     (-10, -40),
#     (-10, -10),
#     (-15, +20)
#     ]
# HULL_POLY4 =[
#     (-50,-120), (+50,-120),
#     (+50,-90),  (-50,-90)
#     ]


WHEEL_COLOR = (0.0,0.0,0.0)
WHEEL_WHITE = (0.3,0.3,0.3)
MUD_COLOR   = (0.8,0.4,0.8)

class Car:
    # def __init__(self, world, init_angle, init_x, init_y, size = 0.02, 
    #     eng_power = 40000, wheel_moment = 1.6, friction_lim = 400,
    #     wheel_rad = 27, wheel_width = 100, wheel_pos = [(-15,+80), (+15,+80),(-15,-82), (+15,-82)],
    #     hull_poly1 = [(-10,+130), (+10,+130),(+10,+110), (-10,+110)], hull_poly2 = [(-15,+120), (+15,+120),(+10, +20), (-10,  20)],
    #     hull_poly3 = [(+15, +20),(+10, -10),(+10, -40),(+10, -90),(-10, -90),(-10, -40),(-10, -10),(-15, +20)],
    #     hull_poly4 = [(-50,-120), (+50,-120),(+50,-90),  (-50,-90)],
    #     drive_train = [0,0,1,1],
    #     hull_densities = [1,1,1,1]):
    def __init__(self, world, init_angle, init_x, init_y, size = 0.02, 
        eng_power = 40000, wheel_moment = 1.6, friction_lim = 400,
        wheel_rad = 27, wheel_width = 14, wheel_pos = [(-55,+80), (+55,+80),(-55,-82), (+55,-82)],
        hull_poly1 = [(-60,+130), (+60,+130), (+60,+110), (-60,+110)], hull_poly2 = [(-15,+120), (+15,+120),(+20, +20), (-20,  20)],
        hull_poly3 = [  (+25, +20),(+50, -10),(+50, -40),(+20, -90),(-20, -90),(-50, -40),(-50, -10),(-25, +20)],
        hull_poly4 = [(-50,-120), (+50,-120),(+50,-90),  (-50,-90)],
        drive_train = [0,0,1,1],
        hull_densities = [1,1,1,1]):

        self.config = {
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

        self.size = size
        self.hull_densities = hull_densities
        self.eng_power = eng_power
        self.wheel_moment = wheel_moment
        self.friction_lim = friction_lim
        self.drive_train = drive_train
        self.wheel_rad = wheel_rad
        self.wheel_width = wheel_width
        self.wheel_pos = wheel_pos
        self.hull_poly1 = hull_poly1
        self.hull_poly2 = hull_poly2
        self.hull_poly3 = hull_poly3
        self.hull_poly4 = hull_poly4


        self.world = world
        self.hull = self.world.CreateDynamicBody(
            position = (init_x, init_y),
            angle = init_angle,
            fixtures = [
                fixtureDef(shape = polygonShape(vertices=[ (x*self.size,y*self.size) for x,y in self.hull_poly1 ]), density=self.hull_densities[0]),
                fixtureDef(shape = polygonShape(vertices=[ (x*self.size,y*self.size) for x,y in self.hull_poly2 ]), density=self.hull_densities[1]),
                fixtureDef(shape = polygonShape(vertices=[ (x*self.size,y*self.size) for x,y in self.hull_poly3 ]), density=self.hull_densities[2]),
                fixtureDef(shape = polygonShape(vertices=[ (x*self.size,y*self.size) for x,y in self.hull_poly4 ]), density=self.hull_densities[3])
                ]
            )
        self.hull.color = (0.8,0.0,0.0)
        self.wheels = []
        self.fuel_spent = 0.0
        self.grass_traveled = 0.0
        self.wheel_poly= [
            (-self.wheel_width,+self.wheel_rad), (+self.wheel_width,+self.wheel_rad),
            (+self.wheel_width,-self.wheel_rad), (-self.wheel_width,-self.wheel_rad)
            ]
        for wx,wy in self.wheel_pos:
            front_k = 1.0 if wy > 0 else 1.0
            w = self.world.CreateDynamicBody(
                position = (init_x+wx*self.size, init_y+wy*self.size),
                angle = init_angle,
                fixtures = fixtureDef(
                    shape=polygonShape(vertices=[ (x*front_k*self.size,y*front_k*self.size) for x,y in self.wheel_poly ]),
                    density=.1,
                    categoryBits=0x0020,
                    maskBits=0x001,
                    restitution=0.0)
                    )
            w.wheel_rad = front_k*self.wheel_rad*self.size
            w.color = WHEEL_COLOR
            w.gas   = 0.0
            w.brake = 0.0
            w.steer = 0.0
            w.phase = 0.0  # wheel angle
            w.omega = 0.0  # angular velocity
            w.skid_start = None
            w.skid_particle = None
            rjd = revoluteJointDef(
                bodyA=self.hull,
                bodyB=w,
                localAnchorA=(wx*self.size,wy*self.size),
                localAnchorB=(0,0),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=180*900*self.size*self.size,
                motorSpeed = 0,
                lowerAngle = -0.4,
                upperAngle = +0.4,
                )
            w.joint = self.world.CreateJoint(rjd)
            w.tiles = set()
            w.userData = w
            self.wheels.append(w)
        self.drawlist =  self.wheels + [self.hull]
        self.particles = []

    def gas(self, gas):
        'control: rear wheel drive'
        gas = np.clip(gas, 0, 1)
        for i,w in enumerate(self.wheels[0:4]):
            if(self.drive_train[i] == 1):
                diff = gas - w.gas
                if diff > 0.1: diff = 0.1  # gradually increase, but stop immediately
                w.gas += diff

    def brake(self, b):
        'control: brake b=0..1, more than 0.9 blocks wheels to zero rotation'
        for w in self.wheels:
            w.brake = b

    def steer(self, s):
        'control: steer s=-1..1, it takes time to rotate steering wheel from side to side, s is target position'
        self.wheels[0].steer = s
        self.wheels[1].steer = s

    def step(self, dt):
        for w in self.wheels:
            # Steer each wheel
            dir = np.sign(w.steer - w.joint.angle)
            val = abs(w.steer - w.joint.angle)
            try:
                if len(val) > 0:
                    val = np.float32(val[0])
            except:
                pass
            w.joint.motorSpeed = float(dir)*min(50.0*val, np.float32(3.0))

            # Position => friction_limit
            grass = True
            friction_limit = self.friction_lim*0.6  # Grass friction if no tile
            for tile in w.tiles:
                friction_limit = max(friction_limit, self.friction_lim*tile.road_friction)
                grass = False
            if grass == True:
                self.grass_traveled += 1
            # Force
            forw = w.GetWorldVector( (0,1) )
            side = w.GetWorldVector( (1,0) )
            v = w.linearVelocity
            vf = forw[0]*v[0] + forw[1]*v[1]  # forward speed
            vs = side[0]*v[0] + side[1]*v[1]  # side speed

            # WHEEL_MOMENT_OF_INERTIA*np.square(w.omega)/2 = E -- energy
            # WHEEL_MOMENT_OF_INERTIA*w.omega * domega/dt = dE/dt = W -- power
            # domega = dt*W/WHEEL_MOMENT_OF_INERTIA/w.omega
            w.omega += dt*self.eng_power*w.gas/self.wheel_moment/(abs(w.omega)+5.0)  # small coef not to divide by zero
            self.fuel_spent += dt*self.eng_power*w.gas

            if w.brake >= 0.9:
                w.omega = 0
            elif w.brake > 0:
                self.brake_force = 15    # radians per second
                dir = -np.sign(w.omega)
                val = self.brake_force*w.brake
                if abs(val) > abs(w.omega): val = abs(w.omega)  # low speed => same as = 0
                w.omega += dir*val
            w.phase += w.omega*dt

            vr = w.omega*w.wheel_rad  # rotating wheel speed
            f_force = -vf + vr        # force direction is direction of speed difference
            p_force = -vs

            # Physically correct is to always apply friction_limit until speed is equal.
            # But dt is finite, that will lead to oscillations if difference is already near zero.
            f_force *= 205000*self.size*self.size  # Random coefficient to cut oscillations in few steps (have no effect on friction_limit)
            p_force *= 205000*self.size*self.size
            force = np.sqrt(np.square(f_force) + np.square(p_force))

            # Skid trace
            if abs(force) > 2.0*friction_limit:
                if w.skid_particle and w.skid_particle.grass==grass and len(w.skid_particle.poly) < 30:
                    w.skid_particle.poly.append( (w.position[0], w.position[1]) )
                elif w.skid_start is None:
                    w.skid_start = w.position
                else:
                    w.skid_particle = self._create_particle( w.skid_start, w.position, grass )
                    w.skid_start = None
            else:
                w.skid_start = None
                w.skid_particle = None

            if abs(force) > friction_limit:
                f_force /= force
                p_force /= force
                force = friction_limit  # Correct physics here
                f_force *= force
                p_force *= force

            w.omega -= dt*f_force*w.wheel_rad/self.wheel_moment
            w.ApplyForceToCenter( (
                float(p_force*side[0] + f_force*forw[0]),
                float(p_force*side[1] + f_force*forw[1])), True )

    def draw(self, viewer, draw_particles=True):
        if draw_particles:
            for p in self.particles:
                viewer.draw_polyline(p.poly, color=p.color, linewidth=5)
        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                path = [trans*v for v in f.shape.vertices]
                viewer.draw_polygon(path, color=obj.color)
                if "phase" not in obj.__dict__: continue
                a1 = obj.phase
                a2 = obj.phase + 1.2  # radians
                s1 = math.sin(a1)
                s2 = math.sin(a2)
                c1 = math.cos(a1)
                c2 = math.cos(a2)
                if s1>0 and s2>0: continue
                if s1>0: c1 = np.sign(c1)
                if s2>0: c2 = np.sign(c2)
                white_poly = [
                    (-self.wheel_width*self.size, +self.wheel_rad*c1*self.size), (+self.wheel_width*self.size, +self.wheel_rad*c1*self.size),
                    (+self.wheel_width*self.size, +self.wheel_rad*c2*self.size), (-self.wheel_width*self.size, +self.wheel_rad*c2*self.size)
                    ]
                viewer.draw_polygon([trans*v for v in white_poly], color=WHEEL_WHITE)

    def _create_particle(self, point1, point2, grass):
        class Particle:
            pass
        p = Particle()
        p.color = WHEEL_COLOR if not grass else MUD_COLOR
        p.ttl = 1
        p.poly = [(point1[0],point1[1]), (point2[0],point2[1])]
        p.grass = grass
        self.particles.append(p)
        while len(self.particles) > 30:
            self.particles.pop(0)
        return p

    def destroy(self):
        self.world.DestroyBody(self.hull)
        self.hull = None
        for w in self.wheels:
            self.world.DestroyBody(w)
        self.wheels = []

