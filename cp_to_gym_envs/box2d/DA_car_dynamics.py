"""
Top-down car dynamics simulation.

Some ideas are taken from this great tutorial http://www.iforce2d.net/b2dtut/top-down-car by Chris Campbell.
This simulation is a bit more detailed, with wheels rotation.

Created by Oleg Klimov. Licensed on the same terms as the rest of OpenAI Gym.
"""

import numpy as np
import math
import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener, shape)

SIZE = 0.02
ENGINE_POWER = 100000000*SIZE*SIZE
WHEEL_MOMENT_OF_INERTIA = 4000*SIZE*SIZE
FRICTION_LIMIT = 1000000*SIZE*SIZE     # friction ~= mass ~= size^2 (calculated implicitly using density)
WHEEL_R = 27
WHEEL_W = 14
WHEELPOS = [
    (-55, +80), (+55, +80),
    (-55, -82), (+55, -82)
    ]
HULL_POLY1 = [
    (-60, +130), (+60, +130),
    (+60, +110), (-60, +110)
    ]
HULL_POLY2 = [
    (-15, +120), (+15, +120),
    (+20, +20), (-20, 20)
    ]
HULL_POLY3 = [
    (+25, +20),
    (+50, -10),
    (+50, -40),
    (+20, -90),
    (-20, -90),
    (-50, -40),
    (-50, -10),
    (-25, +20)
    ]
HULL_POLY4 = [
    (-50, -120), (+50, -120),
    (+50, -90),  (-50, -90)
    ]
WHEEL_COLOR = (0.0,  0.0, 0.0)
WHEEL_WHITE = (0.3, 0.3, 0.3)
MUD_COLOR = (0.4, 0.4, 0.0)

def hex2rgb(h):
    return tuple(int(h[i:i+2], 16)/255.0 for i in (0, 2, 4))

class Car:
    def __init__(self, world, init_angle, init_x, init_y, size = 0.02, 
        eng_power = 40000, wheel_moment = 1.6, friction_lim = 400,
        wheel_rad = 27, wheel_width = 14, wheel_pos = [(-55,+80), (+55,+80),(-55,-82), (+55,-82)],
        hull_poly1 = [(-60,+130), (+60,+130), (+60,+110), (-60,+110)], hull_poly2 = [(-15,+120), (+15,+120),(+20, +20), (-20,  20)],
        hull_poly3 = [  (+25, +20),(+50, -10),(+50, -40),(+20, -90),(-20, -90),(-50, -40),(-50, -10),(-25, +20)],
        hull_poly4 = [(-50,-120), (+50,-120),(+50,-90),  (-50,-90)],
        drive_train = [0,0,1,1],
        hull_densities = [1.0,1.0,1.0,1.0],
        steering_scalar = 0.5,
        rear_steering_scalar = 0.0,
        brake_scalar = 1.0,
        max_speed = 90,
        color = '0000cc'):
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
            "hull_densities": hull_densities,
            "steering_scalar": steering_scalar,
            "rear_steering_scalar": rear_steering_scalar,
            "brake_scalar": brake_scalar,
            "max_speed": max_speed,
            "color": '0000cc'
        }
        self.config_len = 23
        self.size = size
        self.hull_densities = hull_densities
        self.steering_scalar = steering_scalar # NEW: "power steering"
        self.rear_steering_scalar = rear_steering_scalar # NEW: do the back wheels turn inversely and by how much
        self.brake_scalar = brake_scalar # NEW: amount to scale the braking force by (should be greater than 0)
        self.max_speed = max_speed # NEW: amount in mph above which the throttle will not work
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
            position=(init_x, init_y),
            angle=init_angle,
            fixtures=[
                fixtureDef(shape=polygonShape(vertices=[(x*self.size, y*self.size) for x, y in self.hull_poly1]), density=self.hull_densities[0]),
                fixtureDef(shape=polygonShape(vertices=[(x*self.size, y*self.size) for x, y in self.hull_poly2]), density=self.hull_densities[1]),
                fixtureDef(shape=polygonShape(vertices=[(x*self.size, y*self.size) for x, y in self.hull_poly3]), density=self.hull_densities[2]),
                fixtureDef(shape=polygonShape(vertices=[(x*self.size, y*self.size) for x, y in self.hull_poly4]), density=self.hull_densities[3])
                ]
            )
        self.hull.color = hex2rgb(color)
        self.wheels = []
        self.fuel_spent = 0.0
        self.grass_traveled = 0.0
        self.wheel_poly = [
            (+self.wheel_width, -self.wheel_rad), (-self.wheel_width, -self.wheel_rad),
            (-self.wheel_width, +self.wheel_rad), (+self.wheel_width, +self.wheel_rad),
            ]
        for wx, wy in self.wheel_pos:
            front_k = 1.0 if wy > 0 else 1.0
            w = self.world.CreateDynamicBody(
                position=(init_x+wx*self.size, init_y+wy*self.size),
                angle=init_angle,
                fixtures=fixtureDef(
                    shape=polygonShape(vertices=[(x*front_k*self.size,y*front_k*self.size) for x, y in self.wheel_poly]),
                    density=0.1,
                    categoryBits=0x0020,
                    maskBits=0x001,
                    restitution=0.0)
                    )
            w.wheel_rad = front_k*self.wheel_rad*self.size
            w.color = WHEEL_COLOR # TODO: update this to base tire color on tread!!!!
            w.gas = 0.0
            w.brake = 0.0
            w.steer = 0.0
            w.phase = 0.0  # wheel angle
            w.omega = 0.0  # angular velocity
            w.skid_start = None
            w.skid_particle = None
            rjd = revoluteJointDef(
                bodyA=self.hull,
                bodyB=w,
                localAnchorA=(wx*self.size, wy*self.size),
                localAnchorB=(0,0),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=180*900*self.size*self.size,
                motorSpeed=0,
                lowerAngle=-0.4,
                upperAngle=+0.4,
                )
            w.joint = self.world.CreateJoint(rjd)
            w.tiles = set()
            w.userData = w
            self.wheels.append(w)
        self.drawlist = self.wheels + [self.hull]
        self.particles = []
    
    

    def gas(self, gas):
        """control: rear wheel drive

        Args:
            gas (float): How much gas gets applied. Gets clipped between 0 and 1.
        """
        if np.linalg.norm(self.hull.linearVelocity )> self.max_speed:
            self.brake(0.5)
            return
        gas = np.clip(gas, 0, 1)
        for w in self.wheels[2:4]:
            diff = gas - w.gas
            if diff > 0.1: diff = 0.1  # gradually increase, but stop immediately
            w.gas += diff

    def brake(self, b):
        """control: brake

        Args:
            b (0..1): Degree to which the brakes are applied. More than 0.9 blocks the wheels to zero rotation"""
        assert self.brake_scalar >= 0
        # if np.linalg.norm(self.hull.linearVelocity )> 10 and b == 0:
        #     for w in self.wheels:
        #         w.brake = 0.5
        # else:
        for w in self.wheels:
            w.brake = self.brake_scalar*b

    def steer(self, s):
        """control: steer

        Args:
            s (-1..1): target position, it takes time to rotate steering wheel from side-to-side"""
        # scale and clamp s to (-1,1)
        s_scaled = max(-1, min(self.steering_scalar*s,1))
        # s_scaled = self.steering_scalar*s
        self.wheels[0].steer = s_scaled
        self.wheels[1].steer = s_scaled
        self.wheels[2].steer = self.rear_steering_scalar*-s
        self.wheels[3].steer = self.rear_steering_scalar*-s

    def step(self, dt):
        for w in self.wheels:
            # Steer each wheel
            dir = np.sign(w.steer - w.joint.angle)
            val = abs(w.steer - w.joint.angle)
            # TODO: There's a try/except here in my code. Check why it's there/if we need it.
            w.joint.motorSpeed = dir*min(50.0*val, 3.0) # TODO: I'm casting to a float here. not sure why.
            # Matt: Weird, this looks like a differential drive????
            # Position => friction_limit
            grass = True
            friction_limit = self.friction_lim*0.6  # Grass friction if no tile
            for tile in w.tiles:
                friction_limit = max(friction_limit, self.friction_lim*tile.road_friction)
                grass = False
            if grass == True:
                self.grass_traveled += 1 # if we didn't hit any tiles, then we're on the grass?

            # Force
            forw = w.GetWorldVector( (0,1) )
            side = w.GetWorldVector( (1,0) )
            v = w.linearVelocity
            vf = forw[0]*v[0] + forw[1]*v[1]  # forward speed
            vs = side[0]*v[0] + side[1]*v[1]  # side speed

            # WHEEL_MOMENT_OF_INERTIA*np.square(w.omega)/2 = E -- energy
            # WHEEL_MOMENT_OF_INERTIA*w.omega * domega/dt = dE/dt = W -- power
            # domega = dt*W/WHEEL_MOMENT_OF_INERTIA/w.omega

            # add small coef not to divide by zero
            w.omega += dt*self.eng_power*w.gas/self.wheel_moment/(abs(w.omega)+5.0)
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

            # Random coefficient to cut oscillations in few steps (have no effect on friction_limit)
            f_force *= 205000*self.size*self.size
            p_force *= 205000*self.size*self.size
            force = np.sqrt(np.square(f_force) + np.square(p_force))

            # Skid trace
            if abs(force) > 2.0*friction_limit:
                if w.skid_particle and w.skid_particle.grass == grass and len(w.skid_particle.poly) < 30:
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
                p_force*side[0] + f_force*forw[0],
                p_force*side[1] + f_force*forw[1]), True ) #again, we were casting to float here, and now no longer

        # print(np.linalg.norm(self.hull.angularVelocity))

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
                if s1 > 0 and s2 > 0: continue
                if s1 > 0: c1 = np.sign(c1)
                if s2 > 0: c2 = np.sign(c2)
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
        p.poly = [(point1[0], point1[1]), (point2[0], point2[1])]
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

    def get_config_vector(self):
        '''
            return an array version of an unparsed config
        '''
        config = self.unparse_config(self.config)
        unpacked_config = [0 for i in range(self.config_len)]
        unpacked_config[0] = config['eng_power']    
        unpacked_config[1] = config['wheel_moment']
        unpacked_config[2] = config['friction_lim']
        unpacked_config[3] = config['wheel_rad']
        unpacked_config[4] = config['wheel_width']
        # unpacked_config[5] = config['drive_train']
        unpacked_config[5] = config['bumper']['w1']
        unpacked_config[6] = config['bumper']['w2']
        unpacked_config[7] = config['bumper']['d']
        unpacked_config[8] = config['hull_poly2']['w1']
        unpacked_config[9] = config['hull_poly2']['w2']
        unpacked_config[10] = config['hull_poly2']['d']
        unpacked_config[11] = config['hull_poly3']['w1']
        unpacked_config[12] = config['hull_poly3']['w2']
        unpacked_config[13] = config['hull_poly3']['w3']
        unpacked_config[14] = config['hull_poly3']['w4']
        unpacked_config[15] = config['hull_poly3']['d']
        unpacked_config[16] = config['spoiler']['w1']
        unpacked_config[17] = config['spoiler']['w2']
        unpacked_config[18] = config['spoiler']['d']
        unpacked_config[19] = config['steering_scalar']
        unpacked_config[20] = config['rear_steering_scalar']
        unpacked_config[21] = config['brake_scalar']
        unpacked_config[22] = config['max_speed']
        return unpacked_config

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
                config['bumper'] = {'w1':coords[1][0]*2, 'w2':coords[2][0]*2, 'd':config['hull_densities'][0]}
            if("hull_poly2" in config.keys()):
                coords = config["hull_poly2"]
                config['hull_poly2'] = {'w1':coords[1][0]*2,'w2':coords[2][0]*2,'d':config['hull_densities'][1]}
            if("hull_poly3" in config.keys()):
                coords = config["hull_poly3"]
                config['hull_poly3'] = {'w1':coords[0][0]*2,'w2':coords[1][0]*2,'w3':coords[2][0]*2,'w4':coords[3][0]*2,'d':config['hull_densities'][2]}
            if("hull_poly4" in config.keys()):
                coords = config["hull_poly4"]
                config["spoiler"] = {'w1':coords[1][0]*2,'w2':coords[2][0]*2, 'd':config['hull_densities'][3]}
            del config['hull_poly1']
            del config['hull_poly4']
        return config

    

