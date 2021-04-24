from gym.envs.box2d import DACarRacing

def generateMap(savepath,seed):
    env = DACarRacing()
    env.seed(seed)
    env._destroy()
    env.reward = 0.0
    env.prev_reward = 0.0
    env.tile_visited_count = 0
    env.t = 0.0
    env.road_poly = []
    while True:
        success = env._create_track(savepath)
        if success:
            break
        else:
            print("trying again")


if __name__ == '__main__':
    generateMap('maps/map1.pkl', 420)