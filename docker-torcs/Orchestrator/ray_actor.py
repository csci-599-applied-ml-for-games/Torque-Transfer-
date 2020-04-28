import ray
import time

@ray.remote
class Actor(object):

    def __init__(self):
        pass

    def reset_env(self):
        pass

    def steps(self, actions):
        for a in actions:
            self.step(a)

    def step(self, action):
        pass

    def returnYields(self):
        pass


@ray.remote
def something():
    time.sleep(0.01)
    return ray.services.get_node_ip_address()