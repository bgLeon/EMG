import numpy as np
class CurriculumLearner:
    """
    Decides when to stop one spec and which to execute next
    In addition, it controls how many steps the agent has given so far
    """
    def __init__(self, specs, total_steps, r_good = 0.9, num_steps = 100, min_steps = 8000):
        """Parameters
        -------
        specs: list of strings
            list with the path to the ltl sketch for each spec
        r_good: float
            suceess rate threshold to decide moving to the next spec
        num_steps: int
            max number of steps that the agents have to complete the spec.
            if it does it, we consider a hit on its 'suceess rate' 
            (this emulates considering the average reward after running a rollout for 'num_steps')
        min_steps: int
            minimum number of training steps required to the agent before considering moving to another spec
        total_steps: int
            total number of training steps that the agents have to learn all the specs
        """
        self.r_good = r_good
        self.num_steps = num_steps
        self.min_steps = min_steps
        self.total_steps = total_steps
        self.specs = specs
    
    def restart(self):
        self.current_step = 0
        self.succ_rate = {}
        for t in self.specs:
            self.succ_rate[t] = (0,0) # (hits, total)
        self.current_spec = -1

    def add_step(self):
        self.current_step += 1

    def get_current_step(self):
        return self.current_step

    def stop_learning(self):
        return self.total_steps <= self.current_step

    def get_next_spec(self):
        self.last_restart = -1
        self.current_spec = (self.current_spec+1)%len(self.specs)
        return self.get_current_spec()
    
    def get_current_spec(self):
        return self.specs[self.current_spec]

    def update_succ_rate(self, step, reward):
        t = self.get_current_spec()
        hits, total = self.succ_rate[t]
        if reward == 1 and (step-self.last_restart) <= self.num_steps:
            hits += 1.0
        total += 1.0
        self.succ_rate[t] = (hits, total)
        self.last_restart = step
        
    #Not in use 
    def stop_spec(self, step, rew_batch):
        return self.min_steps <= step and self.r_good < self.get_succ_rate()

    def get_succ_rate(self):
        t = self.get_current_spec()
        hits, total = self.succ_rate[t]
        return 0 if total == 0 else (hits/total)
