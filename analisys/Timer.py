import time
import timeit

class Timer:
    
    def __init__():
        begin = None
        end = None
    
    @classmethod
    def start(cls):
        cls.begin = timeit.default_timer()
        
    @classmethod
    def finish(cls, msg=True):
        cls.end = timeit.default_timer()
        elapsed = cls.end - cls.begin
        cls.begin,cls.end = None,None
        if msg:
            print ('elapsed time: %f' % elapsed)
        else:
            return elapsed
    
    @classmethod
    def get_elapsed(cls):
        if cls.begin is None:
            print('Timer was not started.')
        else:
            return cls.finish(msg=False)
        
        
        