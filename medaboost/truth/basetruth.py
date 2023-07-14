from abc import ABCMeta, abstractmethod
import numpy as np


class BaseTruth(object):
    __metaclass__ = ABCMeta
    w_workers = None

    @abstractmethod
    def infer(self, y):
    	pass

    def worker_renorm(self):
        self.w_workers = self.w_workers/np.linalg.norm(self.w_workers,
                                                       ord=1)
    
    def worker_confidence(self):
        self.worker_renorm()
        return self.w_workers
