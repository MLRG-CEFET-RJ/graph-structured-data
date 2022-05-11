from abc import ABC, abstractmethod

class ModelInterface(ABC):
    @abstractmethod
    def load_train_data(self):
      pass

    @abstractmethod
    def load_test_data(self):
      pass

    @abstractmethod
    def fit(self):
      pass

    @abstractmethod
    def predict(self, x_test):
      pass