from abc import ABC, abstractmethod

class ModelInterface(ABC):
    @abstractmethod
    def load_data(self):
      pass

    @abstractmethod
    def fit(self):
      pass

    @abstractmethod
    def predict(self, x_test):
      pass

class Child(ModelInterface):
  def __init__(self):
    ''

  def fit(self):
    print('fitting')
  def predict(self, x_test):
    for x in x_test:
      print('a predict')

child = Child()
child.fit()
child.predict()