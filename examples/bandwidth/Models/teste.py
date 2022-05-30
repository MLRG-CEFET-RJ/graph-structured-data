class A():
  def __init__(self):
    self.a = 14
  def __call__(self, n):
    print('called' + str(n))
    return 'data'

a = A()
print(a(80))