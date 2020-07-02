class Measure(object):

    def __init__(self, num_train=1000, num_test=100):
        self.name = 'None'
        self.num_train = num_train
        self.num_test = num_test

    def __str__(self):
        return self.name