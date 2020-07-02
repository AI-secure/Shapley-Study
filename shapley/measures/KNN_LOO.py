from shapley.measures import Measure

class KNN_LOO(Measure):

    def __init__(self, num_train=1000, num_test=100):
        self.name = 'KNN_LOO'
        self.num_train = num_train
        self.num_test = num_test