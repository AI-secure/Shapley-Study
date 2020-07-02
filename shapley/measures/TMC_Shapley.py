from shapley.measures import Measure

class TMC_Shapley(Measure):

    def __init__(self, num_train=1000, num_test=100):
        self.name = 'TMC_Shapley'
        self.num_train = num_train
        self.num_test = num_test