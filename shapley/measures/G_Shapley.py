from shapley.measures import Measure

class G_Shapley(Measure):

    def __init__(self, num_train=1000, num_test=100):
        self.name = 'G_Shapley'