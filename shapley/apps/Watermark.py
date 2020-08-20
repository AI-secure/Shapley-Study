import numpy as np
from PIL import Image

from shapley.apps import App
from shapley.utils import DShap

class Watermark(App):

    def __init__(self, X, y, X_test, y_test, model_family='ResNet'):
        self.name = 'Watermark'
        self.X = X
        self.y = y
        self.X_test = X_test
        self.y_test = y_test
        self.num_train = len(X)
        self.num_test = len(X_test)
        self.watermarked = np.zeros(self.num_train)
        self.model_family = model_family

    def run(self, measure):

        with open('./CIFAR_data/watermarked_labels.txt','r') as f: # watermark data needed
            for i, line in zip(range(100), f):
                j = np.random.randint(self.num_train)
                while self.watermarked[j] == 1:
                    j = np.random.randint(self.num_train)
                self.watermarked[j] = 1
                img = np.asarray(Image.open("./CIFAR_data/trigger_set/%d.jpg" % (i + 1)).convert('RGB').resize((32, 32))).transpose(2, 0, 1)
                lbl = int(float(line.strip('\n')))
                self.X[j] = img
                self.y[j] = lbl

        dshap = DShap(X=self.X,
              y=self.y,
              X_test=self.X_test,
              y_test=self.y_test,
              num_test=self.num_test,
              model_family=self.model_family,
              measure=measure)
        self.result = dshap.run(save_every=10, err = 0.5)
        print('done!')
        print('result shown below:')
        print(self.result)