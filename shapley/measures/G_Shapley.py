from shapley.measures import Measure

class G_Shapley(Measure):

    def __init__(self):
        self.name = 'G_Shapley'


    def score(self, X_train, y_train, X_test, y_test, model_family='', model=None, iterations=10, err=0.5):

        sources = {i:np.array([i]) for i in range(X_train.shape[0])}
        address = None
        learning_rate = self._one_step_lr()
        model = ShapNN(self.problem, batch_size=1, max_epochs=1,
                     learning_rate=learning_rate, weight_decay=0.,
                     validation_fraction=0, optimizer='sgd',
                     address=address, hidden_units=self.hidden_units)
        for iteration in range(iterations):
            model.fit(np.zeros((0, self.X.shape[-1])), self.y)
            if 10 * (iteration+1) / iterations % 1 == 0:
                print('{} out of {} G-Shapley iterations'.format(iteration + 1, iterations))
            marginal_contribs = np.zeros(len(sources.keys()))
            model.fit(self.X, self.y, self.X_test, self.y_test, sources=sources,
                      metric=self.metric, max_epochs=1, batch_size=1)
            val_result = model.history['metrics']
            marginal_contribs[1:] += val_result[0][1:]
            marginal_contribs[1:] -= val_result[0][:-1]
            individual_contribs = np.zeros(self.X.shape[0])
            for i, index in enumerate(model.history['idxs'][0]):
                individual_contribs[sources[index]] += marginal_contribs[i]
                individual_contribs[sources[index]] /= len(sources[index])
            self.mem_g = np.concatenate(
                [self.mem_g, np.reshape(individual_contribs, (1,-1))])
            self.idxs_g = np.concatenate(
                [self.idxs_g, np.reshape(model.history['idxs'][0], (1,-1))])