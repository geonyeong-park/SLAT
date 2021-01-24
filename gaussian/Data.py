from sklearn.datasets import make_moons, make_circles
import numpy as np
np.random.seed(3)

mu = np.array([0.9, 0.2])
cov = np.array([[0.3, 0.], [0., 0.2**2]])

class DataGen():
    def __init__(self, n_samples=300, noise=0.1, data='uniform'):
        self.n_samples = n_samples
        self.noise = noise
        self.data = data
        assert data == 'uniform' or data == 'twomoon' or data == 'circles'
        self.generate_data()

    def generate_data(self):
        if self.data == 'uniform':
            self.X_train, self.y_train = self.generate_domain()
            self.X_test, self.y_test = self.generate_domain()
        elif self.data == 'twomoon':
            self.X_train, self.y_train = make_moons(2*self.n_samples, noise=self.noise)
            self.X_test, self.y_test = make_moons(2*self.n_samples, noise=self.noise)
        elif self.data == 'circles':
            self.X_train, self.y_train = make_circles(2*self.n_samples, noise=self.noise, factor=.6)
            self.X_test, self.y_test = make_circles(2*self.n_samples, noise=self.noise, factor=.6)

    def generate_domain(self):
        n_Y1 = int(self.n_samples*0.5)
        n_Y0 = self.n_samples - n_Y1
        Y1 = np.ones(n_Y1)
        X1 = np.random.multivariate_normal(mu, cov, n_Y1)
        Y0 = np.zeros(n_Y0)
        X0 = np.random.multivariate_normal(-mu, [[0.3, 0.], [0., 0.2**2]], n_Y0)
        Y = np.concatenate((Y1,Y0))
        X = np.concatenate((X1,X0))

        return X, Y
