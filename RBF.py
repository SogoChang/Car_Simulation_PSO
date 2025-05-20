# rbf.py
import numpy as np

class RBFNetwork:
    def __init__(self, n_input=3, n_hidden=5):
        self.n_input = n_input
        self.n_hidden = n_hidden
        # 每個 hidden node 有一個 center (3 維) 與寬度 beta
        self.centers = np.random.uniform(0, 50, (n_hidden, n_input))
        self.betas = np.random.uniform(1, 10, n_hidden)
        self.weights = np.random.uniform(-1, 1, n_hidden)

    def _gaussian(self, x, c, beta):
        return np.exp(-beta * np.linalg.norm(x - c) ** 2)

    def forward(self, x):
        x = np.array(x)
        hidden_outputs = np.array([
            self._gaussian(x, c, b) for c, b in zip(self.centers, self.betas)
        ])
        output = np.dot(self.weights, hidden_outputs)
        return output  # 可用於 mapping 到方向盤角度

    def set_parameters(self, params):
        split1 = self.n_hidden * self.n_input
        split2 = split1 + self.n_hidden
        self.centers = np.array(params[:split1]).reshape(self.n_hidden, self.n_input)
        self.betas = np.array(params[split1:split2])
        self.weights = np.array(params[split2:])
    
    def get_parameter_size(self):
        return self.n_hidden * self.n_input + self.n_hidden + self.n_hidden
