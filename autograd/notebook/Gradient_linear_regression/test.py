import numpy as np
import matplotlib.pyplot as plt
from gradient import gradient_descent_runner, get_loss

points = np.loadtxt("data.csv", delimiter=",").astype('float32')

learning_rate = 0.00005
initial_b = 100 # initial y-intercept guess
initial_m = -8 # initial slope guess
num_iterations = 100000

b_history, m_history, e_history = gradient_descent_runner(points,
                                                          initial_b,
                                                          initial_m,
                                                          learning_rate,
                                                          num_iterations)

plt.plot(e_history[:50])
plt.show()
