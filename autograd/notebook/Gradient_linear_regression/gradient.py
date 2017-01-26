import numpy as np
from numba import jit, float32, int64
import numba

@jit(cache=True)
def get_loss(b, m, points):
    totalError = 0.
    b_gradient = 0.
    m_gradient = 0.
    N = len(points)
    for i in range(N):
        x = points[i,0]
        y = points[i,1]
        totalError += (y - (m * x + b)) ** 2
        m_gradient += -x * (y - (m * x + b)) * 2
        b_gradient += -1 * (y - (m * x + b)) * 2
    err = totalError/N 
    m_gradient /= len(points)
    b_gradient /= len(points)
    return err, m_gradient, b_gradient


@jit(cache=True)
def gradient_descent_one_step(b_current, m_current, points, learningRate):
    loss_now, m_gradient, b_gradient = get_loss(b_current, m_current, points)
    new_m = m_current - (learningRate * m_gradient)
    new_b = b_current - (learningRate * b_gradient)
    return new_b, new_m, loss_now


@jit(cache=True)
def step_gradient(b_current, m_current, points, learningRate):
    b_gradient = 0.
    m_gradient = 0.
    err        = 0.
    N = len(points)
    for i in range(N): 
        x = points[i,0]
        y = points[i,1]
        b_gradient += (2.0/N) * (((m_current * x) + b_current) - y)
        m_gradient += (2.0/N) * x * (((m_current * x) + b_current) - y)
        err        += (((m_current * x) + b_current) - y)**2
    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)
    err   = err/N
    return new_b, new_m, err


def gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations):
    b = initial_b
    b_history = []
    m = initial_m
    m_history = []
    e_history = []
    for i in range(num_iterations):
        b, m, err = gradient_descent_one_step(b, m, points, learning_rate)
        b_history.append(b)
        m_history.append(m)
        e_history.append(err)
    return np.asarray(b_history), np.asarray(m_history), np.asarray(e_history)

