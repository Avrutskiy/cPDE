"""Independent numerical fixtures for example regression tests, not example APIs."""
import numpy as np
from cpde import Data


def grid(size):
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size), indexing='ij')
    return np.column_stack((x.ravel(), y.ravel()))


def table(order):
    return np.array([(n, dx, n-dx) for n in range(order+1) for dx in range(n+1)], np.int32)


def coordinates(points, schema, session):
    samples = np.zeros((len(schema), 2, len(points)), np.float32)
    for i, (order, dx, dy) in enumerate(schema):
        if order == 0:
            samples[i] = np.asarray(points).T
        elif order == 1:
            samples[i, 0 if dx else 1] = 1
    return Data(samples, schema, session=session).name_variables('xy')


def initialize(net, seed=7, zero_output=False):
    """Paper initialization: weights +/-2/sqrt(fan-in), biases +/-0.1."""
    rng = np.random.default_rng(seed)
    state = net.state_dict()
    for i, (width, height) in enumerate(zip(net.layers[:-1], net.layers[1:])):
        weights = rng.uniform(-2/np.sqrt(width), 2/np.sqrt(width), (height, width))
        bias = rng.uniform(-.1, .1, height)
        state['weights'][i][:] = np.concatenate((weights.ravel(order='F'), bias))
    if zero_output:
        state['weights'][-1][:] = 0
    net.load_state_dict(state)


def disk_rule(radial=20, angular=96):
    nodes, weights = np.polynomial.legendre.leggauss(radial)
    radius = (nodes+1)/2
    angle = 2*np.pi*np.arange(angular)/angular
    directions = np.column_stack((np.cos(angle), np.sin(angle)))
    points = (radius[:, None, None]*directions).reshape(-1, 2)
    measures = np.repeat(weights/2*radius, angular)*2*np.pi/angular
    return points, measures


def fourier(points, schema, coefficients):
    """Analytic mixed derivatives of the paper's four-term Fourier sum.

    coefficients[0:4] multiply sin*sin, sin*cos, cos*sin, cos*cos.
    Frequencies are n*x and k*y (no extra pi); coefficients are divided by n*k.
    """
    frequency = np.arange(1, 11, dtype=np.float64)
    result = []
    for _, dx, dy in schema:
        phase_x = points[:, 0, None]*frequency + dx*np.pi/2
        phase_y = points[:, 1, None]*frequency + dy*np.pi/2
        sx, cx = np.sin(phase_x), np.cos(phase_x)
        sy, cy = np.sin(phase_y), np.cos(phase_y)
        factor = frequency[:, None]**(int(dx)-1)*frequency[None, :]**(int(dy)-1)
        values = np.zeros(len(points))
        for c, x, y in zip(coefficients, (sx, sx, cx, cx), (sy, cy, sy, cy)):
            values += np.einsum('pn,nk,pk->p', x, c*factor, y, optimize=True)
        result.append(values)
    return np.asarray(result)

