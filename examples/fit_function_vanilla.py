"""Vanilla Fourier fitting: match only the value array."""
import os
import numpy as np
from cpde import Data, Network, Session

x, y = np.meshgrid(np.linspace(-1, 1, 27), np.linspace(-1, 1, 27), indexing='ij')
x, y = x.ravel(), y.ravel()

# Data axes: value row, coordinate channel, sample.
schema = np.array([[0, 0, 0]], dtype=np.int32)
samples = np.array([x, y], dtype=np.float32)[None, :, :]

# Build the target values with ordinary Python/NumPy before defining the cost.
coefficients = np.random.default_rng(2026).uniform(-1, 1, (4, 10, 10))
target_values = np.zeros((1, 1, x.size))
for n in range(1, 11):
    for k in range(1, 11):
        a, b, c, d = coefficients[:, n-1, k-1] / (n*k)
        sx, cx = np.sin(n*x), np.cos(n*x)
        sy, cy = np.sin(k*y), np.cos(k*y)
        target_values[0, 0] += a*sx*sy + b*sx*cy + c*cx*sy + d*cx*cy

# Normalize the squared value error by the target variance.
weights = np.array([[1 / target_values[0, 0].std()**2]])

with Session(os.environ['CPDE_SERVER']) as session:
    inputs = Data(samples, schema, session=session).name_variables('xy')
    targets = Data(target_values, schema, session=session).name_variables('xy')
    net = Network([2, 128, 128, 128, 128, 1], session=session).name_outputs('u')
    cost = net.cost(
        inputs=inputs,
        targets=targets,
        weights=weights,
        name='array matching',
    )
    net.train(cost=cost, epochs=1500, optimizer='rprop', reset_steps=2e-4,
              verbose=2, log_every=100)

    prediction = net.eval(inputs)
    values = prediction.to_numpy()  # Shape: (value row, output, sample).
    print(values)
    prediction.close()
