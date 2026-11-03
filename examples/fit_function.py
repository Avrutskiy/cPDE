"""Fourier fitting: match the existing array of values and derivatives."""
import os
import numpy as np
from cpde import Data, Network, Session, table

x, y = np.meshgrid(np.linspace(-1, 1, 27), np.linspace(-1, 1, 27), indexing='ij')
x, y = x.ravel(), y.ravel()

# Only the third-order requests are listed; lower derivatives will be added automatically as dependencies.
derivatives = ['xxx', 'xxy', 'xyy', 'yyy']
schema = table(derivatives, variables='xy')  # Ten rows including values.

# Input array axes: derivative row, coordinate channel, sample.
samples = np.zeros((len(schema), 2, x.size), dtype=np.float32)
for row, (order, dx, dy) in enumerate(schema):
    if order == 0:
        samples[row] = [x, y]
    elif dx == 1 and dy == 0:
        samples[row, 0] = 1
    elif dx == 0 and dy == 1:
        samples[row, 1] = 1
# Higher coordinate derivatives stay zero.

# Build the target array with ordinary Python/NumPy before defining the cost.
coefficients = np.random.default_rng(2026).uniform(-1, 1, (4, 10, 10))
target_values = np.zeros((len(schema), 1, x.size))
for row, (_, dx, dy) in enumerate(schema):
    for n in range(1, 11):
        for k in range(1, 11):
            a, b, c, d = coefficients[:, n-1, k-1] * n**int(dx) * k**int(dy) / (n*k)
            sx, cx = np.sin(n*x + dx*np.pi/2), np.cos(n*x + dx*np.pi/2)
            sy, cy = np.sin(k*y + dy*np.pi/2), np.cos(k*y + dy*np.pi/2)
            target_values[row, 0] += a*sx*sy + b*sx*cy + c*cx*sy + d*cx*cy

# One normalized squared-error weight per derivative row, for the single output.
weights = (1 / target_values[:, 0].std(axis=1)**2)[None, :]

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

    print('Derivative table (order, x count, y count):')
    print(schema)
    prediction = net.eval(inputs)
    values = prediction.to_numpy()  # Shape: (derivative row, output, sample).
    print(values)
    prediction.close()
