"""Fourier fitting: explicit value and first-derivative cost."""
import os
import numpy as np
from cpde import Data, Network, Session, table

# The paper's 27 x 27 grid on [-1, 1]^2.
x, y = np.meshgrid(np.linspace(-1, 1, 27), np.linspace(-1, 1, 27), indexing='ij')
x, y = x.ravel(), y.ravel()

# The paper's Fourier family: 400 independent coefficients, frequencies 1..10.
coefficients = np.random.default_rng(2026).uniform(-1, 1, (4, 10, 10))
f = np.zeros(x.size)
fx = np.zeros(x.size)
fy = np.zeros(x.size)
for n in range(1, 11):
    for k in range(1, 11):
        a, b, c, d = coefficients[:, n-1, k-1] / (n*k)
        sx, cx = np.sin(n*x), np.cos(n*x)
        sy, cy = np.sin(k*y), np.cos(k*y)
        f += a*sx*sy + b*sx*cy + c*cx*sy + d*cx*cy
        fx += n*(a*cx*sy + b*cx*cy - c*sx*sy - d*sx*cy)
        fy += k*(a*sx*cy - b*sx*sy + c*cx*cy - d*cx*sy)

# Only the first-order requests are listed; values will be added automatically as a dependency.
derivatives = ['x', 'y']
schema = table(derivatives, variables='xy')
samples = np.zeros((len(schema), 2, x.size), dtype=np.float32)
for row, (order, dx, dy) in enumerate(schema):
    if order == 0:
        samples[row] = [x, y]
    elif dx == 1 and dy == 0:
        samples[row, 0] = 1  # dx/dx = 1.
    elif dx == 0 and dy == 1:
        samples[row, 1] = 1  # dy/dy = 1.

with Session(os.environ['CPDE_SERVER']) as session:
    inputs = Data(samples, schema, session=session).name_variables('xy')
    net = Network([2, 128, 128, 128, 128, 1], session=session).name_outputs('u')
    cost = net.cost(
        inputs=inputs,
        expression="""
            (u - f)**2 / s**2
            + (u_x - fx)**2 / sx**2
            + (u_y - fy)**2 / sy**2
        """,
        parameters={'f': f, 'fx': fx, 'fy': fy,
                    's': f.std(), 'sx': fx.std(), 'sy': fy.std()},
        name='value and first-derivative matching',
    )
    net.train(cost=cost, epochs=1500, optimizer='rprop', reset_steps=2e-4,
              verbose=2, log_every=100)

    print('Derivative table (order, x count, y count):')
    print(schema)
    prediction = net.eval(inputs)
    values = prediction.to_numpy()  # Shape: (derivative row, output, sample).
    print(values)
    prediction.close()
