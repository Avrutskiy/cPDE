"""Shape derivative of an integral over the network-defined ellipse {u > 0}."""
import os
import numpy as np
from cpde import Data, Network, Quadrature, Session, table

# Two spatial variables; the network's input features are x**2 and y**2.
# Only the first-order requests are listed; values will be added automatically as a dependency.
derivatives = ['x', 'y']
schema = table(derivatives, variables='xy')

with Session(os.environ['CPDE_SERVER']) as session:
    # u(x,y) = a*x**2 + b*y**2 + c. Its zero contour moves with the weights.
    net = Network([2, 1], session=session).name_outputs('u')
    state = net.state_dict()
    state['weights'][0][:] = [-1.0, -2.0, 1.0]
    net.load_state_dict(state)
    a, b, c = map(float, net.state_dict()['weights'][0])
    axes = np.sqrt(-c/np.array([a, b]))

    # Quadrature on the current ellipse: midpoint samples in squared radius.
    angle = 2*np.pi*np.arange(64)/64
    radius = np.sqrt((np.arange(8) + 0.5)/8)
    circle = np.column_stack((np.cos(angle), np.sin(angle)))
    points = (radius[:, None, None]*circle[None, :, :]*axes).reshape(-1, 2)
    # A scalar weight is applied to every point; their sum is the ellipse area.
    interior = Quadrature(
        points=points, weights=np.pi*np.prod(axes)/len(points),
        variables='xy', dimension=2,
    )
    edge = circle*axes
    speed = np.sqrt((axes[0]*np.sin(angle))**2 + (axes[1]*np.cos(angle))**2)
    normal = -np.column_stack((2*a*edge[:, 0], 2*b*edge[:, 1]))
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    # Here weights is a vector: arc-length weights vary along the ellipse.
    boundary = Quadrature(
        points=edge, weights=speed*(2*np.pi/64), N=normal,
        variables='xy', dimension=1,
    )

    interior_samples = np.zeros((len(schema), 2, len(points)), dtype=np.float32)
    boundary_samples = np.zeros((len(schema), 2, len(edge)), dtype=np.float32)
    for row, (order, dx, dy) in enumerate(schema):
        if order == 0:
            interior_samples[row] = (points**2).transpose()
            boundary_samples[row] = (edge**2).transpose()
        elif dx == 1:
            interior_samples[row, 0] = 2*points[:, 0]
            boundary_samples[row, 0] = 2*edge[:, 0]
        elif dy == 1:
            interior_samples[row, 1] = 2*points[:, 1]
            boundary_samples[row, 1] = 2*edge[:, 1]
    interior_inputs = Data(interior_samples, schema, session=session).name_variables('xy')
    boundary_inputs = Data(boundary_samples, schema, session=session).name_variables('xy')

    energy = net.cost(
        inputs=interior_inputs,
        quadrature=interior,
        expression='1 + 0.5*(u_x**2 + u_y**2)',
        name='energy over u > 0',
    )
    boundary_motion = net.cost(
        inputs=boundary_inputs,
        quadrature=boundary,
        expression='''
            (1 + 0.5*(u_x**2 + u_y**2))
            * u / sqrt(u_x**2 + u_y**2)
        ''',
        freeze=['u_x', 'u_y'],
        name='moving boundary variation',
    )
    cost = energy.with_variation(boundary_motion)

    # For u>0, outward normal speed is delta(u)/|grad(u)|. The attachment
    # contributes that boundary term while leaving the reported energy unchanged.
    fixed = energy.evaluate(gradient=True)
    moving = cost.evaluate(gradient=True)
    print('Energy:', moving['value'])
    print('Gradient [a, b, c] with geometry held fixed:', fixed['gradient'][0])
    print('Gradient [a, b, c] including boundary motion:', moving['gradient'][0])
    # This demonstrates one shape derivative. Rebuild geometry from the new
    # weights before evaluating at another shape; no moving-mesh training here.
