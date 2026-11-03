"""Solve u_xx + u_yy = 1 in a disk, with residual derivatives and curved-wall derivative costs."""
import os
import numpy as np
from cpde import Data, Network, Quadrature, Session, table

# Only the required derivatives are listed; lower derivatives and values will be added automatically as dependencies.
interior_derivatives = ['xxx', 'xxy', 'xyy', 'yyy']
boundary_derivatives = ['xx', 'xy', 'yy']
interior_schema = table(interior_derivatives, variables='xy')
boundary_schema = table(boundary_derivatives, variables='xy')

# Random interior vertices and a uniform boundary, triangulated together.
rng = np.random.default_rng(2026)
radius = np.sqrt(rng.uniform(0, 1, 900))
theta = rng.uniform(0, 2*np.pi, 900)
angle = 2*np.pi*np.arange(96)/96
edge = np.column_stack((np.cos(angle), np.sin(angle)))
vertices = np.vstack((np.column_stack((radius*np.cos(theta), radius*np.sin(theta))), edge))
# Triangulate the mesh and integrate at triangle centroids with area weights.
# Mesh vertices and triangle indices remain available as interior.vertices/cells.
interior = Quadrature(
    points=vertices,
    triangulation='delaunay',
    rule='centroid',
    variables='xy',
    dimension=2,
)
points = interior.points

# Unit circle: ds = dtheta, and the counterclockwise unit tangent is (-sin, cos).
tangent = np.column_stack((-np.sin(angle), np.cos(angle)))
# A scalar weight is applied to every point: 96 * (2*pi/96) = circumference.
boundary = Quadrature(points=edge, weights=2*np.pi/96,
                      N=edge, T=tangent, variables='xy', dimension=1)

interior_samples = np.zeros((len(interior_schema), 2, len(points)), dtype=np.float32)
for row, (order, dx, dy) in enumerate(interior_schema):
    if order == 0:
        interior_samples[row] = points.transpose()
    elif dx == 1 and dy == 0:
        interior_samples[row, 0] = 1
    elif dx == 0 and dy == 1:
        interior_samples[row, 1] = 1

boundary_samples = np.zeros((len(boundary_schema), 2, len(edge)), dtype=np.float32)
for row, (order, dx, dy) in enumerate(boundary_schema):
    if order == 0:
        boundary_samples[row] = edge.transpose()
    elif dx == 1 and dy == 0:
        boundary_samples[row, 0] = 1
    elif dx == 0 and dy == 1:
        boundary_samples[row, 1] = 1

with Session(os.environ['CPDE_SERVER']) as session:
    interior_inputs = Data(interior_samples, interior_schema, session=session).name_variables('xy')
    boundary_inputs = Data(boundary_samples, boundary_schema, session=session).name_variables('xy')
    net = Network([2, 64, 64, 64, 64, 64, 64, 1], session=session).name_outputs('u')
    equation = net.cost(
        inputs=interior_inputs,
        quadrature=interior,
        expression='''
            (u_xx + u_yy - 1)**2
            + 0.5*(u_xxx + u_xyy)**2
            + 0.5*(u_xxy + u_yyy)**2
        ''',
        name='Poisson residual and spatial derivatives',
    )
    boundary_cost = net.cost(
        inputs=boundary_inputs,
        quadrature=boundary,
        # Unit-circle curvature is 1: d²u/ds² = T^T H(u) T - grad(u) dot N.
        expression='''
            u**2
            + 0.1*(u_x*T.x + u_y*T.y)**2
            + 0.1*(u_xx*T.x**2 + 2*u_xy*T.x*T.y + u_yy*T.y**2
                   - u_x*N.x - u_y*N.y)**2
        ''',
        name='boundary values and first/second tangential derivatives',
    )
    # Each cost runs forward then backward in sequence, accumulating gradients.
    cost = equation + boundary_cost
    net.train(cost=cost, epochs=3000, optimizer='rprop', reset_steps=2e-4,
              verbose=2, log_every=100)

    # Inspect the interior output and its propagated derivatives directly.
    print('Interior derivative table (order, x count, y count):')
    print(interior_schema)
    prediction = net.eval(interior_inputs)
    values = prediction.to_numpy()
    print(values)
    prediction.close()

    # Inspect boundary values and the derivatives used in the tangent cost.
    print('Boundary derivative table (order, x count, y count):')
    print(boundary_schema)
    boundary_prediction = net.eval(boundary_inputs)
    boundary_values = boundary_prediction.to_numpy()
    print(boundary_values)
    boundary_prediction.close()
