"""Circle ground-state experiment: eliminate energy analytically, train 2000 epochs."""
import os
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from cpde import Data, Network, Quadrature, Session, table

# Lower derivatives and values will be added automatically as dependencies.
interior_derivatives = ['xxx', 'xyy', 'xxy', 'yyy']
boundary_derivatives = ['xx', 'xy', 'yy']
interior_schema = table(interior_derivatives, variables='xy')
boundary_schema = table(boundary_derivatives, variables='xy')

# Unit circle: 64 equally spaced boundary points, N=(x,y), curvature=1.
a, b = 1.0, 1.0
theta = np.arange(64)*2*np.pi/64
edge = np.column_stack((np.cos(theta), np.sin(theta)))
normal = edge.copy()
tangent = np.column_stack((-np.sin(theta), np.cos(theta)))
curvature = np.ones(len(edge))
boundary_weights = np.full(len(edge), 2*np.pi/len(edge))
boundary = Quadrature(points=edge, weights=boundary_weights, N=normal, T=tangent,
                      variables='xy', dimension=1)

# A 21x21 lattice gives 305 interior points, plus all 64 boundary points in the residual grid.
X, Y = np.meshgrid(np.linspace(edge[:,0].min(), edge[:,0].max(), 21),
                   np.linspace(edge[:,1].min(), edge[:,1].max(), 21), indexing='ij')
candidates = np.column_stack((X.ravel(), Y.ravel()))
hull = ConvexHull(edge)
inside = np.all(candidates @ hull.equations[:,:2].transpose() + hull.equations[:,2] < -1e-10, axis=1)
bulk_points = candidates[inside]
points = np.vstack((bulk_points, edge))
triangles = Delaunay(points).simplices
corners = points[triangles]
triangle_area = 0.5*np.abs((corners[:,1,0]-corners[:,0,0])*(corners[:,2,1]-corners[:,0,1])
                         -(corners[:,1,1]-corners[:,0,1])*(corners[:,2,0]-corners[:,0,0]))
# Vertex quadrature: each triangle contributes one third of its area to each vertex.
# Boundary residual samples therefore receive area weights, not arc-length weights.
area_weights = np.zeros(len(points))
np.add.at(area_weights, triangles.ravel(), np.repeat(triangle_area/3, 3))
interior = Quadrature(points=points, weights=area_weights, variables='xy', dimension=2)

interior_samples = np.zeros((len(interior_schema), 2, len(points)), dtype=np.float32)
for row, (order, dx, dy) in enumerate(interior_schema):
    if order == 0:
        interior_samples[row] = points.transpose()
    elif order == 1:
        interior_samples[row] = np.array([dx, dy])[:, None]
boundary_samples = np.zeros((len(boundary_schema), 2, len(edge)), dtype=np.float32)
for row, (order, dx, dy) in enumerate(boundary_schema):
    if order == 0:
        boundary_samples[row] = edge.transpose()
    elif order == 1:
        boundary_samples[row] = np.array([dx, dy])[:, None]

with Session(os.environ['CPDE_SERVER']) as session:
    interior_inputs = Data(interior_samples, interior_schema, session=session).name_variables('xy')
    boundary_inputs = Data(boundary_samples, boundary_schema, session=session).name_variables('xy')
    net = Network([2, 128, 128, 128, 1], session=session).name_outputs('u')
    # h=-Delta(u). The residual loss is A - 2*E*B + E**2*C.
    # For nonzero C its exact minimum over E is A - B**2/C.
    A = net.cost(
        inputs=interior_inputs,
        quadrature=interior,
        expression='''
            (u_xx + u_yy)**2
            + 0.5*(u_xxx + u_xyy)**2
            + 0.5*(u_xxy + u_yyy)**2
        ''',
        name='projection A',
    )
    B = net.cost(
        inputs=interior_inputs,
        quadrature=interior,
        expression='''
            -u*(u_xx + u_yy)
            - 0.5*u_x*(u_xxx + u_xyy)
            - 0.5*u_y*(u_xxy + u_yyy)
        ''',
        name='projection B',
    )
    C = net.cost(
        inputs=interior_inputs,
        quadrature=interior,
        expression='''
            u**2 + 0.5*(u_x**2 + u_y**2)
        ''',
        name='projection C',
    )
    mass = net.cost(
        inputs=interior_inputs,
        quadrature=interior,
        expression='u**2',
        name='normalization integral',
    )
    walls = net.cost(
        inputs=boundary_inputs,
        quadrature=boundary,
        # d²u/ds² = T^T H(u) T - curvature*(grad(u) dot N), for outward N.
        expression='''
            u**2
            + 0.1*(u_x*T.x + u_y*T.y)**2
            + 0.1*(
                u_xx*T.x**2 + 2*u_xy*T.x*T.y + u_yy*T.y**2
                - curvature*(u_x*N.x + u_y*N.y)
            )**2
        ''',
        parameters={'curvature': curvature},
        name='wall values and first/second tangential derivatives',
    )
    # All integrals participate in the compound-cost gradient each epoch.
    energy = B / C
    cost = A - B**2 / C + 10 * (mass - 1)**2 + walls
    net.train(
        cost=cost,
        epochs=2000,
        optimizer='rprop',
        state='reset',
        reset_steps=2e-4,
        restore_every=1000,
        restore_step=1e-5,
        verbose=2,
        log_every=100,
    )

    E = energy.evaluate()
    print(f'Final projected energy: {E:.9g}')

    print('Domain residual derivative table (order, x count, y count):')
    print(interior_schema)
    prediction = net.eval(interior_inputs)
    print(prediction.to_numpy())
    prediction.close()

    print('Boundary derivative table (order, x count, y count):')
    print(boundary_schema)
    boundary_prediction = net.eval(boundary_inputs)
    print(boundary_prediction.to_numpy())
    boundary_prediction.close()
