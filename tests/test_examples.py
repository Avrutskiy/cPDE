"""Mathematical checks for the 2D examples (host probe or CPDE_TEST_EXECUTABLE)."""
import os
import contextlib
import io
import runpy
from unittest.mock import patch
import tempfile
import unittest
import numpy as np
from cpde import Network, Quadrature, Session, table as derivative_table
from cpde._expressions import Expression
from cpde._prepare import Polynomial
from example_reference import coordinates, disk_rule, table, fourier
from build_host_probe import build


class ExampleMathTests(unittest.TestCase):
    def test_fourier_all_mixed_derivatives_through_fifth_order(self):
        points = np.array([[-.8, .3], [.1, -.4], [.73, .67]])
        schema = table(5)
        for combination in range(4):
            coefficients = np.zeros((4, 10, 10))
            coefficients[combination, 2, 3] = .7
            actual = fourier(points, schema, coefficients)
            expected = []
            for _, dx, dy in schema:
                fx = np.sin if combination < 2 else np.cos
                fy = np.sin if combination % 2 == 0 else np.cos
                expected.append(.7*3.0**(dx-1)*4.0**(dy-1)
                                * fx(3*points[:, 0]+dx*np.pi/2)
                                * fy(4*points[:, 1]+dy*np.pi/2))
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
        self.assertEqual(len(schema), 21)

    def test_disk_measures_and_second_moments(self):
        points, weights = disk_rule(12, 64)
        self.assertAlmostEqual(weights.sum(), np.pi, places=12)
        self.assertAlmostEqual(np.dot(weights, points[:, 0]**2), np.pi/4, places=12)
        self.assertAlmostEqual(np.dot(weights, points[:, 1]**2), np.pi/4, places=12)

    def test_multiline_expressions_preserve_derivative_namespace(self):
        multiline = Expression('''
            sqrt(1 + u_x**2 + u_y**2)
            - contactcos
            + gravity * u**2 / 2
        ''', 'xy', 'u', parameters={'contactcos': .5, 'gravity': 1})
        single = Expression('sqrt(1+u_x**2+u_y**2)-contactcos+gravity*u**2/2',
                            'xy', 'u', parameters={'contactcos': .5, 'gravity': 1})
        self.assertEqual(multiline.nodes, single.nodes)
        self.assertEqual(multiline.constants, single.constants)
        for name in ('contact_cos', 'u_x'):
            with self.assertRaises(ValueError):
                Expression('u', 'xy', 'u', parameters={name: 1})
        for expression in ('u; u', 'u\n__import__("os")', 'u, u'):
            with self.assertRaises(ValueError):
                Expression(expression, 'xy', 'u')


class DerivativeTableTests(unittest.TestCase):
    def test_requested_third_derivatives_include_all_dependencies(self):
        schema = derivative_table(['xxx', 'xxy', 'xyy', 'yyy'], variables='xy')
        expected = {(n, dx, n-dx) for n in range(4) for dx in range(n+1)}
        self.assertEqual(set(map(tuple, schema)), expected)
        self.assertEqual(len(schema), 10)
        # Separate pure derivatives do not need the unrelated mixed derivative.
        pure = derivative_table(['xx', 'yy'], variables='xy')
        self.assertEqual(set(map(tuple, pure)), {(0,0,0), (1,1,0), (1,0,1), (2,2,0), (2,0,2)})

    def test_derivative_table_aliases_and_validation(self):
        np.testing.assert_array_equal(derivative_table(['xy', 'yx'], variables='xy'),
                                      derivative_table(['xy'], variables='xy'))
        np.testing.assert_array_equal(derivative_table([], variables='xy'), [[0,0,0]])
        for request in (3, 'xxx', ['z'], ['xxxxxxx'], [None]):
            with self.subTest(request=request), self.assertRaises(ValueError):
                derivative_table(request, variables='xy')
        for variables in ('', 'xx', 'x1', None):
            with self.subTest(variables=variables), self.assertRaises(ValueError):
                derivative_table(['x'], variables=variables)
        self.assertFalse(derivative_table(['x'], variables='xy').flags.writeable)


class ExampleRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.executable = os.environ.get('CPDE_TEST_EXECUTABLE') or str(build(cls.temp.name))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.session = Session(self.executable)
        self.addCleanup(self.session.close)

    def test_multiline_legacy_equation_and_reserved_underscore(self):
        data = coordinates(np.array([[.2, .4], [-.3, .7]]), table(2), self.session)
        parser = Polynomial(data, 'u', {})
        a = parser.parse('''
            u_xx
            + u_yy
        ''')
        b = parser.parse('u_xx+u_yy')
        self.assertEqual(set(a), set(b))
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
        with self.assertRaises(ValueError):
            Polynomial(data, 'u', {'contact_cos': .5})

    def test_exact_poisson_solution_and_separate_boundary_cost(self):
        from cpde import Data
        # Test fixture supplies the analytical solution and its derivatives directly.
        # The example itself trains a network on the original x,y coordinates.
        def exact_data(points):
            x, y = points.T
            samples = np.array([(x*x+y*y-1)/4, y/2, x/2,
                                np.full(len(x), .5), np.zeros(len(x)), np.full(len(x), .5)])
            return Data(samples[:, None, :], table(2), session=self.session).name_variables('xy')
        points, weights = disk_rule(5, 24)
        q = Quadrature(points=points, weights=weights, variables='xy', dimension=2)
        angles = 2*np.pi*np.arange(32)/32
        edge = np.column_stack((np.cos(angles), np.sin(angles)))
        tangent = np.column_stack((-np.sin(angles), np.cos(angles)))
        boundary = Quadrature(points=edge, weights=np.full(32, 2*np.pi/32), T=tangent, variables='xy', dimension=1)
        net = Network([1, 1], session=self.session).name_outputs('u')
        state = net.state_dict()
        state['weights'][0][:] = [1, 0]
        net.load_state_dict(state)
        equation = net.cost(inputs=exact_data(points), quadrature=q, expression='(u_xx+u_yy-1)**2')
        endpoints = net.cost(inputs=exact_data(edge), quadrature=boundary, expression='u**2')
        tangent_cost = net.cost(inputs=exact_data(edge), quadrature=boundary,
                                expression='(u_x*T.x+u_y*T.y)**2')
        self.assertAlmostEqual((equation+endpoints+tangent_cost).evaluate(), 0, places=12)
        state['weights'][0][-1] = .1
        net.load_state_dict(state)
        self.assertAlmostEqual(equation.evaluate(), 0, places=12)
        self.assertAlmostEqual(endpoints.evaluate(), .01*2*np.pi, places=7)
        self.assertAlmostEqual(tangent_cost.evaluate(), 0, places=12)
        # A nonconstant boundary field must be penalized; check its weight gradient.
        linear = Network([2, 1], session=self.session).name_outputs('u')
        state = linear.state_dict(); state['weights'][0][:] = [.3, -.4, .2]
        linear.load_state_dict(state)
        linear_tangent = linear.cost(quadrature=boundary, expression='(u_x*T.x+u_y*T.y)**2')
        result = linear_tangent.evaluate(gradient=True)
        self.assertAlmostEqual(result['value'], np.pi*(.3**2+.4**2), places=6)
        np.testing.assert_allclose(result['gradient'][0], [2*np.pi*.3, -2*np.pi*.4, 0], rtol=1e-6, atol=1e-6)

    def load_example(self, name, *, training_calls=1):
        # Execute the actual standalone script, capturing its training call.
        # Keep this test's session open so its cost and arrays can be checked.
        with patch.dict(os.environ, {'CPDE_SERVER': self.executable}), \
             patch('cpde.Session', return_value=contextlib.nullcontext(self.session)), \
             patch.object(Network, 'train', autospec=True, return_value={}) as training, \
             contextlib.redirect_stdout(io.StringIO()):
            scope = runpy.run_module('examples.'+name, run_name='__main__')
        self.assertEqual(training.call_count, training_calls)
        if training_calls:
            self.assertIs(training.call_args.kwargs['cost'], scope['cost'])
        return scope

    def test_attached_variation_includes_motion_of_network_defined_ellipse(self):
        scope = self.load_example('attached_variation', training_calls=0)
        parameters = np.array([scope['a'],scope['b'],scope['c']])

        def exact_integral(weights):
            a,b,c = weights
            area = np.pi*c/np.sqrt(a*b)
            return area*(1-.5*c*(a+b))

        fixed, moving = scope['fixed'], scope['moving']
        self.assertEqual(moving['value'],fixed['value'])
        np.testing.assert_allclose(moving['value'],exact_integral(parameters),rtol=1e-7)
        # Each perturbation changes the ellipse as well as the integrand.
        step = 1e-5
        expected = np.array([(exact_integral(parameters+step*d)-exact_integral(parameters-step*d))/(2*step)
                             for d in np.eye(3)])
        np.testing.assert_allclose(moving['gradient'][0],expected,rtol=3e-6,atol=1e-6)
        self.assertEqual(fixed['gradient'][0][-1],0)
        self.assertGreater(moving['gradient'][0][-1],0)
        prediction = scope['net'].eval(scope['boundary_inputs'])
        np.testing.assert_allclose(prediction.to_numpy()[0],0,atol=2e-7)
        prediction.close()

    def test_fourier_array_matching_value_and_bias_gradient(self):
        scope = self.load_example('fit_function')
        net, inputs, targets, cost = (scope[k] for k in ('net', 'inputs', 'targets', 'cost'))
        points = np.column_stack((scope['x'], scope['y']))
        reference = fourier(points, scope['schema'], scope['coefficients'])
        np.testing.assert_allclose(scope['target_values'][:, 0], reference, rtol=1e-12, atol=1e-9)
        self.assertEqual(net.layers, [2, 128, 128, 128, 128, 1])
        self.assertEqual(inputs.sample_count, 729)
        actual = cost.evaluate(gradient=True)
        prediction = net.eval(inputs)
        errors = prediction.to_numpy()[:, 0].astype(np.float64)-targets.to_numpy()[:, 0]
        prediction.close()
        expected = np.sum(scope['weights'][0, :, None]*errors**2)
        self.assertAlmostEqual(actual['value']/expected, 1, places=6)
        bias_gradient = 2*scope['weights'][0, 0]*errors[0].sum()
        np.testing.assert_allclose(actual['gradient'][-1][-1], bias_gradient, rtol=2e-5, atol=2e-5)

    def test_explicit_first_derivative_cost_matches_bulk_cost(self):
        from cpde import Data
        scope = self.load_example('fit_function_explicit')
        net, inputs, explicit = (scope[k] for k in ('net', 'inputs', 'cost'))
        rows = {(0, 0): scope['f'], (1, 0): scope['fx'], (0, 1): scope['fy']}
        values = np.array([rows[(dx, dy)] for _, dx, dy in scope['schema']])
        reference = fourier(np.column_stack((scope['x'], scope['y'])),
                            scope['schema'], scope['coefficients'])
        np.testing.assert_allclose(values, reference, rtol=1e-12, atol=1e-12)
        targets = Data(values[:, None, :], scope['schema'], session=self.session).name_variables('xy')
        bulk = net.cost(inputs=inputs, targets=targets, weights=(1/values.std(axis=1)**2)[None, :])
        actual, expected = explicit.evaluate(gradient=True), bulk.evaluate(gradient=True)
        np.testing.assert_allclose(actual['value'], expected['value'], rtol=2e-6)
        for a, b in zip(actual['gradient'], expected['gradient']):
            np.testing.assert_allclose(a, b, rtol=2e-5, atol=2e-5)

    def test_poisson_script_geometry(self):
        try: import scipy.spatial
        except ImportError: self.skipTest('optional geometry extra is not installed')
        scope = self.load_example('poisson_disk')
        interior, boundary = scope['interior'], scope['boundary']
        # Straight triangles approximate the disk from inside.
        self.assertLess(interior.weights.sum(), np.pi)
        self.assertGreater(interior.weights.sum(), .999*np.pi)
        np.testing.assert_allclose(interior.points, interior.vertices[interior.cells].mean(axis=1))
        np.testing.assert_allclose(scope['interior_samples'][0], interior.points.T, atol=3e-8)
        np.testing.assert_array_equal(np.unique(interior.cells), np.arange(len(interior.vertices)))
        self.assertAlmostEqual(boundary.weights.sum(), 2*np.pi, places=12)
        np.testing.assert_allclose(np.linalg.norm(boundary.T, axis=1), 1, atol=1e-14)
        np.testing.assert_allclose(np.sum(boundary.T*boundary.points, axis=1), 0, atol=1e-14)
        np.testing.assert_allclose(boundary.T, np.column_stack((-boundary.points[:, 1], boundary.points[:, 0])))
        net = scope['net']
        separate = net.cost(inputs=scope['boundary_inputs'], quadrature=boundary, expression='u**2') + net.cost(
            inputs=scope['boundary_inputs'], quadrature=boundary, expression='0.1*(u_x*T.x + u_y*T.y)**2') + net.cost(
            inputs=scope['boundary_inputs'], quadrature=boundary,
            expression='0.1*(u_xx*T.x**2 + 2*u_xy*T.x*T.y + u_yy*T.y**2 - u_x*N.x - u_y*N.y)**2')
        combined = scope['boundary_cost'].evaluate(gradient=True)
        split = separate.evaluate(gradient=True)
        self.assertAlmostEqual(combined['value'], split['value'], places=6)
        for a,b in zip(combined['gradient'], split['gradient']):
            np.testing.assert_allclose(a,b,rtol=2e-5,atol=2e-6)


if __name__ == '__main__':
    unittest.main()
