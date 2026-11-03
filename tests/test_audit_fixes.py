"""Regressions for the selected November CUDA audit fixes."""
import contextlib
import copy
import io
import os
import tempfile
import unittest
import warnings
from unittest.mock import Mock, patch
import numpy as np
from cpde import Data, Network, Quadrature, Session
from cpde import _prepare as prep
from build_host_probe import build


class AuditFixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.executable = os.environ.get('CPDE_TEST_EXECUTABLE') or str(build(cls.temp.name))

    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()

    def setUp(self):
        self.session = Session(self.executable)
        self.addCleanup(self.session.close)

    def linear(self):
        net = Network([1, 1], session=self.session, seed=17).name_outputs('u')
        state = net.state_dict(); state['weights'][0][:] = [.5, 0]
        net.load_state_dict(state)
        q = Quadrature(points=[[1]], weights=1, variables='x', dimension=1)
        return net, q

    def train(self, net, cost, **kw):
        with contextlib.redirect_stdout(io.StringIO()):
            net.train(cost=cost, epochs=1, verbose=0, **kw)

    def test_wrapper_initialization_seed_layout_and_optimizer_buffers(self):
        rng = np.random.default_rng(42)
        net = Network([2, 3, 1], session=self.session, seed=42)
        state = net.state_dict()
        for index, (w, h) in enumerate(zip(net.layers, net.layers[1:])):
            expected = np.r_[rng.uniform(-2/np.sqrt(w), 2/np.sqrt(w), h*w), rng.uniform(-.1, .1, h)].astype('f4')
            np.testing.assert_array_equal(state['weights'][index], expected)
            for gradient in state['gradient_history'][index].values(): np.testing.assert_array_equal(gradient, 0)
            np.testing.assert_array_equal(state['optimizer_state'][index]['steps'], np.float32(.002))
        same = Network([2, 3, 1], session=self.session, seed=42).state_dict()
        for a, b in zip(state['weights'], same['weights']): np.testing.assert_array_equal(a, b)

    def test_explicit_steps_clear_history_before_sign_reversal(self):
        for sign in (-1, 1):
            for replacement in (.01, 100):
                net, q = self.linear()
                state = net.state_dict(); state['weights'][0][0] *= sign; net.load_state_dict(state)
                cost = net.cost(quadrature=q, expression='(u-target)**2', parameters={'target':sign})
                self.train(net, cost, optimizer='rprop', reset_steps=.1)
                before = net.state_dict()['weights'][0].copy()
                cost.set_parameters(target=-sign)
                self.train(net, cost, reset_steps=replacement)
                after = net.state_dict()
                expected = before-sign*replacement if replacement < 1 else before
                np.testing.assert_allclose(after['weights'][0], expected, rtol=0, atol=1e-7)
                if replacement == 100:
                    np.testing.assert_array_equal(after['gradient_history'][0]['previous'], 0)
                    np.testing.assert_array_equal(after['optimizer_state'][0]['steps'], np.float32(.1))

    def test_cancelled_attached_variation_keeps_value(self):
        net, q = self.linear()
        a = net.cost(quadrature=q, expression='u')
        cancelled = a.with_variation(-a)
        self.assertEqual(cancelled.evaluate(), .5)
        result = cancelled.evaluate(gradient=True)
        self.assertEqual(result['value'], .5)
        for gradient in result['gradient']: np.testing.assert_array_equal(gradient, 0)

    def test_division_gradients_with_large_and_small_common_scales(self):
        net, q = self.linear()
        for scale in ('1', '1e200', '1e-200'):
            numerator = net.cost(quadrature=q, expression=f'{scale}*u')
            denominator = net.cost(quadrature=q, expression=f'{scale}*(u+1)')
            inline = net.cost(quadrature=q, expression=f'{scale}*u/({scale}*(u+1))')
            for label, cost in [('graph', numerator/denominator), ('expression', inline)]:
                with self.subTest(scale=scale, path=label):
                    result = cost.evaluate(gradient=True)
                    self.assertAlmostEqual(result['value'], 1/3, places=14)
                    np.testing.assert_allclose(result['gradient'][0], [4/9, 4/9], rtol=2e-6)
            numerator.close(); denominator.close(); inline.close()

    def test_invalid_state_and_float32_options_fail_before_mutation(self):
        net, q = self.linear(); cost = net.cost(quadrature=q, expression='u')
        for optimizer, key in [('adam', 'v'), ('rprop', 'steps')]:
            self.train(net, cost, optimizer=optimizer, state='reset')
            original = net.state_dict(); bad = copy.deepcopy(original)
            bad['optimizer_state'][0][key][:] = -1
            with self.assertRaises(ValueError): net.load_state_dict(bad)
            np.testing.assert_array_equal(net.state_dict()['weights'][0], original['weights'][0])
        for kw in [dict(reset_steps=1e-50), dict(learning_rate=1e-50)]:
            with self.assertRaises(ValueError): self.train(net, cost, **kw)

    def test_dimension_rejection_precedes_rng_or_server_allocation(self):
        session = Mock()
        with patch('numpy.random.default_rng') as rng:
            for layers in ([65536, 65536], [32768, 32768], [1]*257):
                with self.assertRaises(ValueError): Network(layers, session=session)
            rng.assert_not_called(); session._request.assert_not_called()
        for dims in [(65536, 65536), (2, 2**30)]:
            with self.assertRaises(ValueError): prep.checked_product(*dims)
        with self.assertRaises(ValueError): prep.check_context([1, 100000], 100, 1000)

    def test_alternating_forward_gradient_chunks_and_frozen_costs(self):
        net = Network([1, 4, 3, 1], session=self.session, seed=4).name_outputs('u')
        q = Quadrature(points=np.linspace(-1, 1, 13)[:, None], weights=1/13, variables='x', dimension=1)
        cost = net.cost(quadrature=q, expression='u**2+u_xx**2', chunk_size=5)
        expected = cost.evaluate(gradient=True)
        for chunk in [1, 13, 3, 8, 2]:
            result = net.eval(q, derivatives=['', 'x', 'xx'], chunk_size=chunk)
            self.assertTrue(np.isfinite(result.to_numpy()).all())
            result.close()
            got = cost.evaluate(gradient=True)
            self.assertAlmostEqual(got['value'], expected['value'], places=7)
            for a, b in zip(got['gradient'], expected['gradient']): np.testing.assert_allclose(a, b, atol=2e-7)
        frozen = net.cost(quadrature=q, expression='u**2+u_xx**2', freeze=['u', 'u_xx'], chunk_size=5)
        result = frozen.evaluate(gradient=True)
        self.assertAlmostEqual(result['value'], expected['value'], places=7)
        for gradient in result['gradient']: np.testing.assert_array_equal(gradient, 0)
        constant = net.cost(quadrature=q, expression='2')
        self.assertAlmostEqual(constant.evaluate(gradient=True)['value'], 2)

    def test_cache_warnings_and_inspection_rebind(self):
        self.session._cache_warning_threshold = 3
        net, q = self.linear()
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter('always')
            cost = net.cost(quadrature=q, expression='scale*u', parameters={'scale':1})
            first = cost.inspect()
            for value in range(2, 8): cost.set_parameters(scale=value)
            self.assertFalse(seen)
            q.replace(points=[[1], [2]], weights=.5)
            old_id = cost._id
            self.assertEqual(cost.inspect()['source'], first['source'])
            self.assertNotEqual(cost._id, old_id)
            for literal in range(2, 7): net.cost(quadrature=q, expression=f'{literal}*u')
            self.assertEqual(sum('program' in str(w.message) for w in seen), 1)
            data = Data(np.ones((1, 1, 9), np.float32), [[0, 0]], session=self.session).name_variables('x')
            for width in range(1, 8): net.eval(data, chunk_size=width).close()
            self.assertEqual(sum('contexts' in str(w.message) for w in seen), 1)
            self.assertEqual(len(net._context_keys), 3)

    def test_inspection_refreshes_empty_to_nonempty_derivative_mapping(self):
        net, _ = self.linear()
        q = Quadrature(points=np.empty((0, 1)), weights=1, variables='x', dimension=1)
        cost = net.cost(quadrature=q, expression='u_x')
        empty_source = cost.inspect()['source']
        q.replace(points=[[1]], weights=1)
        populated_source = cost.inspect()['source']
        self.assertNotEqual(populated_source, empty_source)
        self.assertAlmostEqual(cost.evaluate(), .5)
        self.assertEqual(cost.inspect()['source'], populated_source)


if __name__ == '__main__': unittest.main()
