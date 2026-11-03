import contextlib
import copy
import io
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
import numpy as np
from cpde import Session, Data, Network
from cpde._wire import Reader
from build_host_probe import build


class InterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.executable = os.environ.get('CPDE_TEST_EXECUTABLE') or str(build(cls.temp.name))

    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()

    def setUp(self):
        self.session = Session(self.executable)
        self.addCleanup(self.session.close)

    def data(self, samples, table, names='x'):
        return Data(np.asarray(samples), np.asarray(table, np.int32), session=self.session).name_variables(names)

    def linear(self, a=.4, b=-.2):
        net = Network([1, 1], session=self.session).name_outputs('u')
        state = net.state_dict(); state['weights'][0][:] = [a, b]; net.load_state_dict(state)
        return net

    def train(self, net, **kw):
        kw.setdefault('verbose', 0)
        kw.setdefault('log_every', 1)
        with contextlib.redirect_stdout(io.StringIO()): return net.train(**kw)

    def test_evaluation_ownership_order_host_edits_and_chaining(self):
        table = [[2, 2], [0, 0], [1, 1]]
        x = np.array([[[.3, .4, .2]], [[1, 2, 3]], [[.8, .2, .5]]], np.float32)
        data = self.data(x, table); net = self.linear(2, 3)
        prediction = net.eval(data, chunk_size=2)
        expected = 2*x; expected[1] += 3
        np.testing.assert_allclose(prediction.to_numpy(), expected)
        net.eval(self.data(x*2, table))
        np.testing.assert_array_equal(prediction.derivatives, data.derivatives)
        np.testing.assert_allclose(prediction.to_numpy(), expected)
        detached = prediction.to_numpy(); detached[:] = -5
        np.testing.assert_allclose(prediction.to_numpy(), expected)
        with prediction.edit_host() as a: a[1] += 1
        expected[1] += 1
        np.testing.assert_allclose(net.eval(prediction).to_numpy(), expected*2 + np.array([0, 3, 0])[:, None, None])
        path = Path(self.temp.name)/'data.npz'; prediction.save(path)
        restored = Data.load(path, session=self.session)
        np.testing.assert_allclose(restored.to_numpy(), expected)
        self.assertEqual(restored._identity, prediction._identity)

    def test_three_costs_total_and_common_gradient_padded(self):
        x = np.array([[-.7, -.2, .1, .4, .9], [1, .8, .6, .5, .2], [.2, -.3, .4, .2, -.1]], np.float32)
        d = self.data(x[:, None, :], [[0, 0], [1, 1], [2, 2]])
        targets = self.data(np.full((3, 1, 5), .1), [[0, 0], [1, 1], [2, 2]])
        net = self.linear(); z = .4*x; z[0] -= .2
        wm = np.array([[[.5, 0, 1, 2, .3], [.2, 1, 0, 1, .4], [0, .2, 1, 0, .5]]], np.float32)
        matching = net.cost(inputs=d, targets=targets, weights=wm, chunk_size=2)
        nu = np.array([.1, .2, .3, .4, .5], np.float32); src = np.linspace(-.1, .2, 5, dtype=np.float32)
        ew = np.array([.2, .5, 1, 0, 2], np.float32)
        eq = net.cost(inputs=d, equations={'a': 'u*u_x + nu*u_xx', 'b': 'u_x-u'},
                      parameters={'nu': nu}, rhs={'a': src}, weights={'a': ew, 'b': .3}, chunk_size=3)
        boundary_x = np.array([[.2, .5, .7], [1, .3, -.2]], np.float32)
        boundary = self.data(boundary_x[:, None, :], [[0, 0], [1, 1]])
        bv = np.array([[1, .2, .3], [.4, -.1, .5], [.8, .9, -.2]], np.float32)
        scalar = net.cost(inputs=boundary, vector_a='u, u_x, u_x', vector_b=bv, rhs=np.array([.1, .2, .3]), weights=.7, chunk_size=2)
        result = self.train(net, cost=matching+eq+scalar, epochs=1, optimizer='adam')
        gm = 2*wm[0]*(z-.1); lm = np.sum(wm[0]*(z-.1)**2)
        r1 = z[0]*z[1] + nu*z[2] - src; r2 = z[1]-z[0]
        ge = np.vstack([2*ew*r1*z[1]-.6*r2, 2*ew*r1*z[0]+.6*r2, 2*ew*r1*nu])
        le = np.sum(ew*r1*r1+.3*r2*r2)
        bz = .4*boundary_x; bz[0] -= .2
        rb = bv[:, 0]*bz[0]+(bv[:, 1]+bv[:, 2])*bz[1]-[.1, .2, .3]
        gb = np.vstack([1.4*rb*bv[:, 0], 1.4*rb*(bv[:, 1]+bv[:, 2])]); lb = np.sum(.7*rb*rb)
        expected = np.array([np.sum((gm+ge)*x)+np.sum(gb*boundary_x), np.sum(gm[0]+ge[0])+np.sum(gb[0])])
        state = net.state_dict()
        np.testing.assert_allclose(state['gradient_history'][0]['current'], expected, rtol=2e-6, atol=2e-6)
        np.testing.assert_allclose(state['optimizer_state'][0]['m'], .1*expected, rtol=2e-6, atol=2e-6)
        before = next(r for r in result['reports'] if r['phase']=='before_update')
        np.testing.assert_allclose(before['contributions'], [lm, le, lb], rtol=2e-6)
        self.assertAlmostEqual(before['total'], float(lm+le+lb), places=5)
        self.assertEqual(state['metadata']['adam_step'], 1)

    def test_full_vs_chunked_updates_both_optimizers(self):
        d = self.data(np.array([[[.2, -.4, .8, .3, .9]], [[1, .4, .6, .7, .8]]]), [[0, 0], [1, 1]])
        for optimizer in ('adam', 'rprop'):
            a = self.linear(); b = self.linear(); b.load_state_dict(a.state_dict())
            ca = a.cost(inputs=d, equations={'e': 'u*u_x-u'}, chunk_size=2)
            cb = b.cost(inputs=d, equations={'e': 'u*u_x-u'})
            self.train(a, cost=ca, epochs=4, optimizer=optimizer)
            self.train(b, cost=cb, epochs=4, optimizer=optimizer)
            np.testing.assert_allclose(a.state_dict()['weights'][0], b.state_dict()['weights'][0], rtol=1e-6, atol=1e-6)

    def test_multioutput_matching_and_target_row_mapping(self):
        d = self.data([[[.2, .3, .4]], [[1, .8, .6]]], [[0, 0], [1, 1]])
        n = Network([1, 3], session=self.session).name_outputs('uvw')
        st = n.state_dict(); st['weights'][0][:] = [.2, .3, .4, -.1, .2, -.3]; n.load_state_dict(st)
        y = n.eval(d).to_numpy()
        target = self.data(np.full((2, 3, 3), .15), [[1, 1], [0, 0]])
        w = np.array([[1, 2], [0, 3], [.4, .5]], np.float32)
        c = n.cost(inputs=d, targets=target, weights=w, chunk_size=2)
        result = self.train(n, cost=c, epochs=1, optimizer='adam')
        expected = np.sum(w.T[:, :, None]*(y-.15)**2)
        before = next(r for r in result['reports'] if r['phase']=='before_update')
        self.assertAlmostEqual(before['total'], float(expected), places=5)
        small = self.data([[[.5]]], [[0, 0]])
        fresh = Network([1, 4], session=self.session)
        self.assertEqual(fresh.eval(small).to_numpy().shape, (1, 4, 1))

    def test_continuation_reset_and_save_without_history(self):
        d = self.data([[[.2, -.4, .8, .3, .9]]], [[0, 0]])
        for method in ('rprop', 'adam'):
            n = self.linear(); reference = self.linear(); initial = n.state_dict(); reference.load_state_dict(initial)
            c = n.cost(inputs=d, equations={'value': 'u'}); cr = reference.cost(inputs=d, equations={'value': 'u'})
            self.train(n, cost=c, epochs=3, optimizer=method)
            path = Path(self.temp.name)/f'{method}.npz'; n.save(path)
            restored = Network([1, 1], session=self.session, state=path)
            restored_cost = restored.cost(inputs=d, equations={'value': 'u'})
            self.train(restored, cost=restored_cost, epochs=4)
            self.train(reference, cost=cr, epochs=7, optimizer=method)
            a, b = restored.state_dict(), reference.state_dict()
            np.testing.assert_array_equal(a['weights'][0], b['weights'][0])
            for k in a['optimizer_state'][0]: np.testing.assert_array_equal(a['optimizer_state'][0][k], b['optimizer_state'][0][k])
            n.save(path, include_history=False); restored.load(path)
            nohistory = restored.state_dict()
            np.testing.assert_array_equal(nohistory['gradient_history'][0]['current'], [0, 0])
            if method == 'rprop':
                np.testing.assert_array_equal(nohistory['gradient_history'][0]['previous'], [0, 0])
                old = n.state_dict()['optimizer_state'][0]['steps']
                np.testing.assert_array_equal(nohistory['optimizer_state'][0]['steps'], old)
            else:
                np.testing.assert_array_equal(nohistory['optimizer_state'][0]['m'], n.state_dict()['optimizer_state'][0]['m'])
                self.train(restored, cost=restored_cost, epochs=1, reset_moments=True)
                self.assertEqual(restored.state_dict()['metadata']['adam_step'], 1)
            self.train(restored, cost=restored_cost, epochs=1, optimizer='adam', state='reset')
            self.assertEqual(restored.state_dict()['metadata']['adam_step'], 1)

    def test_sparse_closure_zero_objective_and_weight_shapes(self):
        d = self.data([[[.2, .3, .4]], [[1, 1, 1]], [[.1, .2, .3]]], [[0, 0], [1, 1], [2, 2]])
        n = self.linear(); targets = n.eval(d)
        c = n.cost(inputs=d, targets=targets, weights=np.array([[1, 0, 0]]), chunk_size=2)
        r = Reader(c._semantic); r.get('i'); r.get('i'); np.testing.assert_array_equal(r.array('i4'), [0])
        before = n.state_dict()['weights'][0].copy(); self.train(n, cost=c, epochs=2)
        np.testing.assert_array_equal(before, n.state_dict()['weights'][0])
        high = n.cost(inputs=d, equations={'d': 'u_xx'})
        r = Reader(high._semantic); r.get('i'); r.get('i'); np.testing.assert_array_equal(r.array('i4'), [0, 1, 2])
        with self.assertRaises(ValueError): n.cost(inputs=d, targets=targets, weights=[1, 1, 1])
        with self.assertRaises(ValueError): n.cost(inputs=d, targets=targets, weights=[[-1, 0, 0]])

    def test_multivariate_names_parser_and_rejections(self):
        table = [[0, 0, 0], [1, 1, 0], [1, 0, 1], [2, 1, 1]]
        d = self.data(np.ones((4, 2, 3)), table, 'xy')
        n = Network([2, 2], session=self.session).name_outputs('uv')
        a = n.cost(inputs=d, equations={'e': 'u_xy + v_yx + u*v'})
        b = n.cost(inputs=d, equations={'e': 'u_yx + v_xy + v*u'})
        self.assertEqual(a._semantic, b._semantic)
        for expression in ('u*u*v', 'u+1', 'sin(u)', 'u_z', 'u=0', '__import__("os")'):
            with self.assertRaises(ValueError, msg=expression): n.cost(inputs=d, equations={'e': expression})
        with self.assertRaises(ValueError): n.cost(inputs=d, equations={'e': 'u'}, weights={'wrong': 1})
        with self.assertRaises(ValueError): n.cost(inputs=d, equations={'e': 'u'}, rhs={'e': 2})
        with self.assertRaises(ValueError): self.data(np.ones((2, 1, 2)), [[0, 0], [2, 2]])

    def test_nonlinear_weight_gradient_finite_difference(self):
        x = np.array([[[.2, -.3, .6], [.1, .4, -.2]], [[.7, .2, .5], [-.1, .5, .9]], [[.2, .1, -.3], [.6, .3, .1]]], np.float32)
        d = self.data(x, [[0, 0], [1, 1], [2, 2]])
        n = Network([2, 3, 2], session=self.session).name_outputs('uv')
        initial = n.state_dict(); rng = np.random.default_rng(12)
        for a in initial['weights']: a[:] = rng.uniform(-.4, .4, len(a))
        n.load_state_dict(initial)
        c = n.cost(inputs=d, equations={'a': 'u*v_x + u_xx', 'b': 'v*v + u_x'}, weights={'a': .7}, chunk_size=2)
        self.train(n, cost=c, epochs=1, optimizer='adam'); analytic = [h['current'].copy() for h in n.state_dict()['gradient_history']]
        def loss(state):
            n.load_state_dict(state); p = n.eval(d).to_numpy()
            return np.sum(.7*(p[0, 0]*p[1, 1]+p[2, 0])**2+(p[0, 1]**2+p[1, 0])**2, dtype=np.float64)
        eps = .002
        for layer, a in enumerate(initial['weights']):
            numeric = []
            for i in range(len(a)):
                plus, minus = copy.deepcopy(initial), copy.deepcopy(initial)
                plus['weights'][layer][i] += eps; minus['weights'][layer][i] -= eps
                numeric.append((loss(plus)-loss(minus))/(2*eps))
            np.testing.assert_allclose(analytic[layer], numeric, rtol=3e-3, atol=3e-5)

    def test_reporting_outputs_context_growth_and_checkpoint(self):
        d = self.data([[[.2, .3, .4]], [[1, 1, 1]], [[.1, .2, .3]]], [[0, 0], [1, 1], [2, 2]])
        n = self.linear(); c = n.cost(inputs=d, equations={'v': 'u'}, chunk_size=2)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            result = n.train(cost=c, epochs=2, verbose=2, eval_every=1, print_outputs=True, save_every=1)
        self.assertIn('outputs:', out.getvalue()); self.assertIn('total=', out.getvalue())
        self.assertEqual(result['completed'], 2)
        self.assertEqual(n.checkpoint['metadata']['updates'], 2)
        self.assertEqual(result['reports'][-1]['phase'], 'after_update')

    def test_stop_at_common_update_and_resume(self):
        d = self.data([[[.2, .3, .4]]], [[0, 0]])
        n = self.linear(); c = n.cost(inputs=d, equations={'v': 'u'}, chunk_size=2)
        def stop_after_checkpoint():
            until = time.monotonic()+5
            while n.checkpoint is None and time.monotonic()<until: time.sleep(.001)
            self.session.stop()
        thread = threading.Thread(target=stop_after_checkpoint); thread.start()
        result = self.train(n, cost=c, epochs=100000, optimizer='adam', save_every=1)
        thread.join(); self.assertTrue(result['stopped']); self.assertGreater(result['completed'], 0)
        state = n.state_dict(); self.assertEqual(state['metadata']['updates'], result['completed'])
        self.assertEqual(state['metadata']['remaining'], 100000-result['completed'])
        # A short remaining stage verifies resume without running the large stop-test budget.
        state['metadata']['remaining'] = 2; n.load_state_dict(state)
        with contextlib.redirect_stdout(io.StringIO()): resumed = n.resume(cost=c, verbose=0)
        self.assertEqual(resumed['completed'], 2)
        with d.edit_host() as a: a += 1
        state['metadata']['remaining'] = 2; n.load_state_dict(state)
        with self.assertRaises(ValueError): n.resume(cost=c)

    def test_gradient_diagnostics_and_reporting_do_not_change_updates(self):
        d = self.data(np.array([[[.2, -.4, .8, .3, .9]]]), [[0, 0]])
        n = Network([1, 3, 1], session=self.session).name_outputs('u')
        state = n.state_dict()
        rng = np.random.default_rng(9)
        for a in state['weights']: a[:] = rng.uniform(-.3, .3, len(a))
        n.load_state_dict(state)
        control = Network([1, 3, 1], session=self.session, state=state).name_outputs('u')
        c = n.cost(inputs=d, equations={'e': 'u'}, chunk_size=2)
        cc = control.cost(inputs=d, equations={'e': 'u'}, chunk_size=2)
        with contextlib.redirect_stdout(io.StringIO()):
            result = n.train(cost=c, epochs=5, optimizer='adam', log_every=0,
                             diagnostics_every=2, verbose=0)
            reference = control.train(cost=cc, epochs=5, optimizer='adam', log_every=1, verbose=0)
        self.assertEqual([r['epoch'] for r in result['gradient_reports']], [2, 4, 5])
        self.assertEqual([r['epoch'] for r in result['reports'] if r['phase']=='before_update'], [5])
        self.assertEqual(reference['gradient_reports'], [])
        actual, expected = n.state_dict(), control.state_dict()
        for a, b in zip(actual['weights'], expected['weights']): np.testing.assert_array_equal(a, b)
        for a, b in zip(actual['optimizer_state'], expected['optimizer_state']):
            for key in a: np.testing.assert_array_equal(a[key], b[key])
        report = result['gradient_reports'][-1]
        self.assertEqual(report, n.last_gradient_report)
        self.assertEqual(report['phase'], 'before_update')
        self.assertEqual(report['update'], 4)
        gradients = [h['current'].astype(np.float64) for h in actual['gradient_history']]
        flat = np.concatenate(gradients)
        np.testing.assert_allclose(report['l2_norm'], np.linalg.norm(flat), rtol=1e-12)
        self.assertEqual(report['max_abs'], np.abs(flat).max())
        self.assertEqual(report['nonfinite_count'], 0)
        for layer, gradient in zip(report['layers'], gradients):
            np.testing.assert_allclose(layer['l2_norm'], np.linalg.norm(gradient), rtol=1e-12)
            self.assertEqual(layer['max_abs'], np.abs(gradient).max())
            self.assertIsNone(layer['first_nonfinite_index'])

    def test_multiblock_cost_reductions_and_partial_chunk(self):
        x = np.linspace(-.8, .9, 2053, dtype=np.float32)
        d = self.data(np.stack((x, np.ones_like(x)))[:, None, :], [[0, 0], [1, 1]])
        a = self.linear(); b = self.linear(); b.load_state_dict(a.state_dict())
        target = self.data(np.zeros((2, 1, len(x)), np.float32), [[0, 0], [1, 1]])
        def costs(n, chunk):
            matching = n.cost(inputs=d, targets=target, weights=[[.3, .2]], chunk_size=chunk)
            polynomial = n.cost(inputs=d, equations={'p': 'u*u+u_x'}, chunk_size=chunk)
            return matching + polynomial + matching
        ra = self.train(a, cost=costs(a, 777), epochs=1, optimizer='adam', diagnostics_every=1)
        rb = self.train(b, cost=costs(b, None), epochs=1, optimizer='adam', diagnostics_every=1)
        np.testing.assert_allclose(ra['reports'][0]['contributions'], rb['reports'][0]['contributions'], rtol=1e-12)
        np.testing.assert_allclose(a.state_dict()['gradient_history'][0]['current'],
                                   b.state_dict()['gradient_history'][0]['current'], rtol=2e-5)
        np.testing.assert_allclose(ra['gradient_reports'][0]['l2_norm'], rb['gradient_reports'][0]['l2_norm'], rtol=2e-5)

    def test_nonfinite_objective_rejects_update_with_sample_and_label(self):
        n = self.linear()
        d = self.data([[[0, 0, 0, 0, 1e30]]], [[0, 0]])
        c = n.cost(inputs=d, equations={'bad': 'u'}, chunk_size=2, name='overflowing-density')
        before = n.state_dict()['weights'][0].copy()
        with self.assertRaisesRegex(RuntimeError, 'non-finite objective.*sample 4.*overflowing-density'):
            # First iteration skips scalar collection; finite checks must still run.
            self.train(n, cost=c, epochs=2, log_every=0, optimizer='adam')
        after = n.state_dict()
        np.testing.assert_array_equal(before, after['weights'][0])
        self.assertEqual(after['metadata']['updates'], 0)
        self.assertEqual(after['metadata']['adam_step'], 0)
        np.testing.assert_array_equal(after['optimizer_state'][0]['m'], 0)
        np.testing.assert_array_equal(after['optimizer_state'][0]['v'], 0)

    def test_nonfinite_output_gradient_with_finite_objective(self):
        n = self.linear(0, 1e-20)
        d = self.data([[[0]]], [[0, 0]])
        c = n.cost(inputs=d, equations={'e': 'scale*u'}, parameters={'scale': 1e30})
        before = n.state_dict()['weights'][0].copy()
        with self.assertRaisesRegex(RuntimeError, 'non-finite dE/dY component 0.*sample 0'):
            self.train(n, cost=c, epochs=1, optimizer='adam')
        after = n.state_dict()
        np.testing.assert_array_equal(before, after['weights'][0])
        self.assertEqual(after['metadata']['updates'], 0)

    def test_nonfinite_weight_gradient_and_session_recovery(self):
        n = self.linear(0, 1e10)
        d = self.data([[[1e30]]], [[0, 0]])
        c = n.cost(inputs=d, equations={'e': 'u'})
        before = n.state_dict()['weights'][0].copy()
        # Local objective and dE/dY are finite, but dE/dW = x*dE/dY overflows.
        with self.assertRaisesRegex(RuntimeError, 'non-finite weight gradient in layer 0, flattened index 0'):
            self.train(n, cost=c, epochs=1, optimizer='adam')
        self.assertEqual(n.last_gradient_report['nonfinite_count'], 1)
        self.assertEqual(n.last_gradient_report['layers'][0]['first_nonfinite_index'], 0)
        self.assertTrue(np.isnan(n.last_gradient_report['l2_norm']))
        after = n.state_dict()
        np.testing.assert_array_equal(before, after['weights'][0])
        self.assertEqual(after['metadata']['adam_step'], 0)
        np.testing.assert_array_equal(after['optimizer_state'][0]['m'], 0)
        np.testing.assert_array_equal(after['optimizer_state'][0]['v'], 0)
        # The process is still usable; a later backward pass overwrites bad buffers.
        after['weights'][0][:] = [0, 0]
        after['gradient_history'] = None  # Do not reload the rejected nonfinite diagnostic buffer.
        n.load_state_dict(after)
        d.set_samples(np.zeros((1, 1, 1), np.float32))
        result = self.train(n, cost=c, epochs=1, optimizer='adam', diagnostics_every=1)
        self.assertEqual(result['gradient_reports'][0]['l2_norm'], 0)
        self.assertEqual(result['gradient_reports'][0]['max_abs'], 0)
        self.assertEqual(result['gradient_reports'][0]['nonfinite_count'], 0)

    def test_diagnostics_argument_and_legacy_capability_validation(self):
        n = self.linear()
        d = self.data([[[0]]], [[0, 0]])
        c = n.cost(inputs=d, equations={'e': 'u'})
        for interval in (-1, 1.5, True):
            with self.assertRaises(ValueError): n.train(cost=c, epochs=1, diagnostics_every=interval)
        capabilities = self.session._capabilities
        self.session._capabilities = 0
        try:
            with self.assertRaisesRegex(ValueError, 'CPDE/2'):
                n.train(cost=c, epochs=1, diagnostics_every=1)
        finally:
            self.session._capabilities = capabilities


if __name__ == '__main__': unittest.main()
