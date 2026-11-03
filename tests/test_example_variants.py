"""Check delivered vanilla/extended examples against their numerical objectives."""
import contextlib
import io
import os
import runpy
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from cpde import Network, Session
from build_host_probe import build
from example_reference import fourier


class ExampleVariantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.executable = os.environ.get('CPDE_TEST_EXECUTABLE') or str(build(cls.temp.name))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_delivered_pairs(self):
        for base in ['fit_function', 'poisson_disk', 'schrodinger_circle']:
            geometry = []
            for vanilla in [True, False]:
                name = base + ('_vanilla' if vanilla else '')
                with self.subTest(example=name), Session(self.executable) as session:
                    with patch.dict(os.environ, {'CPDE_SERVER': self.executable}), \
                         patch('cpde.Session', return_value=contextlib.nullcontext(session)), \
                         patch.object(Network, 'train', return_value={}) as train, \
                         contextlib.redirect_stdout(io.StringIO()):
                        s = runpy.run_module('examples.'+name, run_name='__main__')
                    net=s['net']
                    if base=='fit_function':
                        if not vanilla:
                            self.assertEqual(s['derivatives'], ['xxx','xxy','xyy','yyy'])
                        self.assertEqual(len(s['schema']),1 if vanilla else 10)
                        self.assertEqual(s['inputs'].sample_count,729)
                        self.assertEqual(train.call_count,1)
                        geometry.append(np.column_stack((s['x'],s['y'])))
                        expected = fourier(geometry[-1], s['schema'], s['coefficients'])
                        np.testing.assert_allclose(s['targets'].to_numpy()[:, 0], expected, rtol=2e-6, atol=2e-7)
                        continue
                    q,b=s['interior'],s['boundary']
                    geometry.append(q.points)
                    if not vanilla:
                        self.assertEqual(s['boundary_derivatives'], ['xx','xy','yy'])
                    self.assertEqual(len(s['boundary_schema']),1 if vanilla else 6)
                    self.assertEqual(len(s['interior_schema']),5 if vanilla else 10)
                    pred=net.eval(s['interior_inputs']);u=pred.to_numpy()[:,0].astype(float);pred.close()
                    schema=s['interior_schema']
                    def row(suffix):
                        return np.flatnonzero(np.all(schema[:,1:]==[suffix.count(v) for v in 'xy'],axis=1))[0]
                    lap=u[row('xx')]+u[row('yy')]
                    energy=s['E'] if base=='schrodinger_circle' else 0
                    residual=-lap-energy*u[0] if base=='schrodinger_circle' else lap-1
                    density=residual**2
                    if not vanilla:
                        for d in 'xy':
                            derivative=u[row('xx'+d)]+u[row('yy'+d)]+energy*u[row(d)]
                            density += .5*derivative**2
                    equation=s['A']-s['B']**2/s['C'] if base=='schrodinger_circle' else s['equation']
                    np.testing.assert_allclose(equation.evaluate(),np.dot(q.weights,density),rtol=3e-6,atol=2e-7)
                    self.assertEqual(train.call_count,1)
                    if vanilla:
                        pred=net.eval(s['boundary_inputs']);u=pred.to_numpy()[0,0].astype(float);pred.close()
                        walls=s['walls'] if base=='schrodinger_circle' else s['boundary_cost']
                        np.testing.assert_allclose(walls.evaluate(),np.dot(b.weights,u**2),rtol=3e-6,atol=2e-7)
            np.testing.assert_array_equal(*geometry)


if __name__=='__main__':
    unittest.main()
