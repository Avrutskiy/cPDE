"""Check analytical energy elimination against the explicit residual and gradient."""
import contextlib
import io
import os
import runpy
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from cpde import Network, Session
from cpde.cost import IntegralCost
from build_host_probe import build


class ProjectionCircleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.executable = os.environ.get('CPDE_TEST_EXECUTABLE') or str(build(cls.temp.name))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_reduced_value_and_envelope_gradient(self):
        original = Network.train
        for suffix in ('', '_vanilla'):
            calls = []
            def train_one(net, **options):
                calls.append(options.copy())
                options['epochs'] = 1
                return original(net, **options)
            with self.subTest(flavor=suffix), Session(self.executable) as session:
                with patch.dict(os.environ, {'CPDE_SERVER': self.executable}), patch('cpde.Session', return_value=contextlib.nullcontext(session)), patch.object(Network, 'train', train_one), patch.object(IntegralCost, 'set_parameters', side_effect=AssertionError('No E update loop')), contextlib.redirect_stdout(io.StringIO()):
                    s = runpy.run_module('examples.schrodinger_circle'+suffix, run_name='__main__')
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0]['epochs'], 2000)
                self.assertEqual(s['net'].state_dict()['metadata']['updates'], 1)
                self.assertEqual(len(s['bulk_points']), 305)
                self.assertEqual(len(s['boundary'].points), 64)
                E = s['B'].evaluate()/s['C'].evaluate()
                self.assertEqual(E, s['E'])
                expression = '(-u_xx-u_yy-E*u)**2'
                if not suffix:
                    expression += '+0.5*(-u_xxx-u_xyy-E*u_x)**2+0.5*(-u_xxy-u_yyy-E*u_y)**2'
                direct = s['net'].cost(inputs=s['interior_inputs'], quadrature=s['interior'], expression=expression, parameters={'E': E})
                reduced = s['A']-s['B']**2/s['C']
                actual, expected = reduced.evaluate(gradient=True), direct.evaluate(gradient=True)
                np.testing.assert_allclose(actual['value'], expected['value'], rtol=2e-5, atol=2e-6)
                ga = np.concatenate(actual['gradient']).astype(float)
                ge = np.concatenate(expected['gradient']).astype(float)
                relative = np.linalg.norm(ga-ge)/np.linalg.norm(ge)
                self.assertLess(relative, 2e-4)
                print(suffix or 'extended', 'value', actual['value'], 'gradient relative difference', relative)
                # The projected energy must minimize the same discrete residual.
                for offset in (-0.1, 0.1):
                    shifted = s['net'].cost(inputs=s['interior_inputs'], quadrature=s['interior'], expression=expression, parameters={'E': E+offset})
                    self.assertGreater(shifted.evaluate(), expected['value'])


if __name__ == '__main__':
    unittest.main()
