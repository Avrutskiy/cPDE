"""Contracts specific to the production CPU executable and session selection."""
import os
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from cpde import Session, Network, Quadrature


class BackendSelectionTests(unittest.TestCase):
    def test_invalid_selection_does_not_launch(self):
        with patch('cpde.api.subprocess.Popen') as launch:
            for backend in ('gpu', 'CPU', '', 1):
                with self.subTest(backend=backend), self.assertRaises(ValueError):
                    Session(backend=backend)
            with self.assertRaises(ValueError):
                Session(backend='cpu', device=1)
            launch.assert_not_called()

    def test_missing_cpu_executable_does_not_fall_back(self):
        with patch.dict(os.environ, {'CPDE_SERVER': '/unrelated/gpu/server'}, clear=True), \
                patch('cpde.api.shutil.which', return_value=None), \
                patch('cpde.api.subprocess.Popen') as launch:
            with self.assertRaisesRegex(FileNotFoundError, 'CPDE_CPU_SERVER'):
                Session(backend='cpu')
            launch.assert_not_called()


@unittest.skipUnless(os.environ.get('CPDE_TEST_CPU_EXECUTABLE'), 'production CPU executable not supplied')
class CpuBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.executable = str(Path(os.environ['CPDE_TEST_CPU_EXECUTABLE']).resolve())

    def test_selection_discovery_and_mismatch(self):
        with Session(self.executable) as session:
            self.assertEqual(session.backend, 'cpu')
        with patch.dict(os.environ, {'CPDE_CPU_SERVER': self.executable}):
            with Session(backend='cpu') as session:
                self.assertEqual(session.backend, 'cpu')
        with patch.dict(os.environ, {}, clear=True), patch('cpde.api.shutil.which', return_value=self.executable):
            with Session(backend='cpu') as session:
                self.assertEqual(session.backend, 'cpu')
        with self.assertRaisesRegex(RuntimeError, 'requested cuda.*reports cpu'):
            Session(self.executable, backend='cuda')

    def test_expression_gradient_training_and_state(self):
        # u = a*x+b. Compare the integral and both gradients analytically.
        x = np.array([-.7, -.1, .3, .9])
        weights = np.array([.2, .1, .3, .4])
        with Session(self.executable, backend='cpu') as session:
            net = Network([1, 1], session=session, seed=9).name_outputs('u')
            state = net.state_dict()
            state['weights'][0][:] = [.4, -.2]
            net.load_state_dict(state)
            q = Quadrature(points=x[:, None], weights=weights, variables='x', dimension=1)
            cost = net.cost(quadrature=q, expression='(u-1)**2+0.5*u_x**2')
            actual = cost.evaluate(gradient=True)
            u = .4*x-.2
            expected_value = np.dot(weights, (u-1)**2+.5*.4**2)
            expected_gradient = [np.dot(weights,2*(u-1)*x+.4),np.dot(weights,2*(u-1))]
            self.assertAlmostEqual(actual['value'], expected_value, places=6)
            np.testing.assert_allclose(actual['gradient'][0],expected_gradient,rtol=2e-6,atol=1e-7)
            cost.compile()
            self.assertIn('CPU backend',cost.inspect()['compile_log'])
            net.train(cost=cost,epochs=3,optimizer='rprop',reset_steps=.01,verbose=0)
            self.assertLess(cost.evaluate(),actual['value'])
            saved = net.state_dict()
            copy = Network([1,1],session=session,state=saved).name_outputs('u')
            copied_cost=copy.cost(quadrature=q,expression='(u-1)**2+0.5*u_x**2')
            self.assertEqual(copied_cost.evaluate(),cost.evaluate())
