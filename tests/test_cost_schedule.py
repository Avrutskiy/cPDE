"""Count actual forwards in a test-only instrumented serial host engine."""
import tempfile
import unittest
import shutil
import numpy as np
from cpde import Network, Quadrature, Session
from build_host_probe import build


@unittest.skipUnless(shutil.which('g++'), 'host-only forward instrumentation requires g++')
class CostScheduleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.executable = str(build(cls.temp.name, count_forwards=True))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_actual_forward_counts_and_accumulated_gradients(self):
        # Run with A split into three chunks and B into two, then without chunks.
        # Counts include every actual network forward; the final live one is reused.
        cases = [
            (lambda a,b: a, lambda A,B: (1,0), (3,1)),
            (lambda a,b: 2*a-3*b+4, lambda A,B: (2,-3), (5,2)),
            (lambda a,b: a*a+b, lambda A,B: (2*A,1), (7,2)),
            (lambda a,b: a/b, lambda A,B: (1/B,-A/B**2), (9,3)),
            (lambda a,b: a.with_variation(b), lambda A,B: (1,1), (5,2)),
            (lambda a,b: a.with_variation(b*b), lambda A,B: (1,2*B), (6,2)),
            (lambda a,b: a.with_variation(b)**2, lambda A,B: (2*A,2*A), (9,3)),
            (lambda a,b: a+a+a/2-b, lambda A,B: (2.5,-1), (5,2)),
            (lambda a,b: a**1+b**0, lambda A,B: (1,0), (5,2)),
            (lambda a,b: a*a, lambda A,B: (2*A,0), (5,1)),
            (lambda a,b: b/a, lambda A,B: (-B/A**2,1/A), (9,3)),
        ]
        for index,(make,coefficients,expected_counts) in enumerate(cases):
            for mode,(chunk_a,chunk_b) in enumerate(((3,2),(None,None))):
                with self.subTest(index=index,chunked=mode==0):
                    with Session(self.executable) as session:
                        net = Network([2,3,1],session=session).name_outputs('u')
                        state = net.state_dict()
                        for w in state['weights']: w[:] = np.linspace(.1,.5,len(w))
                        net.load_state_dict(state)
                        p = np.column_stack((np.linspace(-.7,.9,7),np.linspace(.2,.6,7)))
                        qa = Quadrature(points=p,weights=np.ones(7)/7,variables='xy',dimension=2)
                        qb = Quadrature(points=p[:3],weights=np.ones(3)/3,variables='xy',dimension=2)
                        a = net.cost(quadrature=qa,expression='u**2',chunk_size=chunk_a)
                        b = net.cost(quadrature=qb,expression='1+u_x**2+u_y**2',chunk_size=chunk_b)
                        A,B = a.evaluate(gradient=True),b.evaluate(gradient=True)
                        result = make(a,b).evaluate(gradient=True)
                        ca,cb = coefficients(A['value'],B['value'])
                        for actual,ga,gb in zip(result['gradient'],A['gradient'],B['gradient']):
                            np.testing.assert_allclose(actual,ca*ga+cb*gb,rtol=3e-6,atol=1e-6)
                    # close() joins stderr reader; remove the A/B reference evaluations.
                    trace = [tuple(map(int,line.split()[1:])) for line in session._diagnostics
                             if line.startswith('CPDE_TEST_FORWARD ')]
                    trace = trace[5 if mode==0 else 2:]
                    self.assertEqual(len(trace),expected_counts[mode])
                    if index in (3,10):
                        # B's derivatives make its forward more expensive despite its
                        # smaller point set. It runs last even for the expression B/A.
                        if mode==0:
                            self.assertEqual(trace[:5],[(1,3)]*3+[(3,2)]*2)
                            self.assertEqual(trace[5:].count((1,3)),3)
                            self.assertEqual(trace[5:].count((3,2)),1)
                        else:
                            self.assertEqual(trace,[(1,7),(3,3),(1,7)])


if __name__ == '__main__':
    unittest.main()
