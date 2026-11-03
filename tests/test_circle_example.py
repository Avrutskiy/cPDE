"""Independent geometry and phase checks for the unit circle."""
import contextlib
import io
import os
import runpy
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from cpde import Network,Session
from cpde.cost import IntegralCost
from build_host_probe import build


class CircleExampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory()
        cls.executable=os.environ.get('CPDE_TEST_EXECUTABLE') or str(build(cls.temp.name))
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_geometry_and_direct_training(self):
        original_train=Network.train;original_update=IntegralCost.set_parameters
        calls=[];energies=[]
        def train_one(net,**options):
            calls.append(options.copy());options['epochs']=1
            return original_train(net,**options)
        def update(cost,**parameters):
            energies.append(parameters['E']);return original_update(cost,**parameters)
        with Session(self.executable) as session:
            with patch.dict(os.environ,{'CPDE_SERVER':self.executable}),patch('cpde.Session',return_value=contextlib.nullcontext(session)),patch.object(Network,'train',train_one),patch.object(IntegralCost,'set_parameters',update),contextlib.redirect_stdout(io.StringIO()):
                s=runpy.run_module('examples.schrodinger_circle',run_name='__main__')
            self.assertEqual([c['epochs'] for c in calls],[2000])
            self.assertEqual([c['state'] for c in calls],['reset'])
            self.assertEqual([c['reset_steps'] for c in calls],[.0002])
            self.assertEqual([c['restore_every'] for c in calls],[1000])
            self.assertEqual([c['restore_step'] for c in calls],[1e-5])
            self.assertEqual(len(energies),0)
            self.assertEqual(s['net'].state_dict()['metadata']['updates'],1)
            q,b=s['interior'],s['boundary']
            self.assertEqual(len(s['bulk_points']),305)
            self.assertEqual(len(b.points),64);self.assertEqual(len(q.points),369)
            np.testing.assert_array_equal(q.points[-64:],b.points)
            self.assertTrue(np.all(q.weights>0))
            edge=b.points
            polygon_area=.5*abs(np.sum(edge[:,0]*np.roll(edge[:,1],-1)-edge[:,1]*np.roll(edge[:,0],-1)))
            self.assertAlmostEqual(q.weights.sum(),polygon_area,places=12)
            self.assertAlmostEqual(np.sum(q.weights[-64:]),np.sum(s['area_weights'][-64:]),places=12)
            self.assertEqual(s['net'].layers,[2,128,128,128,1])
            np.testing.assert_allclose(s['curvature'],1,atol=0)
            np.testing.assert_allclose(b.N,b.points,atol=0)
            self.assertAlmostEqual(b.weights.sum(),2*np.pi,places=12)
            self.assertTrue(np.all(s['curvature']>0))
            np.testing.assert_allclose(np.sum(b.N*b.T,axis=1),0,atol=2e-14)
            np.testing.assert_allclose(np.linalg.norm(b.T,axis=1),1,atol=2e-14)
            def curve(theta):
                return np.column_stack((np.cos(theta),np.sin(theta)))
            theta=s['theta'];dt=1e-4
            left,right=curve(theta-dt),curve(theta+dt)
            velocity=(right-left)/(2*dt);acceleration=(right-2*edge+left)/dt**2
            speed=np.linalg.norm(velocity,axis=1)
            np.testing.assert_allclose(velocity/speed[:,None],b.T,atol=2e-7)
            measured_curvature=(velocity[:,0]*acceleration[:,1]-velocity[:,1]*acceleration[:,0])/speed**3
            np.testing.assert_allclose(s['curvature'],measured_curvature,rtol=2e-6)
            # Independently differentiate a polynomial restricted to the curved wall.
            polynomial=lambda p:p[:,0]**2+p[:,0]*p[:,1]+p[:,1]**2+.3*p[:,0]
            fp=(polynomial(right)-polynomial(left))/(2*dt)
            fpp=(polynomial(right)-2*polynomial(edge)+polynomial(left))/dt**2
            speed_prime=np.sum(velocity*acceleration,axis=1)/speed
            finite_second=(fpp-fp*speed_prime/speed)/speed**2
            grad=np.column_stack((2*edge[:,0]+edge[:,1]+.3,edge[:,0]+2*edge[:,1]))
            exact_second=2+2*b.T[:,0]*b.T[:,1]-s['curvature']*np.sum(grad*b.N,axis=1)
            np.testing.assert_allclose(exact_second,finite_second,rtol=2e-5,atol=2e-6)
            self.assertNotIn('pretraining',s)
            output=s['net'].eval(s['boundary_inputs']);bp=output.to_numpy().astype(float)[:,0];output.close()
            schema=s['boundary_schema']
            def boundary_row(derivative):
                return np.flatnonzero(np.all(schema[:,1:]==[derivative.count(v) for v in 'xy'],axis=1))[0]
            grad=np.column_stack((bp[boundary_row('x')],bp[boundary_row('y')]))
            second=bp[boundary_row('xx')]*b.T[:,0]**2+2*bp[boundary_row('xy')]*b.T[:,0]*b.T[:,1]+bp[boundary_row('yy')]*b.T[:,1]**2-s['curvature']*np.sum(grad*b.N,axis=1)
            expected_walls=np.dot(b.weights,bp[0]**2+.1*np.sum(grad*b.T,axis=1)**2+.1*second**2)
            np.testing.assert_allclose(s['walls'].evaluate(),expected_walls,rtol=3e-6,atol=1e-7)
            self.assertEqual(set(s['boundary_derivatives']),{'xx','xy','yy'})
            prediction=s['net'].eval(s['interior_inputs']);u=prediction.to_numpy().astype(float)[:,0];prediction.close()
            schema=s['interior_schema']
            def row(derivative):
                counts=[derivative.count(v) for v in 'xy']
                return np.flatnonzero(np.all(schema[:,1:]==counts,axis=1))[0]
            lap=sum(u[row(v+v)] for v in 'xy')
            residual=-lap-s['E']*u[0]
            residual_gradient=np.column_stack([-sum(u[row(v+v+d)] for v in 'xy')-s['E']*u[row(d)] for d in 'xy'])
            expected_equation=np.dot(q.weights,residual**2+.5*np.sum(residual_gradient**2,axis=1))
            np.testing.assert_allclose((s['A']-s['B']**2/s['C']).evaluate(),expected_equation,rtol=2e-6,atol=1e-7)
            self.assertEqual(set(s['interior_derivatives']),{'xxx','xyy','xxy','yyy'})
            grad=np.column_stack([u[row(d)] for d in 'xy'])
            grad_h=np.column_stack([-sum(u[row(v+v+d)] for v in 'xy') for d in 'xy'])
            expected=np.dot(q.weights,-u[0]*lap+.5*np.sum(grad*grad_h,axis=1))/np.dot(q.weights,u[0]**2+.5*np.sum(grad**2,axis=1))
            # CUDA evaluates float32 outputs; the independent contraction is float64.
            np.testing.assert_allclose(s['E'],expected,rtol=2e-6,atol=1e-9)


if __name__=='__main__':unittest.main()
