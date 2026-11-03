import contextlib
import io
import os
from pathlib import Path
import tempfile
import unittest
import numpy as np
from cpde import Session, Data, Network, Quadrature, table, sqrt, exp, log, sin, cos, tanh
from cpde._expressions import Expression
from build_host_probe import build


class GeometryTests(unittest.TestCase):
    def test_scalar_weights_replacement_and_persistence(self):
        q=Quadrature(points=[[0,0],[1,0],[0,1]],weights=np.float32(.25),variables='xy',dimension=2)
        self.assertEqual(q.weights.sum(),.75)
        np.testing.assert_array_equal(q.weights,[.25,.25,.25])
        with self.assertRaises(ValueError): q.weights.setflags(write=True)
        q.replace(points=[[1,2],[3,4]],weights=np.array(-.5))
        np.testing.assert_array_equal(q.weights,[-.5,-.5])
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'scalar.npz';q.save(path);loaded=Quadrature.load(path)
            np.testing.assert_array_equal(loaded.weights,q.weights)
        q.replace(points=np.empty((0,2)),weights=0)
        self.assertEqual(q.weights.shape,(0,))

    def test_scalar_weight_validation_including_empty_points(self):
        for points in (np.zeros((3,2)),np.empty((0,2))):
            for weights in (np.nan,np.inf,-np.inf,[.5],[[.5]],np.ones((len(points),1))):
                with self.subTest(shape=points.shape,weights=weights),self.assertRaises(ValueError):
                    Quadrature(points=points,weights=weights,variables='xy',dimension=2)

    def test_basic_delaunay(self):
        try: import scipy.spatial
        except ImportError: self.skipTest('optional geometry extra is not installed')
        square=Quadrature(points=[[0,0],[1,0],[1,1],[0,1]],variables='xy',dimension=2,triangulation='delaunay')
        self.assertAlmostEqual(square.weights.sum(),1)
        cube=Quadrature(points=[[x,y,z] for x in (0,1) for y in (0,1) for z in (0,1)],variables='xyz',dimension=3,triangulation='delaunay')
        self.assertAlmostEqual(cube.weights.sum(),1)

    def test_simplex_measures_rules_and_orientation(self):
        triangle = [[0,0],[2,0],[0,3]]
        for rule in ('centroid','vertices'):
            q=Quadrature(points=triangle,cells=[[0,1,2]],variables='xy',dimension=2,rule=rule)
            self.assertAlmostEqual(q.weights.sum(),3)
            self.assertAlmostEqual(np.dot(q.weights,q.points[:,0]+2*q.points[:,1]),8)
        q=Quadrature(points=[[0,0,0],[2,0,0],[0,3,0],[0,0,4]],cells=[[0,1,2,3]],variables='xyz',dimension=3)
        self.assertAlmostEqual(q.weights.sum(),4)
        q=Quadrature(points=[[0,0],[1,0],[1,1],[0,1]],cells=[[0,1],[1,2],[2,3],[3,0]],variables='xy',dimension=1)
        np.testing.assert_allclose(np.sum(q.weights[:,None]*q.N,axis=0),0,atol=1e-15)
        self.assertAlmostEqual(np.sum(q.weights*np.sum(q.points*q.N,axis=1)),2)
        reversed_q=Quadrature(points=q.vertices,cells=q.cells[:,::-1],variables='xy',dimension=1)
        np.testing.assert_allclose(reversed_q.N,-q.N)
        np.testing.assert_allclose(reversed_q.weights,q.weights)

    def test_validation_replacement_and_persistence(self):
        q=Quadrature(points=[[0,0]],weights=[-2],variables='xy',dimension=0)
        with self.assertRaises(ValueError): q.points.setflags(write=True)
        with self.assertRaises(ValueError): q.replace(points=[[0,0]],weights=[float('nan')])
        self.assertEqual(q.revision,0)
        q.replace(points=[[1,2],[3,4]],weights=[1,2]); self.assertEqual(q.revision,1)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'q.npz';q.save(path);r=Quadrature.load(path)
            np.testing.assert_array_equal(r.points,q.points);self.assertEqual(r.identity,q.identity)
        with self.assertRaises(ValueError): Quadrature(points=[[0,0],[1,0],[2,0]],cells=[[0,1,2]],variables='xy',dimension=2)
        with self.assertRaises(ValueError): Quadrature(points=[[0,0]],weights=[1],variables='xy',dimension=1,N=[[2,0]])
        with self.assertRaises(ValueError): Quadrature(points=[[0,0]],weights=[1],cells=[[0,0]],variables='xy',dimension=1)

    def test_expression_rejects_unsafe_and_ambiguous_syntax(self):
        for text in ('__import__("os")','u[0]','u if u else 1','abs(u)','u**u','coord.z','u_zz'):
            with self.subTest(text=text),self.assertRaises(ValueError): Expression(text,'xy','u')
        with self.assertRaises(ValueError): Expression('u_x','xy','u',freeze=['u'])
        with self.assertRaises(ValueError): Expression('u','xy','u',parameters={'sqrt':1})
        p=Expression('u_xy+u_yx','xy','u',freeze=['u_yx'])
        self.assertEqual(p.components,['u_xy']);self.assertEqual(p.freeze,{'u_xy'})


class IntegralTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory()
        cls.executable=os.environ.get('CPDE_TEST_EXECUTABLE') or str(build(cls.temp.name))
    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()
    def setUp(self):
        self.session=Session(self.executable);self.addCleanup(self.session.close)
        if not self.session._capabilities & 4: self.skipTest('executable has no expression compiler')
    def net(self,dimensions=2):
        n=Network([dimensions,1],session=self.session).name_outputs('u')
        s=n.state_dict();s['weights'][0][:]=[.3,-.4,.8] if dimensions==2 else [.4,.7];n.load_state_dict(s)
        return n
    def q(self,n=7):
        x=np.linspace(-.7,.9,n)
        return Quadrature(points=np.stack([x,x*x],axis=1),weights=np.linspace(.1,.3,n),variables='xy',dimension=2)
    def finite_difference(self,n,c,step=.002):
        original=n.state_dict();actual=c.evaluate(gradient=True)['gradient'];expected=[]
        for layer,a in enumerate(original['weights']):
            result=np.zeros_like(a)
            for j in range(len(a)):
                old=float(a[j]);a[j]=old+step;n.load_state_dict(original);plus=c.evaluate()
                a[j]=old-step;n.load_state_dict(original);minus=c.evaluate()
                a[j]=old;result[j]=(plus-minus)/(2*step)
            expected.append(result)
        n.load_state_dict(original)
        for a,b in zip(actual,expected): np.testing.assert_allclose(a,b,rtol=2e-3,atol=2e-4)
    def test_signed_integral_general_functions_and_gradient(self):
        n=self.net();q=self.q();q.replace(points=q.points,weights=[1,-2,3,-4,5,-6,7])
        c=n.cost(quadrature=q,expression='sqrt(1+u_x**2+u_y**2)+exp(u)*sin(u)+cos(u)+log(2+u)+tanh(u)+coord.x',chunk_size=3)
        c.compile();self.assertIn('__global__',c.inspect()['source'])
        self.finite_difference(n,c)
        direct=n.eval(q,derivatives=['','x','y']).to_numpy()
        np.testing.assert_allclose(direct[0,0],.3*q.points[:,0]-.4*q.points[:,1]+.8,atol=1e-7)
        constant=n.cost(quadrature=q,expression='-2');self.assertEqual(constant.evaluate(),-8)
        unnamed=Network([2,1],session=self.session)
        self.assertEqual(unnamed.cost(quadrature=q,expression='1').evaluate(),4)

    def test_scalar_quadrature_weights_integrate_value_and_gradient(self):
        n=self.net();points=np.array([[0,0],[1,0],[0,2]],float)
        q=Quadrature(points=points,weights=.25,variables='xy',dimension=2)
        cost=n.cost(quadrature=q,expression='u**2',chunk_size=2)
        result=cost.evaluate(gradient=True)
        coefficients=n.state_dict()['weights'][0].astype(float)
        features=np.column_stack((points,np.ones(3)))
        u=features@coefficients
        np.testing.assert_allclose(result['value'],.25*np.dot(u,u),rtol=2e-7)
        np.testing.assert_allclose(result['gradient'][0],.5*features.T@u,rtol=2e-7,atol=1e-7)
        q.replace(points=points,weights=-.5)
        negative=cost.evaluate(gradient=True)
        np.testing.assert_allclose(negative['value'],-2*result['value'],rtol=1e-12)
        np.testing.assert_allclose(negative['gradient'][0],-2*result['gradient'][0],rtol=1e-7)

    def test_compiled_cost_ignores_unused_overflowing_derivatives(self):
        n=Network([1,2,1],session=self.session).name_outputs('u')
        state=n.state_dict();state['weights'][0][:]=[2,2,0,0];state['weights'][1][:]=[1,1,0];n.load_state_dict(state)
        schema=table(['xxxxxx'],variables='x')
        samples=np.array([.25,1,0,0,0,0,3e38],np.float32).reshape(7,1,1)
        data=Data(samples,schema,session=self.session).name_variables('x')
        target=Data(np.zeros_like(samples),schema,session=self.session).name_variables('x')
        legacy=n.cost(inputs=data,targets=target,weights=[[1,0,0,0,0,0,0]]).evaluate(gradient=True)
        compiled=n.cost(inputs=data,expression='u**2').evaluate(gradient=True)
        self.assertAlmostEqual(compiled['value'],legacy['value'],places=6)
        for a,b in zip(compiled['gradient'],legacy['gradient']):
            np.testing.assert_allclose(a,b,rtol=2e-6,atol=1e-7)

    def test_pruned_permuted_rows_and_frozen_derivative_dependencies(self):
        q=self.q();n=Network([2,3,1],session=self.session).name_outputs('u')
        full=q._data(self.session,['xxx','xxy','xyy','yyy'])
        order=np.arange(len(full.derivatives))[::-1]
        data=Data(full.to_numpy()[order],full.derivatives[order],session=self.session).name_variables('xy')
        expression='u**2+u_xx*u_y'
        a=n.cost(inputs=data,quadrature=q,expression=expression,freeze=['u_xx'],chunk_size=3)
        b=n.cost(quadrature=q,expression=expression,freeze=['u_xx'],chunk_size=2)
        actual,expected=a.evaluate(gradient=True),b.evaluate(gradient=True)
        self.assertAlmostEqual(actual['value'],expected['value'],places=6)
        for x,y in zip(actual['gradient'],expected['gradient']):
            np.testing.assert_allclose(x,y,rtol=2e-5,atol=1e-6)
    def test_nonlinear_network_and_cost_graph_finite_difference(self):
        q=self.q();n=Network([2,3,1],session=self.session).name_outputs('u')
        state=n.state_dict();rng=np.random.default_rng(1)
        for a in state['weights']:a[:]=rng.uniform(-.4,.4,len(a))
        n.load_state_dict(state)
        a=n.cost(quadrature=q,expression='u**2+u_x**2+u_y**2')
        b=n.cost(quadrature=q,expression='1')
        c=exp(a/b)+sqrt(a+2)+log(a+3)+sin(a)+cos(a)+tanh(a)+(2-a)**2
        self.finite_difference(n,c)
    def test_freeze_and_nested_variations_with_shared_leaves(self):
        n=self.net();q=self.q()
        a=n.cost(quadrature=q,expression='u')
        correction=n.cost(quadrature=q,expression='u*u_x',freeze=['u_x'])
        frozen=n.cost(quadrature=q,expression='u**2',freeze=['u'])
        A=a.evaluate(gradient=True);B=correction.evaluate(gradient=True)
        np.testing.assert_allclose(B['gradient'][0],.3*A['gradient'][0],rtol=2e-7)
        np.testing.assert_array_equal(frozen.evaluate(gradient=True)['gradient'][0],0)
        attached=(a*a).with_variation(correction.with_variation(a))
        graph=attached+correction+a
        result=graph.evaluate(gradient=True)
        self.assertAlmostEqual(result['value'],A['value']**2+B['value']+A['value'])
        np.testing.assert_allclose(result['gradient'][0],(2*A['value']+2)*A['gradient'][0]+2*B['gradient'][0],rtol=3e-7)
    def test_gradient_evaluation_preserves_optimizer_history_and_progress(self):
        n=self.net();c=n.cost(quadrature=self.q(),expression='u**2')
        with contextlib.redirect_stdout(io.StringIO()):n.train(cost=c,epochs=3,optimizer='adam',verbose=0)
        before=n.state_dict();c.evaluate(gradient=True);after=n.state_dict()
        self.assertEqual(before['metadata'],after['metadata'])
        for key in ('weights','optimizer_state','gradient_history'):
            for a,b in zip(before[key],after[key]):
                if isinstance(a,dict):
                    for k in a:np.testing.assert_array_equal(a[k],b[k])
                else:np.testing.assert_array_equal(a,b)
        bad=log(-c)
        with self.assertRaises(RuntimeError):bad.evaluate(gradient=True)
        after=n.state_dict()
        for a,b in zip(before['gradient_history'],after['gradient_history']):np.testing.assert_array_equal(a['current'],b['current'])
    def test_graph_training_matches_legacy_and_reports_root(self):
        n=self.net();m=self.net();q=self.q();d=q._data(self.session,[''])
        legacy=n.cost(inputs=d,equations={'e':'u'})
        compiled=m.cost(inputs=d,expression='u**2')
        with contextlib.redirect_stdout(io.StringIO()):
            a=n.train(cost=legacy,epochs=4,optimizer='adam',verbose=0)
            b=m.train(cost=compiled,epochs=4,optimizer='adam',verbose=0)
        np.testing.assert_allclose(n.state_dict()['weights'][0],m.state_dict()['weights'][0],atol=1e-7)
        self.assertIn('leaf_values',b['reports'][-1]);self.assertNotIn('contributions',b['reports'][-1])
        with contextlib.redirect_stdout(io.StringIO()):r=m.train(cost=(compiled+1)**2,epochs=1,verbose=0)
        self.assertAlmostEqual(r['reports'][-1]['total'],(compiled.evaluate()+1)**2,places=8)

    def test_training_recomputes_dynamic_multipliers_before_each_update(self):
        n,m=self.net(),self.net()
        qa,qb=self.q(7),self.q(3)
        a=n.cost(quadrature=qa,expression='u**2',chunk_size=3)
        b=n.cost(quadrature=qb,expression='1+u_x**2',chunk_size=2)
        reference_a=m.cost(quadrature=qa,expression='u**2',chunk_size=3)
        reference_b=m.cost(quadrature=qb,expression='1+u_x**2',chunk_size=2)
        with contextlib.redirect_stdout(io.StringIO()):
            n.train(cost=a+10*b*b,epochs=3,optimizer='adam',verbose=0)
            # Independent reference: fix the correct multiplier for each step.
            for _ in range(3):
                multiplier=20*reference_b.evaluate()
                m.train(cost=reference_a+multiplier*reference_b,epochs=1,optimizer='adam',verbose=0)
        np.testing.assert_allclose(n.state_dict()['weights'][0],m.state_dict()['weights'][0],atol=1e-7)
        self.assertEqual(n.state_dict()['metadata']['updates'],3)
        self.assertEqual(n.state_dict()['metadata']['adam_step'],3)
    def test_replacement_parameters_empty_and_zero_gradient(self):
        n=self.net();q=self.q();c=n.cost(quadrature=q,expression='co*u',parameters={'co':np.ones(7)})
        c.compile();code=c.inspect()['source']
        q.replace(points=[[0,0],[1,0]],weights=[2,-1])
        with self.assertRaises(ValueError):c.evaluate()
        c.set_parameters(co=[3,4]);c.compile();self.assertEqual(code,c.inspect()['source'])
        np.testing.assert_allclose(c.evaluate(),2*3*.8-4*1.1,atol=1e-6)
        empty=Quadrature(points=np.empty((0,2)),weights=[],variables='xy',dimension=2)
        z=n.cost(quadrature=empty,expression='log(u)')
        self.assertEqual(z.evaluate(),0);np.testing.assert_array_equal(z.evaluate(gradient=True)['gradient'][0],0)
        np.testing.assert_array_equal((0*c).evaluate(gradient=True)['gradient'][0],0)
        with self.assertRaises(RuntimeError):(c/z).evaluate()
    def test_padding_zero_weights_domains_and_recovery(self):
        n=self.net();q=Quadrature(points=[[0,0],[1,0],[2,0]],weights=[0,1,1],variables='xy',dimension=2)
        c=n.cost(quadrature=q,expression='u/coord.x',chunk_size=2)
        np.testing.assert_allclose(c.evaluate(),1.1+1.4/2,atol=2e-7)
        before=n.state_dict()['weights'][0].copy()
        bad=n.cost(quadrature=q,expression='sqrt(-u)')
        with contextlib.redirect_stdout(io.StringIO()),self.assertRaisesRegex(RuntimeError,'sample 1'):
            n.train(cost=bad,epochs=1,verbose=0)
        np.testing.assert_array_equal(before,n.state_dict()['weights'][0]);self.assertTrue(np.isfinite(c.evaluate()))

    def test_geometry_refresh_retains_structural_training_progress(self):
        n=self.net();q=self.q();c=n.cost(quadrature=q,expression='scale*u**2',parameters={'scale':1.0})
        with contextlib.redirect_stdout(io.StringIO()):n.train(cost=c,epochs=2,optimizer='adam',verbose=0)
        q.replace(points=[[0,0],[1,0]],weights=[.5,.5]);c.set_parameters(scale=2.0)
        with contextlib.redirect_stdout(io.StringIO()):result=n.train(cost=c,epochs=1,verbose=0)
        self.assertEqual(result['metadata']['updates'],3)
        self.assertEqual(result['metadata']['stage_completed'],3)
        self.assertEqual(result['metadata']['adam_step'],3)
    def test_matching_equations_directions_and_legacy_graphs(self):
        n=self.net();q=self.q();data=q._data(self.session,['','x'])
        target=Data(np.zeros((2,1,7)),data.derivatives,session=self.session).name_variables('xy')
        c=n.cost(quadrature=q,targets=target,weights=[[1,.2]])
        direct=n.cost(quadrature=q,expression='u**2+.2*u_x**2')
        self.assertAlmostEqual(c.evaluate(),direct.evaluate(),places=12)
        self.finite_difference(n,c)
        edge=Quadrature(points=[[0,0],[1,0]],cells=[[0,1]],variables='xy',dimension=1)
        flux=n.cost(quadrature=edge,expression='u_x*N.x+u_y*N.y')
        np.testing.assert_allclose(flux.evaluate(),.4,atol=1e-7)
        old=n.cost(inputs=data,equations={'e':'u'})
        self.finite_difference(n,(old+old)/3+direct)
    def test_invalid_graph_ownership_and_direction_fields(self):
        n=self.net();q=self.q();c=n.cost(quadrature=q,expression='u')
        with self.assertRaises(ValueError):n.cost(quadrature=q,expression='N.x*u')
        with self.assertRaises(TypeError):bool(c)
        with self.assertRaises(ValueError):c+self.net().cost(quadrature=q,expression='u')
        graph=c+1
        with self.assertRaises(ValueError):c.close()
        graph.close();c.close()
        with self.assertRaises(ValueError):graph.evaluate()

    def test_moving_drop_footprint_requires_boundary_variation(self):
        # A paraboloid height represented exactly through supplied coordinate
        # features. Radial Gauss quadrature avoids a mesh-discretization FD error.
        n=Network([3,1],session=self.session).name_outputs('u')
        state=n.state_dict();state['weights'][0][:]=[0,0,-.5,.7];n.load_state_dict(state)
        radial,rw=np.polynomial.legendre.leggauss(12)
        angles=2*np.pi*np.arange(48)/48
        def geometry(a):
            radius=np.sqrt(a/.5);r=(radial+1)*radius/2
            p=np.stack([r[:,None]*np.cos(angles),r[:,None]*np.sin(angles)],axis=-1).reshape(-1,2)
            w=np.repeat(rw*radius/2*r,48)*2*np.pi/48
            edge=radius*np.stack([np.cos(angles),np.sin(angles)],axis=1)
            return p,w,edge,np.full(48,2*np.pi*radius/48)
        def features(p):
            result=np.zeros((3,3,len(p)),np.float32)
            result[0]=np.stack([p[:,0],p[:,1],np.sum(p*p,axis=1)])
            result[1,0]=1;result[1,2]=2*p[:,0]
            result[2,1]=1;result[2,2]=2*p[:,1]
            return result
        p,w,edge,ew=geometry(.7)
        q=Quadrature(points=p,weights=w,variables='xy',dimension=2)
        boundary=Quadrature(points=edge,weights=ew,variables='xy',dimension=1)
        d=Data(features(p),[[0,0,0],[1,1,0],[1,0,1]],session=self.session).name_variables('xy')
        bd=Data(features(edge),d.derivatives,session=self.session).name_variables('xy')
        fixed=n.cost(inputs=d,quadrature=q,expression='sqrt(1+u_x**2+u_y**2)-.5',chunk_size=77)
        correction=n.cost(inputs=bd,quadrature=boundary,
            expression='u*(sqrt(1+u_x**2+u_y**2)-.5)/sqrt(u_x**2+u_y**2)',freeze=['u_x','u_y'])
        total=fixed.with_variation(correction)
        self.assertEqual(fixed.evaluate(gradient=True)['gradient'][0][-1],0)
        actual=total.evaluate(gradient=True)['gradient'][0][-1]
        expected=(np.sqrt(1+1.4)-.5)*np.pi/.5
        np.testing.assert_allclose(actual,expected,rtol=2e-6)
        def energy(a):
            p,w,edge,ew=geometry(a);q.replace(points=p,weights=w);boundary.replace(points=edge,weights=ew)
            d.set_samples(features(p));bd.set_samples(features(edge))
            state['weights'][0][-1]=a;n.load_state_dict(state)
            return total.evaluate()
        step=.002;fd=(energy(.7+step)-energy(.7-step))/(2*step)
        np.testing.assert_allclose(actual,fd,rtol=2e-4)


if __name__=='__main__':unittest.main()
