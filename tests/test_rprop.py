"""RPROP bound rejection and history tests; use CPDE_TEST_EXECUTABLE for CUDA."""
import contextlib
import io
import os
from pathlib import Path
import tempfile
import unittest
import numpy as np
from cpde import Data, Network, Session
from build_host_probe import build


class RpropBoundTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory()
        cls.executable=os.environ.get('CPDE_TEST_EXECUTABLE') or str(build(cls.temp.name))

    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()

    def setUp(self):
        self.session=Session(self.executable);self.addCleanup(self.session.close)
        self.data=Data(np.ones((1,1,1),np.float32),[[0,0]],session=self.session).name_variables('x')
        self.net=Network([1,1],session=self.session).name_outputs('u')
        self.cost=self.net.cost(inputs=self.data,expression='(u-target)**2',parameters={'target':0})

    def initialize(self,weight,step,previous):
        state=self.net.state_dict();state['metadata']['optimizer']='rprop'
        state['weights'][0][:]=[weight,0]
        state['optimizer_state'][0]['steps'][:]=step
        state['gradient_history'][0]['current'][:]=0
        state['gradient_history'][0]['previous'][:]=previous
        self.net.load_state_dict(state)

    def update(self,target,epochs=1):
        self.cost.set_parameters(target=target)
        with contextlib.redirect_stdout(io.StringIO()):
            self.net.train(cost=self.cost,epochs=epochs,optimizer='rprop',verbose=0)
        return self.net.state_dict()

    def test_rejected_step_then_sign_reversal_both_bounds(self):
        for sign in [-1,1]:
            with self.subTest(sign=sign):
                self.initialize(sign*19,100,-sign)
                rejected=self.update(sign*100)
                self.assertEqual(rejected['weights'][0][0],sign*19)
                np.testing.assert_array_equal(rejected['optimizer_state'][0]['steps'],np.float32(.1))
                np.testing.assert_array_equal(rejected['gradient_history'][0]['previous'],0)
                inward=self.update(0)
                self.assertAlmostEqual(float(inward['weights'][0][0]),sign*18.9,places=5)

    def test_repeated_rejection_and_checkpoint_keep_reset(self):
        self.initialize(20,100,-1)
        state=self.update(100,epochs=8)
        self.assertEqual(state['weights'][0][0],20)
        self.assertAlmostEqual(float(state['optimizer_state'][0]['steps'][0]),.1)
        self.assertEqual(state['gradient_history'][0]['previous'][0],0)
        path=Path(self.temp.name)/'rejected.npz';self.net.save(path);self.net.load(path)
        inward=self.update(0)
        self.assertAlmostEqual(float(inward['weights'][0][0]),19.9,places=5)

    def test_accepted_step_still_rolls_back(self):
        self.initialize(.5,.1,0)
        accepted=self.update(1)
        self.assertAlmostEqual(float(accepted['weights'][0][0]),.6,places=6)
        reversed_state=self.update(-1)
        self.assertAlmostEqual(float(reversed_state['weights'][0][0]),.5,places=6)
        self.assertAlmostEqual(float(reversed_state['optimizer_state'][0]['steps'][0]),.05,places=6)

    def test_zero_gradient_does_not_move_or_reset_step(self):
        self.initialize(0,2,0)
        state=self.update(0)
        np.testing.assert_array_equal(state['weights'][0],0)
        np.testing.assert_array_equal(state['optimizer_state'][0]['steps'],2)

    def test_periodic_restore_threshold_and_selective_history_clear(self):
        floor=np.float32(1e-5)
        for small,unchanged in [(0,floor),(floor/2,2*floor)]:
            with self.subTest(small=small,unchanged=unchanged):
                self.initialize(0,0,0)
                state=self.net.state_dict()
                state['optimizer_state'][0]['steps'][:]=[small,unchanged]
                state['gradient_history'][0]['current'][:]=[3,4]
                state['gradient_history'][0]['previous'][:]=[5,6]
                self.net.load_state_dict(state)
                with contextlib.redirect_stdout(io.StringIO()):
                    self.net.train(cost=self.cost,epochs=1,optimizer='rprop',verbose=0,
                                   restore_every=1,restore_step=floor)
                state=self.net.state_dict()
                np.testing.assert_array_equal(state['weights'][0],0)
                np.testing.assert_array_equal(state['optimizer_state'][0]['steps'],[floor,unchanged])
                # After the active-buffer switch, current holds the old previous buffer.
                np.testing.assert_array_equal(state['gradient_history'][0]['current'],[0,6])
                np.testing.assert_array_equal(state['gradient_history'][0]['previous'],0)

    def test_restore_schedule_survives_call_checkpoint_and_cost_change(self):
        self.initialize(.5,0,-1)
        state=self.net.state_dict();state['metadata']['updates']=998;self.net.load_state_dict(state)
        options=dict(epochs=1,optimizer='rprop',verbose=0,restore_every=1000,restore_step=1e-5)
        with contextlib.redirect_stdout(io.StringIO()):
            self.cost.set_parameters(target=1)
            self.net.train(cost=self.cost,**options)
            before=self.net.state_dict()
            self.assertEqual(before['metadata']['updates'],999)
            np.testing.assert_array_equal(before['optimizer_state'][0]['steps'],0)
            path=Path(self.temp.name)/'before-restore.npz';self.net.save(path);self.net.load(path)
            self.cost.set_parameters(target=-1)
            self.net.train(cost=self.cost,**options)
            restored=self.net.state_dict()
            self.assertEqual(restored['metadata']['updates'],1000)
            np.testing.assert_array_equal(restored['weights'][0],before['weights'][0])
            np.testing.assert_array_equal(restored['optimizer_state'][0]['steps'],np.float32(1e-5))
            for values in restored['gradient_history'][0].values():np.testing.assert_array_equal(values,0)
            self.net.train(cost=self.cost,**options)
            advanced=self.net.state_dict()
            np.testing.assert_allclose(advanced['weights'][0],before['weights'][0]-np.float32(1e-5),rtol=0,atol=1e-8)

    def test_restore_legacy_array_cost_batch_and_split_are_identical(self):
        self.initialize(.5,0,0);initial=self.net.state_dict()
        other=Network([1,1],session=self.session).name_outputs('u');other.load_state_dict(initial)
        target=Data(np.ones((1,1,1),np.float32),[[0,0]],session=self.session).name_variables('x')
        left=self.net.cost(inputs=self.data,targets=target,weights=np.ones((1,1),np.float32))
        right=other.cost(inputs=self.data,targets=target,weights=np.ones((1,1),np.float32))
        options=dict(optimizer='rprop',verbose=0,restore_every=3,restore_step=1e-5)
        with contextlib.redirect_stdout(io.StringIO()):
            self.net.train(cost=left,epochs=7,**options)
            for epochs in [2,2,3]:other.train(cost=right,epochs=epochs,**options)
        a,b=self.net.state_dict(),other.state_dict()
        np.testing.assert_array_equal(a['weights'][0],b['weights'][0])
        np.testing.assert_array_equal(a['optimizer_state'][0]['steps'],b['optimizer_state'][0]['steps'])
        for key in a['gradient_history'][0]:np.testing.assert_array_equal(a['gradient_history'][0][key],b['gradient_history'][0][key])

    def test_restore_validation_and_older_server_capability(self):
        cases=[dict(restore_every=x,restore_step=1e-5) for x in [-1,True,1.5]]
        cases += [dict(restore_every=1,restore_step=x) for x in [0,-1,float('nan'),float('inf'),1e-50,1e39]]
        cases += [dict(restore_every=1),dict(restore_step=1e-5)]
        for options in cases:
            with self.subTest(options=options),self.assertRaises(ValueError):
                self.net.train(cost=self.cost,epochs=1,**options)
        with self.assertRaisesRegex(ValueError,'RPROP'):
            self.net.train(cost=self.cost,epochs=1,optimizer='adam',restore_every=1,restore_step=1e-5)
        capabilities=self.session._capabilities
        try:
            self.session._capabilities &= ~8
            with self.assertRaisesRegex(ValueError,'capability'):
                self.net.train(cost=self.cost,epochs=1,restore_every=1,restore_step=1e-5)
            # With the option disabled the original request format still works.
            self.update(0)
        finally:self.session._capabilities=capabilities


if __name__=='__main__':unittest.main()
