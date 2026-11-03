"""Independent CUDA audit probes; reports defects without changing the engine.

Run with a native server or the serial host probe as the sole argument.
The Taylor-series oracle does not use the engine's hand-expanded chain rules.
"""
import contextlib
import io
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cpde import Data, Network, Quadrature, Session, table


def quiet_train(net, cost, **options):
    with contextlib.redirect_stdout(io.StringIO()):
        return net.train(cost=cost, epochs=1, verbose=0, **options)


def linear(session, weight=0.5):
    net = Network([1, 1], session=session).name_outputs('u')
    state = net.state_dict()
    state['weights'][0][:] = [weight, 0]
    net.load_state_dict(state)
    q = Quadrature(points=[[1]], weights=1, variables='x', dimension=1)
    return net, q


def edge_cases(session):
    results = {}
    net, q = linear(session)
    a = net.cost(quadrature=q, expression='u')
    graph = a.with_variation(-a)
    value = graph.evaluate()
    evaluated = graph.evaluate(gradient=True)
    results['cancelled_attachment'] = dict(value_only=value,
        with_gradient=evaluated['value'], gradient=evaluated['gradient'][0].tolist())

    # A zero total adjoint must not erase a value needed by another leaf's adjoint.
    b = net.cost(quadrature=q, expression='u**2')
    expression = a.with_variation(-a) * b
    result = expression.evaluate(gradient=True)
    results['cancelled_attachment_product'] = dict(value_only=expression.evaluate(),
        with_gradient=result['value'], gradient=result['gradient'][0].tolist(),
        expected_gradient=[0.5, 0.5])

    net, q = linear(session)
    target = net.cost(quadrature=q, expression='(u-target)**2', parameters={'target': 1})
    quiet_train(net, target, optimizer='rprop', reset_steps=0.1)
    before = net.state_dict()['weights'][0].copy()
    target.set_parameters(target=-1)
    quiet_train(net, target, optimizer='rprop', reset_steps=100)
    results['reset_steps_rollback'] = dict(before=before.tolist(),
        after=net.state_dict()['weights'][0].tolist(),
        old_step=0.1, new_step=100)

    # Opposite tiny gradients have a nonzero sign, but their float product underflows.
    net, q = linear(session)
    state = net.state_dict()
    state['metadata']['optimizer'] = 'rprop'
    state['optimizer_state'][0]['steps'][:] = 0.1
    state['gradient_history'][0]['previous'][:] = 1e-30
    net.load_state_dict(state)
    tiny = net.cost(quadrature=q, expression='-1e-30*u')
    quiet_train(net, tiny, optimizer='rprop')
    state = net.state_dict()
    results['rprop_sign_underflow'] = dict(after=state['weights'][0].tolist(),
        steps=state['optimizer_state'][0]['steps'].tolist(),
        expected_after=[0.6, 0.1], expected_steps=[0.05, 0.05])

    net, q = linear(session)
    huge = net.cost(quadrature=q, expression='1e25*u')
    quiet_train(net, huge, optimizer='adam')
    state = net.state_dict()
    results['adam_finite_gradient_overflow'] = dict(weights=state['weights'][0].tolist(),
        moments_finite=bool(np.isfinite(state['optimizer_state'][0]['v']).all()),
        v=[str(float(x)) for x in state['optimizer_state'][0]['v']],
        updates=state['metadata']['updates'])

    net, q = linear(session)
    numerator = net.cost(quadrature=q, expression='1e200*u')
    denominator = net.cost(quadrature=q, expression='1e200*(u+1)')
    inline = net.cost(quadrature=q, expression='1e200*u/(1e200*(u+1))')
    ratio = (numerator/denominator).evaluate(gradient=True)
    inline_ratio = inline.evaluate(gradient=True)
    results['division_partial_overflow'] = dict(value=ratio['value'],
        graph_gradient=ratio['gradient'][0].tolist(),
        expression_gradient=inline_ratio['gradient'][0].tolist(),
        expected_gradient=[1/1.5**2]*2)

    net, q = linear(session)
    state = net.state_dict()
    state['metadata']['optimizer'] = 'adam'
    state['optimizer_state'][0] = {'m': np.zeros(2, np.float32),
                                   'v': -np.ones(2, np.float32)}
    net.load_state_dict(state)
    objective = net.cost(quadrature=q, expression='u')
    quiet_train(net, objective, optimizer='adam')
    after = net.state_dict()
    results['negative_adam_variance_accepted'] = dict(
        weights=after['weights'][0].tolist(),
        variance=after['optimizer_state'][0]['v'].tolist(),
        updates=after['metadata']['updates'])

    # Saturation at representable preactivations.
    net = Network([1, 1, 1], session=session).name_outputs('u')
    state = net.state_dict()
    state['weights'][0][:] = [1, 0]
    state['weights'][1][:] = [1, 0]
    net.load_state_dict(state)
    q = Quadrature(points=[[-20], [20]], weights=1, variables='x', dimension=1)
    y = net.eval(q, derivatives=['', 'x']).to_numpy()
    results['sigmoid_asymmetric_saturation'] = dict(values=y[0, 0].tolist(),
        derivatives=y[1, 0].tolist(), expected_derivative=math.exp(-20)/(1+math.exp(-20))**2)
    return results


class Jets:
    """Dense normalized multivariate Taylor series through a total degree."""
    def __init__(self, counts):
        self.counts = [tuple(map(int, c)) for c in counts]
        self.degree = max(map(sum, self.counts))
        self.pairs = []
        for c in self.counts:
            self.pairs.append([(i, j) for i, a in enumerate(self.counts)
                              for j, b in enumerate(self.counts)
                              if tuple(x+y for x, y in zip(a, b)) == c])

    def multiply(self, a, b):
        return np.array([sum(a[i]*b[j] for i, j in pairs) for pairs in self.pairs])

    def sigmoid(self, a):
        # exp(-a) = exp(-a0) * sum_{k=0}^degree (-delta a)^k/k!
        zero = np.zeros_like(a); zero[0] = 1
        delta = -a.copy(); delta[0] = 0
        power = zero.copy(); exponential = zero.copy()
        for k in range(1, self.degree+1):
            power = self.multiply(power, delta)
            exponential += power/math.factorial(k)
        denominator = np.exp(-a[0])*exponential; denominator[0] += 1
        # 1/(d0 + delta d), by the finite geometric series in delta d.
        delta = -denominator/denominator[0]; delta[0] = 0
        power = zero.copy(); inverse = zero.copy()
        for _ in range(self.degree):
            power = self.multiply(power, delta); inverse += power
        return inverse/denominator[0]

    def network(self, arrays, neurons, linear_flags, inputs):
        current = inputs.astype(np.result_type(inputs, *arrays))
        for index, flat in enumerate(arrays):
            if not linear_flags[index]:
                current = np.stack([self.sigmoid(a) for a in current])
            matrix = flat.reshape((neurons[index+1], neurons[index]+1), order='F')
            current = matrix[:, :-1] @ current
            current[:, 0] += matrix[:, -1]
        return current


def high_order(session):
    results = []
    rng = np.random.default_rng(8031)
    for variables, architecture, linear_flags in [
            ('x', [1, 3, 2, 1], [1, 0, 0, 1]),
            ('xy', [2, 3, 2], [1, 0, 1]),
            ('xyz', [3, 2, 2, 1], [1, 0, 1, 1]),
            ('xyzabc', [6, 2, 1], [1, 0, 1])]:
        dimensions = len(variables)
        requested = [variables] if dimensions == 6 else [''.join(v*n for v, n in zip(variables, powers))
                     for powers in itertools.product(range(7), repeat=dimensions)
                     if sum(powers) == 6]
        schema = table(requested, variables=variables)
        jets = Jets(schema[:, 1:])
        factorials = np.array([math.prod(math.factorial(int(n)) for n in c)
                               for c in schema[:, 1:]])
        # Arbitrary, mutually consistent polynomial input features exercise every
        # partition term, including mixed derivatives, not just affine inputs.
        inputs = rng.uniform(-0.2, 0.2, (dimensions, len(schema)))
        net = Network(architecture, linear=linear_flags, session=session).name_outputs(
            'uv' if architecture[-1] == 2 else 'u')
        state = net.state_dict()
        for a in state['weights']: a[:] = rng.uniform(-0.65, 0.65, len(a))
        net.load_state_dict(state)
        raw = (inputs*factorials).T[:, :, None].astype(np.float32)
        inputs = raw[:, :, 0].T.astype(float)/factorials
        data = Data(raw, schema, session=session).name_variables(variables)
        actual = net.eval(data, chunk_size=1).to_numpy()[:, :, 0].T
        arrays = [a.astype(float) for a in state['weights']]
        expected = jets.network(arrays, architecture, linear_flags, inputs)*factorials
        errors = np.abs(actual-expected)/(1+np.abs(expected))
        # A signed linear functional over ALL jet outputs checks every adjoint.
        coef = rng.uniform(-0.5, 0.5, expected.shape)/factorials
        expressions = []
        parameters = {}
        for o, name in enumerate(net.outputs):
            for r, counts in enumerate(schema[:, 1:]):
                suffix = ''.join(v*int(n) for v, n in zip(variables, counts))
                component = name + ('_'+suffix if suffix else '')
                key = f'c{o}r{r}'; parameters[key] = coef[o, r]
                expressions.append(f'{key}*{component}')
        # Split to stay within the documented 512-node expression limit.
        leaves = [net.cost(inputs=data, expression='+'.join(expressions[k:k+40]),
                           parameters=parameters)
                  for k in range(0, len(expressions), 40)]
        cost = sum(leaves)
        gradient = cost.evaluate(gradient=True)['gradient']
        expected_gradient = []
        for layer, a in enumerate(arrays):
            g = np.zeros_like(a)
            for i in range(len(a)):
                perturbed = [w.astype(complex) for w in arrays]
                perturbed[layer][i] += 1e-25j
                y = jets.network(perturbed, architecture, linear_flags, inputs)*factorials
                g[i] = np.sum(coef*y).imag/1e-25
            expected_gradient.append(g)
        grad_error = max(float(np.max(np.abs(a-b)/(1+np.abs(b))))
                         for a, b in zip(gradient, expected_gradient))
        results.append(dict(variables=variables, architecture=architecture,
            derivative_rows=len(schema), forward_scaled_error=float(errors.max()),
            gradient_scaled_error=grad_error,
            passed=bool(errors.max()<5e-5 and grad_error<5e-5)))
    return results


def chunk_stress(session):
    rng = np.random.default_rng(918)
    net = Network([2, 7, 3, 1], session=session).name_outputs('u')
    state = net.state_dict()
    for a in state['weights']: a[:] = rng.uniform(-0.4, 0.4, len(a))
    net.load_state_dict(state)
    count = 10003
    q = Quadrature(points=rng.uniform(-1, 1, (count, 2)), weights=1/count,
                   variables='xy', dimension=2)
    expression = 'u**2+.2*u_xxx**2+.1*u_xxy**2+.1*u_yyyyyy**2'
    full = net.cost(quadrature=q, expression=expression)
    chunked = net.cost(quadrature=q, expression=expression, chunk_size=259)
    a = full.evaluate(gradient=True); b = chunked.evaluate(gradient=True)
    error = max(float(np.max(np.abs(x-y))) for x, y in zip(a['gradient'], b['gradient']))
    value_error = abs(a['value']-b['value'])
    return dict(samples=count, chunk=259, value_difference=value_error,
                gradient_max_difference=error, passed=bool(error<5e-5 and value_error<5e-7))


def main():
    with Session(sys.argv[1]) as session:
        result = dict(executable=sys.argv[1], edge_cases=edge_cases(session),
                      high_order=high_order(session), chunk_stress=chunk_stress(session))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
