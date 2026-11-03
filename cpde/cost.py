"""Scalar cost graphs and compiled integral leaves."""
import hashlib
import json
import numbers
import weakref
import numpy as np
from ._wire import Writer
from ._expressions import Expression, FUNCTIONS
from . import _prepare as prep


class CostOperators:
    def _check_release(self):
        if any(not p._closed for p in getattr(self,'_parents',())):
            raise ValueError('cost is retained by a live graph; close that graph first')
    def __bool__(self): raise TypeError('a cost is not a Python boolean; use evaluate()')
    def __float__(self): raise TypeError('use cost.evaluate() explicitly')
    def __add__(self, b): return GraphCost(3, self, b)
    def __radd__(self, a): return self if isinstance(a, numbers.Real) and a == 0 else GraphCost(3, a, self)
    def __sub__(self, b): return GraphCost(4, self, b)
    def __rsub__(self, a): return GraphCost(4, a, self)
    def __mul__(self, b): return GraphCost(5, self, b)
    def __rmul__(self, a): return GraphCost(5, a, self)
    def __truediv__(self, b): return GraphCost(6, self, b)
    def __rtruediv__(self, a): return GraphCost(6, a, self)
    def __neg__(self): return GraphCost(8, self)
    def __pow__(self, exponent):
        if not isinstance(exponent, numbers.Real) or not np.isfinite(exponent): raise ValueError('cost powers require a finite real constant')
        return GraphCost(7, self, parameter=float(exponent))
    def with_variation(self, correction): return GraphCost(15, self, correction)
    def evaluate(self, *, gradient=False):
        net, leaves, graph, _ = prepare_graph(self)
        _require_graph(net)
        for leaf in leaves:
            if leaf.inputs is not None:
                net._context(leaf.inputs, leaf._rows, leaf.chunk_size, backward=gradient, track=True)
        w = Writer().put('Q', net._id).array([c._id for c in leaves], 'u8').put('i', int(gradient)).raw(graph)
        try: r = net.session._request(14, w.bytes())
        except RuntimeError as e:
            raise RuntimeError(str(e) + ' (costs: ' + ', '.join(f'{i}={c.label}' for i,c in enumerate(leaves)) + ')') from e
        value = r.get('d')
        if not gradient: r.end(); return value
        count = r.get('i'); arrays = [r.array('f4') for _ in range(count)]; r.end()
        return dict(value=value, gradient=arrays)
    def compile(self):
        net, leaves, _, _ = prepare_graph(self)
        _require_graph(net)
        for leaf in leaves:
            if isinstance(leaf, IntegralCost):
                r = net.session._request(13, Writer().put('Q', leaf._id).put('i',1).bytes())
                leaf._generated_source, leaf._compile_log = r.string(), r.string(); r.end()
        return self


def _require_graph(net):
    if not net.session._capabilities & 2: raise ValueError('cost graphs require an integral-capable CPDE/3 executable')


class GraphCost(CostOperators):
    def __setattr__(self,name,value):
        if name in ('net','op','a','b','parameter') and name in self.__dict__:
            raise AttributeError('cost graph nodes are immutable')
        object.__setattr__(self,name,value)
    def __init__(self, op, a, b=None, parameter=0):
        for child in (a, b):
            if child is not None and not isinstance(child, CostOperators):
                if not isinstance(child, numbers.Real) or not np.isfinite(child): raise ValueError('cost operands must be costs or finite scalars')
        networks = [_network(child) for child in (a, b) if isinstance(child, CostOperators)]
        if not networks or any(n is not networks[0] for n in networks): raise ValueError('all costs must belong to one network')
        self.net, self.op, self.a, self.b, self.parameter = networks[0], op, a, b, parameter
        self._closed = False
        for child in (a,b):
            if isinstance(child,CostOperators):
                if not hasattr(child,'_parents'): child._parents=weakref.WeakSet()
                child._parents.add(self)
    def close(self):
        self._check_release();self._closed = True


def _network(cost):
    if hasattr(cost, 'net'): return cost.net
    return cost.terms[0].net


def prepare_graph(root):
    from .api import CostSum
    net = _network(root)
    if net._closed: raise ValueError('network is closed')
    nodes = []; constants = []; leaves = []; memo = {}; visiting = set()
    def visit(node):
        if isinstance(node, numbers.Real):
            nodes.append([0, -1, -1]); constants.append(float(node)); return len(nodes)-1
        if not isinstance(node, CostOperators) or _network(node) is not net or getattr(node, '_closed', False):
            raise ValueError('graph contains a closed cost or foreign network')
        key = id(node)
        if key in visiting: raise ValueError('cyclic cost graph')
        if key in memo: return memo[key]
        visiting.add(key)
        if isinstance(node, CostSum):
            indices = [visit(c) for c in node.terms]
            result = indices[0]
            for index in indices[1:]:
                nodes.append([3, result, index]); constants.append(0); result = len(nodes)-1
        elif isinstance(node, GraphCost):
            a = visit(node.a); b = -1 if node.b is None else visit(node.b)
            nodes.append([node.op, a, b]); constants.append(node.parameter); result = len(nodes)-1
        else:
            if isinstance(node, IntegralCost): node._sync()
            else: net._inputs(node.inputs)
            leaves.append(node); nodes.append([1, len(leaves)-1, -1]); constants.append(0); result = len(nodes)-1
        visiting.remove(key); memo[key] = result; return result
    root_index = visit(root)
    if len(nodes)>4096: raise ValueError('cost graph exceeds 4096 nodes')
    graph = Writer().array(nodes, 'i4').array(constants, 'f8').put('i', root_index).bytes()
    binding = hashlib.sha256(graph + ''.join(c.signature for c in leaves).encode()).hexdigest()
    structures = []
    for c in leaves:
        if isinstance(c,IntegralCost):
            structures.append(json.dumps(dict(expression=c.expression.inspect(),layout=c._layouts,chunk=c.chunk_size,
                geometry=None if c.quadrature is None else c.quadrature.identity,
                inputs=None if c._explicit_inputs is None else c._explicit_inputs._identity),sort_keys=True))
        else:
            structures.append(hashlib.sha256(c._semantic).hexdigest()+c.inputs._identity+
                ('' if c.targets is None else c.targets._identity))
    structural = hashlib.sha256(graph + ''.join(structures).encode()).hexdigest()
    signature = structural + '|' + binding
    return net, leaves, graph, signature


class IntegralCost(CostOperators):
    def __init__(self, net, *, quadrature, inputs, expression, parameters=None, freeze=(), chunk_size=None, name=None):
        from .api import _integer
        self.net, self.quadrature, self._explicit_inputs = net, quadrature, inputs
        if quadrature is None and inputs is None: raise ValueError('an integral needs quadrature or input Data')
        if inputs is None and net.layers[0] != len(quadrature.variables):
            raise ValueError('coordinate quadrature requires one input channel per variable')
        if not net.session._capabilities & 4: raise ValueError('expressions require a CPDE/3 executable with CPU expression support or CUDA NVRTC')
        variables = quadrature.variables if quadrature is not None else inputs.variables
        self.expression = Expression(expression, variables, net.outputs, parameters, freeze)
        self.parameters = {k: np.array(v, dtype=np.float64, copy=True) for k, v in (parameters or {}).items()}
        self._layouts = {k: v.ndim == 0 for k, v in self.parameters.items()}
        self.chunk_size = 0 if chunk_size is None else _integer(chunk_size, 'chunk_size', 1)
        self.label, self.targets = name or expression, None
        self._closed, self._id, self._revision = False, None, None
        self._parameter_revision = 0
        self._generated_source = self._compile_log = None
        self._sync()

    def _sync(self):
        if self._closed: raise ValueError('cost is closed')
        q = self.quadrature
        revision = (q.revision if q is not None else 0, self._parameter_revision)
        if self._revision == revision:
            if self.inputs is not None: self.net._inputs(self.inputs)
            return
        inputs = self._explicit_inputs
        if inputs is None: inputs = q._data(self.net.session, self.expression.suffixes)
        if inputs is not None:
            self.net._inputs(inputs)
            if inputs.variables != self.expression.variables: raise ValueError('input variables do not match quadrature')
        count = len(q.points) if q is not None else inputs.sample_count
        for name, value in self.parameters.items():
            if value.shape not in ((),(count,)) or not np.isfinite(value).all():
                raise ValueError('parameter '+name+' must be finite scalar or final-point vector')
        if inputs is not None and count != inputs.sample_count: raise ValueError('quadrature and input sample counts differ')
        nodes, constants, fields = self.expression.bind(inputs, q, count, self.parameters)
        rows = []
        if inputs is not None:
            requested = [int(a)//self.net.layers[-1] for op,a,mask in nodes if op == 1]
            rows = prep.closure(inputs.derivatives, requested)
            self.net._context(inputs, rows, self.chunk_size)
        if len(fields) and count: prep.checked_product(len(fields), count, name='integral fields')
        weights = np.ones(count) if q is None else q.weights
        w = Writer().put('Q', self.net._id).put('Q', inputs._id if inputs is not None else 0)
        w.put('i', self.chunk_size).put('i', count).array(nodes, 'i4').array(constants, 'f8')
        w.put('i', len(fields)).array(fields, 'f8').array(weights, 'f8')
        r = self.net.session._request(12, w.bytes()); new_id = r.get('Q'); r.end()
        old_id = self._id
        self._id, self.inputs, self._revision = new_id, inputs, revision
        self._rows = rows
        self._generated_source = self._compile_log = None
        remapped = np.array(nodes, dtype=np.int32, copy=True)
        for node in remapped:
            if node[0] == 1 and inputs is not None:
                node[1] = self._rows.index(int(node[1])//self.net.layers[-1])*self.net.layers[-1] + int(node[1])%self.net.layers[-1]
        self.net.session._track_program(hashlib.sha256(remapped.tobytes()+np.asarray(constants, dtype=np.float64).tobytes()).digest())
        self._semantic = hashlib.sha256(w.bytes()[16:]).hexdigest()
        if old_id is not None: self.net.session._request(11, Writer().put('i', 2).put('Q', old_id).bytes()).end()
        if q is not None: q._prune(self.net.session)

    @property
    def signature(self):
        q = self.quadrature
        geometry = '' if q is None else f'{q.identity}:{q.revision}'
        data = '' if self._explicit_inputs is None else f'{self.inputs._identity}:{self.inputs._version}'
        return hashlib.sha256((self._semantic+geometry+data).encode()).hexdigest()

    def set_parameters(self, **parameters):
        if set(parameters) - set(self.parameters): raise ValueError('unknown parameter')
        candidate = dict(self.parameters)
        for k, v in parameters.items():
            a = np.array(v, dtype=np.float64, copy=True)
            if (a.ndim==0) != self._layouts[k]: raise ValueError('parameter scalar/array layout cannot change')
            candidate[k] = a
        old = self.parameters; self.parameters = candidate; self._parameter_revision += 1
        try: self._sync()
        except Exception:
            self.parameters = old; self._parameter_revision -= 1; raise
        return self

    def inspect(self):
        if self._closed: raise ValueError('cost is closed')
        self._sync()
        if self._generated_source is None:
            r = self.net.session._request(13,Writer().put('Q',self._id).put('i',0).bytes())
            self._generated_source,self._compile_log = r.string(),r.string();r.end()
        info = self.expression.inspect()
        info.update(source=self._generated_source, compile_log=self._compile_log)
        return info

    def close(self):
        if not self._closed:
            self._check_release()
            self.net.session._request(11, Writer().put('i', 2).put('Q', self._id).bytes()).end()
            self._closed = True


def unary(name, cost):
    if not isinstance(cost, CostOperators): raise TypeError(name + ' expects a cost')
    return GraphCost(FUNCTIONS[name], cost)

def exp(cost): return unary('exp', cost)
def log(cost): return unary('log', cost)
def sqrt(cost): return unary('sqrt', cost)
def sin(cost): return unary('sin', cost)
def cos(cost): return unary('cos', cost)
def tanh(cost): return unary('tanh', cost)


def make_integral(net, *, inputs, quadrature, expression, targets, equations, parameters,
                  rhs, weights, vector_a, vector_b, freeze, chunk_size, name, variation):
    from .quadrature import Quadrature
    from .api import Data
    if quadrature is not None and not isinstance(quadrature, Quadrature): raise ValueError('quadrature must be a Quadrature')
    if inputs is not None: net._inputs(inputs)
    if quadrature is None and inputs is None: raise ValueError('supply quadrature or inputs')
    if sum(x is not None for x in (expression, targets, equations, vector_a)) != 1:
        raise ValueError('select exactly one cost form')
    count = len(quadrature.points) if quadrature is not None else inputs.sample_count
    variables = quadrature.variables if quadrature is not None else inputs.variables
    params = dict(parameters or {})
    def field(value, nonnegative=False):
        a = np.asarray(value, dtype=np.float64)
        if a.shape not in ((), (count,)) or not np.isfinite(a).all() or (nonnegative and np.any(a<0)):
            raise ValueError('density fields must be finite scalars or final-point vectors; density weights must be nonnegative')
        key = 'cpdeField' + str(len(params))
        while key in params: key += 'X'
        params[key] = a
        return key
    if expression is not None:
        if any(x is not None for x in (weights, rhs, vector_b)):
            raise ValueError('put coefficients in expression; rhs, weights and vector_b are not accepted')
    elif targets is not None:
        if parameters is not None or rhs is not None or vector_b is not None: raise ValueError('invalid matching arguments')
        if not isinstance(targets, Data) or targets.session is not net.session: raise ValueError('targets must be Data in this session')
        targets._check()
        if targets.variables != variables or targets.sample_count != count or targets.shape[1] != net.layers[-1]: raise ValueError('target schema mismatch')
        w = np.asarray(weights, dtype=np.float64); d = len(targets.derivatives); o = net.layers[-1]
        if w.shape not in ((o,d), (o,d,count)): raise ValueError('matching weights must cover every output and derivative')
        if inputs is not None and set(map(tuple,inputs.derivatives)) != set(map(tuple,targets.derivatives)): raise ValueError('target derivative schema mismatch')
        values = targets.to_numpy(); parts = []
        for j, row in enumerate(targets.derivatives):
            suffix = ''.join(v*int(n) for v,n in zip(variables,row[1:]))
            for i, output in enumerate(net.outputs):
                symbol = output + ('_'+suffix if suffix else '')
                parts.append(f'{field(w[i,j], True)}*({symbol}-{field(values[j,i])})**2')
        expression = '+'.join(parts)
    elif equations is not None:
        if vector_b is not None or not isinstance(equations,dict) or not equations: raise ValueError('invalid equations')
        rhs, weights = {} if rhs is None else rhs, {} if weights is None else weights
        if not isinstance(rhs,dict) or not isinstance(weights,dict) or set(rhs)-set(equations) or set(weights)-set(equations): raise ValueError('rhs/weights must map equation names')
        expression = '+'.join(f'{field(weights.get(key,1),True)}*(({value})-{field(rhs.get(key,0))})**2' for key,value in equations.items())
    else:
        if parameters is not None or not isinstance(vector_a,str) or vector_b is None: raise ValueError('invalid contraction arguments')
        components = [v.strip() for v in vector_a.split(',')]
        b = np.asarray(vector_b,dtype=np.float64)
        if b.shape == (len(components),): b = np.broadcast_to(b,(count,len(components)))
        if b.shape != (count,len(components)): raise ValueError('vector_b shape mismatch')
        dot = '+'.join(f'{field(b[:,i])}*{component}' for i,component in enumerate(components))
        expression = f'{field(1 if weights is None else weights,True)}*(({dot})-{field(0 if rhs is None else rhs)})**2'
    leaf = IntegralCost(net, quadrature=quadrature, inputs=inputs, expression=expression,
                        parameters=params, freeze=freeze or (), chunk_size=chunk_size, name=name)
    return leaf if variation is None else leaf.with_variation(variation)
