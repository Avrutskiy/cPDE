"""Preparation only: shapes, names, derivative closures, polynomial expressions."""
import ast
import itertools
import re
import numpy as np


def checked_product(*dimensions, name='buffer'):
    """Check the CUDA engine's signed-int indexing before packing or allocating."""
    size = 1
    for dimension in dimensions:
        dimension = int(dimension)
        if dimension <= 0:
            raise ValueError(f'{name} dimensions must be positive')
        size *= dimension
        if size > np.iinfo(np.int32).max:
            raise ValueError(f'{name} exceeds CUDA int32 indexing capacity')
    return size


def check_context(layers, derivatives, width, *, backward=True):
    checked_product(derivatives, derivatives, name='derivative tables')
    checked_product(derivatives, 64, name='derivative pointer stack')
    slots = checked_product(width, derivatives, 2 if backward else 1, name='context')
    checked_product(slots, max(layers), name='layer storage')
    return slots


def positive_float32(value, name):
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        result = float(np.float32(value))
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f'{name} must be positive and representable in float32')
    return result


def finite(value, name):
    with np.errstate(over='ignore', invalid='ignore'):
        a = np.asarray(value, dtype=np.float32)
    if not np.all(np.isfinite(a)):
        raise ValueError(f'{name} must contain finite float32 values')
    return a


def table(value):
    raw = np.asarray(value)
    if raw.ndim != 2 or raw.shape[0] == 0 or not 2 <= raw.shape[1] <= 27:
        raise ValueError('derivative table must have rows (order, variable counts...)')
    if not np.issubdtype(raw.dtype, np.integer):
        raise ValueError('derivative table entries must be integers')
    if np.any(raw < 0) or np.any(raw > 6):
        raise ValueError('derivative orders and counts must be between 0 and 6')
    t = np.array(raw, dtype=np.int32, copy=True)
    if np.any(t[:, 0] != t[:, 1:].sum(axis=1)):
        raise ValueError('derivative order must equal the sum of counts')
    keys = {tuple(row[1:]): i for i, row in enumerate(t)}
    if len(keys) != len(t):
        raise ValueError('duplicate derivative rows')
    for counts in keys:
        for sub in itertools.product(*(range(x + 1) for x in counts)):
            if sub not in keys:
                raise ValueError(f'missing derivative dependency {sub}')
    t.setflags(write=False)
    return t


def names(value, count, label):
    if not isinstance(value, str) or len(value) != count or not value.isascii() or not value.isalpha() or len(set(value)) != count:
        raise ValueError(f'{label} requires {count} unique single ASCII letters')
    return value


def sample_vector(value, count, name, *, scalar=True, nonnegative=False):
    a = finite(value, name)
    if scalar and a.ndim == 0:
        a = np.full(count, a, dtype=np.float32)
    if a.shape != (count,):
        raise ValueError(f'{name} must have shape ({count},)' + (' or be a scalar' if scalar else ''))
    if nonnegative and np.any(a < 0):
        raise ValueError(f'{name} must be nonnegative')
    return np.array(a, copy=True)


def closure(t, selected):
    keys = {tuple(row[1:]): i for i, row in enumerate(t)}
    wanted = {keys[(0,) * (t.shape[1] - 1)]}
    for i in selected:
        for sub in itertools.product(*(range(int(x) + 1) for x in t[i, 1:])):
            wanted.add(keys[sub])
    return sorted(wanted, key=lambda i: (int(t[i, 0]), i))


def component(token, inputs, outputs):
    if not inputs.variables or not outputs:
        raise ValueError('name differentiation variables and network outputs before using expressions')
    if set(inputs.variables) & set(outputs):
        raise ValueError('variable and output names must be disjoint')
    if not isinstance(token, str) or not re.fullmatch(r'[A-Za-z](?:_[A-Za-z]+)?', token):
        raise ValueError(f'invalid component {token!r}')
    if token[0] not in outputs:
        raise ValueError(f'unknown output in {token!r}')
    suffix = token[2:] if len(token) > 1 else ''
    if any(ch not in inputs.variables for ch in suffix):
        raise ValueError(f'unknown derivative variable in {token!r}')
    counts = tuple(suffix.count(ch) for ch in inputs.variables)
    rows = {tuple(r[1:]): i for i, r in enumerate(inputs.derivatives)}
    if counts not in rows:
        raise ValueError(f'derivative {token!r} is not supplied by the input data')
    return rows[counts] * len(outputs) + outputs.index(token[0])


class Polynomial:
    """Sparse monomials in output jets; coefficients are prepared sample vectors."""
    def __init__(self, inputs, outputs, parameters):
        self.inputs, self.outputs = inputs, outputs
        self.parameters = {}
        for name, value in parameters.items():
            if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9]*', name):
                raise ValueError('parameter names must be alphanumeric identifiers without underscores')
            if name in outputs or name in inputs.variables:
                raise ValueError(f'parameter {name!r} collides with a component or variable')
            self.parameters[name] = sample_vector(value, inputs.sample_count, f'parameter {name}')

    def parse(self, expression):
        if not isinstance(expression, str) or '=' in expression:
            raise ValueError('equations contain expressions without an equals sign')
        try:
            root = ast.parse('(' + expression.strip() + '\n)', mode='eval').body
        except SyntaxError as e:
            raise ValueError(f'invalid expression: {expression!r}') from e
        p = self.visit(root)
        constant = p.pop((), None)
        if constant is not None and np.any(constant != 0):
            raise ValueError('put external constant terms in rhs')
        return {key: finite(value, 'expression coefficient') for key, value in p.items() if np.any(value != 0)}

    @staticmethod
    def add(a, b, sign=1):
        out = dict(a)
        for key, value in b.items():
            out[key] = out.get(key, 0) + sign * value
        return out

    @staticmethod
    def multiply(a, b):
        out = {}
        for ka, va in a.items():
            for kb, vb in b.items():
                key = tuple(sorted(ka + kb))
                if len(key) > 2:
                    raise ValueError('only linear and quadratic output expressions are supported')
                out[key] = out.get(key, 0) + va * vb
        return out

    def visit(self, node):
        if isinstance(node, ast.Name):
            if node.id in self.parameters:
                return {(): self.parameters[node.id]}
            return {(component(node.id, self.inputs, self.outputs),): np.ones(self.inputs.sample_count, np.float32)}
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return {(): sample_vector(node.value, self.inputs.sample_count, 'constant')}
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            sign = -1 if isinstance(node.op, ast.USub) else 1
            return {k: sign * v for k, v in self.visit(node.operand).items()}
        if isinstance(node, ast.BinOp):
            a = self.visit(node.left)
            if isinstance(node.op, ast.Pow):
                if not isinstance(node.right, ast.Constant) or type(node.right.value) is not int or node.right.value != 2:
                    raise ValueError('only a squared expression (**2) is supported')
                return self.multiply(a, a)
            b = self.visit(node.right)
            if isinstance(node.op, ast.Add): return self.add(a, b)
            if isinstance(node.op, ast.Sub): return self.add(a, b, -1)
            if isinstance(node.op, ast.Mult): return self.multiply(a, b)
        raise ValueError('supported expression operations are +, -, *, and **2')
