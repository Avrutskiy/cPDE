"""Explicit measures and coordinate adapters; no geometry is inferred from a field."""
import itertools
import math
import uuid
import weakref
import numpy as np
from . import _prepare as prep


def _readonly(value, dtype=np.float64):
    a = np.array(value, dtype=dtype, copy=True)
    # An immutable buffer also prevents callers from re-enabling write access.
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


def table(derivatives, *, variables):
    """Build (order, variable counts...) rows from an explicit derivative list.

    '' requests values; 'xxy' requests two x derivatives and one y derivative.
    Values and every lower dependency are included automatically. Rows are sorted
    by total order, then by the count tuple in the supplied variable order.
    """
    if not isinstance(variables, str) or not 1 <= len(variables) <= 26:
        raise ValueError('variables must contain 1..26 unique ASCII letters')
    prep.names(variables, len(variables), 'variables')
    if isinstance(derivatives, (str, bytes)):
        raise ValueError('derivatives must be a list of suffixes, not a string or maximum order')
    try:
        suffixes = list(derivatives)
    except TypeError as e:
        raise ValueError('derivatives must be a list of suffixes, not a maximum order') from e
    counts = {(0,) * len(variables)}
    for suffix in suffixes:
        if not isinstance(suffix, str) or any(ch not in variables for ch in suffix) or len(suffix) > 6:
            raise ValueError('derivatives must be variable suffixes of order at most six')
        key = tuple(suffix.count(v) for v in variables)
        counts.update(itertools.product(*(range(c+1) for c in key)))
    return prep.table(np.array([(sum(c), *c) for c in sorted(counts, key=lambda c: (sum(c), c))], dtype=np.int32))


class Quadrature:
    def __setattr__(self, name, value):
        if name in ('points','weights','variables','dimension','N','T','vertices','cells','cell_ids','rule','identity','revision') and self.__dict__.get('_sealed',False):
            raise AttributeError('quadrature geometry is read-only; use replace()')
        object.__setattr__(self,name,value)

    def __init__(self, *, points, variables, dimension, weights=None, cells=None,
                 triangulation=None, rule=None, N=None, T=None):
        p = np.asarray(points, dtype=np.float64)
        if p.ndim != 2 or not 1 <= p.shape[1] <= 3 or not np.isfinite(p).all():
            raise ValueError('points must be finite with shape (N, d), d in 1..3')
        with np.errstate(over='ignore'):
            if not np.isfinite(p.astype(np.float32)).all():
                raise ValueError('coordinates must be representable in float32')
        self.variables = prep.names(variables, p.shape[1], 'variables')
        if isinstance(dimension, bool) or not isinstance(dimension, (int, np.integer)) or not 0 <= dimension <= p.shape[1]:
            raise ValueError('dimension must be an integer between zero and ambient dimension')
        self.dimension = int(dimension)
        if sum(x is not None for x in (weights, cells, triangulation)) != 1:
            raise ValueError('select exactly one of weights, cells, or triangulation')
        self.vertices = self.cells = self.cell_ids = None
        inferred_n = inferred_t = None
        if weights is not None:
            if rule is not None: raise ValueError('rule is not used with supplied weights')
            w = np.asarray(weights, dtype=np.float64)
            if w.shape not in ((), (len(p),)) or not np.isfinite(w).all():
                raise ValueError('weights must be a finite scalar or a vector with one entry per point')
            if w.ndim == 0:
                w = np.full(len(p), w.item(), dtype=np.float64)
            self.rule = None
        else:
            k = self.dimension
            if k == 0: raise ValueError('zero-dimensional quadrature requires supplied weights')
            if triangulation is not None:
                if triangulation != 'delaunay' or k != p.shape[1] or k not in (2, 3):
                    raise ValueError('basic Delaunay supports only planar regions and 3D volumes')
                try:
                    from scipy.spatial import Delaunay, QhullError
                except ImportError as e:
                    raise ImportError('Delaunay requires pip install cpde-interface[geometry]') from e
                try:
                    mesh = Delaunay(p)
                except QhullError as e:
                    raise ValueError('Delaunay failed: points are insufficient or degenerate') from e
                if len(mesh.coplanar) or len(np.unique(mesh.simplices)) != len(p):
                    raise ValueError('Delaunay omitted points; supply explicit connectivity')
                cells = mesh.simplices
            c = np.asarray(cells)
            if c.ndim != 2 or c.shape[1] != k+1 or not np.issubdtype(c.dtype, np.integer):
                raise ValueError('cells must be integer indices with dimension+1 columns')
            if np.any(c < 0) or np.any(c >= len(p)):
                raise ValueError('cell index out of bounds')
            canonical = np.sort(c, axis=1)
            if np.any(np.diff(canonical, axis=1) == 0) or len(np.unique(canonical, axis=0)) != len(c):
                raise ValueError('repeated cell vertices or duplicate cells')
            corners = p[c]
            edges = corners[:, 1:] - corners[:, :1]
            singular = np.linalg.svd(edges, compute_uv=False)
            if len(c) and np.any(singular[:, -1] <= 1e-12 * singular[:, 0]):
                raise ValueError('degenerate simplex')
            measure = np.prod(singular, axis=1) / math.factorial(k)
            rule = 'centroid' if rule is None else rule
            if rule not in ('centroid', 'vertices'): raise ValueError('rule must be centroid or vertices')
            repetitions = 1 if rule == 'centroid' else k+1
            ids = np.repeat(np.arange(len(c)), repetitions)
            if k == 1:
                inferred_t = edges[:, 0] / measure[:, None]
                if p.shape[1] == 2:
                    inferred_n = np.stack([inferred_t[:, 1], -inferred_t[:, 0]], axis=1)
            elif k == 2 and p.shape[1] == 3:
                inferred_n = np.cross(edges[:, 0], edges[:, 1]) / (2*measure[:, None])
            if inferred_n is not None: inferred_n = inferred_n[ids]
            if inferred_t is not None: inferred_t = inferred_t[ids]
            self.vertices, self.cells = _readonly(p), _readonly(c, np.int32)
            self.cell_ids = _readonly(ids, np.int32)
            p = corners.mean(axis=1) if rule == 'centroid' else corners.reshape(-1, p.shape[1])
            w = np.repeat(measure / repetitions, repetitions)
            self.rule = rule
        self.points, self.weights = _readonly(p), _readonly(w)
        self.N = self._direction(inferred_n if N is None else N, 'N')
        self.T = self._direction(inferred_t if T is None else T, 'T')
        if self.N is not None and self.T is not None and not np.allclose(np.sum(self.N*self.T, axis=1), 0, atol=1e-6):
            raise ValueError('N and T must be orthogonal')
        self.identity, self.revision = str(uuid.uuid4()), 0
        self._bindings = weakref.WeakKeyDictionary()
        self._sealed = True

    def _direction(self, value, name):
        if value is None: return None
        a = np.asarray(value, dtype=np.float64)
        if a.shape != self.points.shape or not np.isfinite(a).all() or not np.allclose(np.linalg.norm(a, axis=1), 1, rtol=1e-6, atol=1e-6):
            raise ValueError(name + ' must contain unit vectors at final quadrature points')
        return _readonly(a)

    def replace(self, **geometry):
        geometry.setdefault('variables', self.variables)
        geometry.setdefault('dimension', self.dimension)
        replacement = type(self)(**geometry)
        if replacement.variables != self.variables or replacement.dimension != self.dimension:
            raise ValueError('replacement must preserve variables and dimension')
        if (replacement.N is None) != (self.N is None) or (replacement.T is None) != (self.T is None):
            raise ValueError('replacement must preserve the direction field schema')
        identity, revision, bindings = self.identity, self.revision+1, self._bindings
        self.__dict__.update(replacement.__dict__)
        self.__dict__.update(identity=identity, revision=revision, _bindings=bindings)
        return self

    def _data(self, session, suffixes):
        from .api import Data
        schema = table(suffixes, variables=self.variables)
        if not len(self.points): return None
        cache = self._bindings.setdefault(session, {})
        key = (self.revision, schema.tobytes())
        if key not in cache:
            arrays = np.zeros((len(schema), len(self.variables), len(self.points)), dtype=np.float32)
            arrays[0] = self.points.T
            for i, row in enumerate(schema):
                if row[0] == 1: arrays[i, np.argmax(row[1:]), :] = 1
            cache[key] = Data(arrays, schema, session=session).name_variables(self.variables)
        return cache[key]

    def _prune(self, session):
        cache = self._bindings.get(session,{})
        for key,data in list(cache.items()):
            if key[0] == self.revision: continue
            try: data.close()
            except RuntimeError as e:
                if 'retained by a cost' not in str(e): raise
            else: del cache[key]

    def save(self, path):
        fields = dict(points=self.points, weights=self.weights, variables=self.variables,
                      dimension=self.dimension, identity=self.identity, revision=self.revision)
        if self.N is not None: fields['N'] = self.N
        if self.T is not None: fields['T'] = self.T
        np.savez(path, **fields)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            q = cls(points=z['points'], weights=z['weights'], variables=str(z['variables']),
                    dimension=int(z['dimension']), N=z['N'] if 'N' in z else None,
                    T=z['T'] if 'T' in z else None)
            q.__dict__.update(identity=str(z['identity']), revision=int(z['revision']))
            return q
