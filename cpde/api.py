"""Python descriptions and RAM transport for an externally supplied CPDE server."""
from contextlib import contextmanager
from pathlib import Path
import collections
import hashlib
import json
import os
import queue
import shutil
import struct
import subprocess
import threading
import uuid
import warnings
import numpy as np
from . import _prepare as prep
from ._wire import Reader, Writer, decode_state, decode_metadata
from .cost import CostOperators, IntegralCost, GraphCost, prepare_graph, make_integral
from .quadrature import Quadrature


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum or value > 2**31-1:
        raise ValueError(f'{name} must be an integer >= {minimum} within int32 range')
    return int(value)


class Session:
    """Owns a persistent CPU or CUDA server. Never compiles the executable."""
    def __init__(self, executable=None, *, backend=None, device=0, cache_warning_threshold=64):
        if backend not in (None, 'cpu', 'cuda'):
            raise ValueError("backend must be 'cpu', 'cuda', or None")
        device = _integer(device, 'device')
        if backend == 'cpu' and device != 0:
            raise ValueError('the CPU backend accepts only device=0')
        if executable is None:
            selected = backend or 'cuda'
            variable = 'CPDE_CPU_SERVER' if selected == 'cpu' else 'CPDE_SERVER'
            program = 'cpde_cpu_server' if selected == 'cpu' else 'cpde_server'
            executable = os.environ.get(variable) or shutil.which(program)
            if not executable:
                raise FileNotFoundError(f'{program} not found; build it and pass its path, set {variable}, or add it to PATH')
            backend = selected
        self._lock, self._write_lock = threading.RLock(), threading.Lock()
        self._closed = False
        self._diagnostics = collections.deque(maxlen=100)
        self._cache_warning_threshold = _integer(cache_warning_threshold, 'cache_warning_threshold', 1)
        self._program_keys = set()
        self._program_cache_warned = False
        self._process = subprocess.Popen([str(Path(executable).resolve()), '--device', str(device)],
                                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        self._stderr = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr.start()
        self._responses = queue.Queue()
        self._stdout = threading.Thread(target=self._drain_stdout, daemon=True)
        self._stdout.start()
        try:
            r = self._request(1)
            self._protocol = r.string()
            if self._protocol not in ('CPDE/1', 'CPDE/2', 'CPDE/3') or r.get('i') != 6:
                raise RuntimeError('incompatible CPDE executable')
            self._capabilities = r.get('I') if self._protocol != 'CPDE/1' else 0
            r.end()
            self.backend = ('host-probe' if self._capabilities & 64 else
                            'cpu' if self._capabilities & 32 else 'cuda')
            if backend is not None and backend != self.backend:
                raise RuntimeError(f'requested {backend} backend, but executable reports {self.backend}')
        except BaseException:
            self._process.kill(); self._process.wait()
            self._stderr.join(); self._stdout.join()
            self._process.stdin.close(); self._process.stdout.close(); self._process.stderr.close()
            self._closed = True
            raise

    def _drain_stderr(self):
        for line in iter(self._process.stderr.readline, b''):
            self._diagnostics.append(line.decode('utf-8', 'replace').rstrip())

    def _send(self, op, body=b''):
        if self._closed: raise RuntimeError('session is closed')
        if len(body) + 4 > 4*1024**3: raise ValueError('command exceeds the CPDE frame limit')
        frame = struct.pack('<QI', len(body)+4, op) + body
        with self._write_lock:
            view = memoryview(frame)
            while view:
                n = self._process.stdin.write(view)
                if not n: raise RuntimeError('executable input pipe closed')
                view = view[n:]
            self._process.stdin.flush()

    def _read_exact(self, n):
        chunks = []
        while n:
            part = self._process.stdout.read(n)
            if not part:
                raise RuntimeError('CPDE executable stopped: ' + '\n'.join(self._diagnostics))
            chunks.append(part); n -= len(part)
        return b''.join(chunks)

    def _read_frame(self):
        size = struct.unpack('<Q', self._read_exact(8))[0]
        if not 4 <= size <= 4*1024**3: raise RuntimeError('invalid response frame')
        payload = self._read_exact(size)
        return struct.unpack_from('<I', payload)[0], Reader(payload[4:])

    def _drain_stdout(self):
        try:
            while True: self._responses.put(self._read_frame())
        except BaseException as e:
            self._responses.put(e)

    def _receive(self):
        value = self._responses.get()
        if isinstance(value, BaseException): raise value
        return value

    def _request(self, op, body=b''):
        with self._lock:
            self._send(op, body)
            kind, r = self._receive()
            if kind == 1: raise RuntimeError(r.string())
            if kind != 0: raise RuntimeError('unexpected response')
            return r

    def stop(self):
        """May be called by another thread; stops training after a common update."""
        self._send(99)

    def _track_program(self, key):
        if self._program_cache_warned: return
        self._program_keys.add(key)
        if len(self._program_keys) >= self._cache_warning_threshold:
            warnings.warn('Many distinct expressions were requested. The server retains '
                          'programs until Session.close(), even after costs close; reuse expressions '
                          'with parameters or start a new Session when practical.', RuntimeWarning, stacklevel=3)
            self._program_cache_warned = True

    def close(self):
        if self._closed: return
        try:
            self._request(0)
        finally:
            self._closed = True
            self._process.stdin.close()
            self._process.wait()
            self._stderr.join(); self._stdout.join()
            self._process.stdout.close(); self._process.stderr.close()

    def __enter__(self): return self
    def __exit__(self, *args): self.close()


class Data:
    """Numerical layout is (derivative, channel, sample); storage lives in the selected server backend."""
    def __init__(self, samples, derivatives, *, session):
        self.session, self.derivatives = session, prep.table(derivatives)
        a = prep.finite(samples, 'samples')
        if a.ndim != 3 or a.shape[0] != len(self.derivatives) or min(a.shape) <= 0:
            raise ValueError('samples must have shape (derivative, channel, sample) matching the table')
        self.shape = tuple(a.shape)
        prep.checked_product(*self.shape, name='input data')
        prep.checked_product(*self.derivatives.shape, name='derivative schema')
        self.variables = ''; self.channels = ''; self._closed = False
        self._identity, self._version = uuid.uuid4().hex, 0
        w = Writer()
        for x in (*self.shape, self.derivatives.shape[1]-1): w.put('i', x)
        w.array(self.derivatives, 'i4').array(a.transpose(0, 2, 1), 'f4')
        r = session._request(2, w.bytes()); self._id = r.get('Q'); r.end()

    @classmethod
    def _from_handle(cls, session, handle, source, outputs):
        obj = cls.__new__(cls)
        obj.session, obj._id = session, handle
        obj.derivatives = source.derivatives
        obj.shape = (source.shape[0], len(outputs), source.sample_count)
        obj.variables, obj.channels = source.variables, outputs
        obj._identity, obj._version, obj._closed = uuid.uuid4().hex, 0, False
        return obj

    @property
    def sample_count(self): return self.shape[2]

    def _check(self):
        if self._closed: raise RuntimeError('data is closed')

    def name_variables(self, letters):
        self.variables = prep.names(letters, self.derivatives.shape[1]-1, 'variables'); return self

    def to_numpy(self):
        """Detached host copy. Edits to this array do not modify Data."""
        self._check(); r = self.session._request(5, Writer().put('Q', self._id).bytes())
        a = r.array('f4'); r.end()
        return a.reshape(self.shape[0], self.shape[2], self.shape[1]).transpose(0, 2, 1).copy()

    def set_samples(self, samples):
        """Commit a host replacement and upload it once before subsequent commands."""
        self._check(); a = prep.finite(samples, 'samples')
        if a.shape != self.shape: raise ValueError('replacement must preserve data shape and derivative schema')
        w = Writer().put('Q', self._id).array(a.transpose(0, 2, 1), 'f4')
        self.session._request(6, w.bytes()).end(); self._version += 1

    @contextmanager
    def edit_host(self):
        a = self.to_numpy()
        yield a
        self.set_samples(a)

    def save(self, path):
        with open(path, 'wb') as f:
            np.savez(f, samples=self.to_numpy(), derivatives=self.derivatives,
                     variables=self.variables, channels=self.channels,
                     identity=self._identity, version=self._version)

    @classmethod
    def load(cls, path, *, session):
        with np.load(path, allow_pickle=False) as f:
            obj = cls(f['samples'], f['derivatives'], session=session)
            if str(f['variables']): obj.name_variables(str(f['variables']))
            obj.channels = str(f['channels']); obj._identity = str(f['identity']); obj._version = int(f['version'])
            return obj

    def close(self):
        if not self._closed:
            self.session._request(11, Writer().put('i', 0).put('Q', self._id).bytes()).end(); self._closed = True


class Cost(CostOperators):
    def __init__(self, net, inputs, target, label, body, semantic):
        self.net, self.inputs, self.targets, self.label = net, inputs, target, label
        self._semantic, self._closed = semantic, False
        r = net.session._request(7, body); self._id = r.get('Q'); r.end()

    @property
    def signature(self):
        h = hashlib.sha256(self._semantic)
        for d in (self.inputs, self.targets):
            if d is not None: h.update(f'{d._identity}:{d._version}'.encode())
        return h.hexdigest()

    def __add__(self, other):
        return CostSum([self]) + other if isinstance(other, (Cost, CostSum)) else GraphCost(3, self, other)
    def close(self):
        if not self._closed:
            self._check_release()
            self.net.session._request(11, Writer().put('i', 2).put('Q', self._id).bytes()).end(); self._closed = True


class CostSum(CostOperators):
    def __init__(self, terms): self.terms = tuple(terms)
    def __add__(self, other):
        if not isinstance(other, (Cost, CostSum)): return GraphCost(3, self, other)
        terms = other.terms if isinstance(other, CostSum) else (other,)
        if not all(isinstance(c, Cost) for c in terms): return NotImplemented
        joined = self.terms + tuple(terms)
        if any(c.net is not joined[0].net for c in joined): raise ValueError('all costs must belong to the same network')
        return CostSum(joined)


class Network:
    def __init__(self, layers, *, session, linear=None, state=None, seed=None):
        self.session = session
        self.layers = [_integer(x, 'layer width', 1) for x in layers]
        if not 2 <= len(self.layers) <= 256: raise ValueError('architecture requires 2 to 256 layers')
        for width in self.layers: prep.checked_product(2, width, name='initial layer storage')
        for width, height in zip(self.layers, self.layers[1:]):
            prep.checked_product(height, width+1, name='weights and biases')
        initial_count = sum(height*(width+1) for width, height in zip(self.layers, self.layers[1:]))
        if 4*initial_count + 16*len(self.layers) + 20 > 0xffffffff:
            raise ValueError('initial weights exceed protocol frame capacity')
        self.linear = [1] + [0]*(len(self.layers)-2) + [1] if linear is None else [int(x) for x in linear]
        if len(self.linear) != len(self.layers) or any(x not in (0, 1) for x in self.linear) or self.linear[0] != 1 or self.linear[-1] != 1:
            raise ValueError('linear flags must match layers, with linear input and output')
        self.outputs = ''; self._closed = False; self._last_blob = None; self._last_meta = None; self.checkpoint = None
        self.last_gradient_report = None
        self._context_keys = set()
        self._context_cache_warned = False
        rng = np.random.default_rng(seed)
        initial = [np.concatenate((rng.uniform(-2/np.sqrt(width), 2/np.sqrt(width), height*width),
                                   rng.uniform(-.1, .1, height))).astype(np.float32)
                   for width, height in zip(self.layers, self.layers[1:])]
        request = Writer().array(self.layers, 'i4').array(self.linear, 'i4')
        if session._capabilities & 16:
            for values in initial: request.array(values, 'f4')
        r = session._request(3, request.bytes())
        self._id = r.get('Q'); r.end()
        if not session._capabilities & 16:
            # Older executables have no creation-time initializer field.
            initialized = self.state_dict(); initialized['weights'] = initial
            self.load_state_dict(initialized)
        if state is not None:
            if isinstance(state, (str, os.PathLike)): self.load(state)
            else: self.load_state_dict(state)

    def name_outputs(self, letters):
        self.outputs = prep.names(letters, self.layers[-1], 'outputs'); return self

    def _inputs(self, inputs):
        if self._closed: raise RuntimeError('network is closed')
        if not isinstance(inputs, Data) or inputs.session is not self.session: raise ValueError('input Data must belong to this session')
        inputs._check()
        if inputs.shape[1] != self.layers[0]: raise ValueError('input channel count does not match network')
        if set(inputs.variables) & set(self.outputs): raise ValueError('variable and output names must be disjoint')

    def _context(self, data, rows, chunk, *, backward=True, track=False):
        width = min(chunk, data.sample_count) if chunk else data.sample_count
        prep.check_context(self.layers, len(rows), width, backward=backward)
        if track and not self._context_cache_warned:
            schema = data.derivatives[np.asarray(rows, dtype=int)].tobytes()
            self._context_keys.add((hashlib.sha256(schema).digest(), width, backward))
            if len(self._context_keys) >= self.session._cache_warning_threshold:
                warnings.warn('Many derivative/chunk contexts were requested for this network. '
                              'The server may retain their metadata until buffer growth or '
                              'Network.close(); reuse chunk sizes or recreate the network from a '
                              'checkpoint when practical.', RuntimeWarning, stacklevel=3)
                self._context_cache_warned = True

    def eval(self, data, *, chunk_size=None, derivatives=None):
        if isinstance(data, Quadrature):
            data = data._data(self.session, [''] if derivatives is None else derivatives)
            if data is None: raise ValueError('evaluation needs nonempty points; empty integrals still evaluate to zero')
        elif derivatives is not None: raise ValueError('derivatives argument is for Quadrature evaluation')
        self._inputs(data)
        chunk = 0 if chunk_size is None else _integer(chunk_size, 'chunk_size', 1)
        prep.checked_product(len(data.derivatives), self.layers[-1], data.sample_count, name='evaluation output')
        self._context(data, range(len(data.derivatives)), chunk, backward=False, track=True)
        r = self.session._request(4, Writer().put('Q', self._id).put('Q', data._id).put('i', chunk).bytes())
        handle = r.get('Q'); r.end()
        # Unnamed outputs still have the correct number of numerical channels.
        result = Data._from_handle(self.session, handle, data, self.outputs or '?'*self.layers[-1])
        result.channels = self.outputs
        return result

    def cost(self, *, inputs=None, quadrature=None, expression=None, targets=None, equations=None, parameters=None, rhs=None, weights=None,
             vector_a=None, vector_b=None, chunk_size=None, name=None, freeze=None, variation=None):
        if quadrature is not None or expression is not None or freeze is not None:
            return make_integral(self, inputs=inputs, quadrature=quadrature, expression=expression,
                targets=targets, equations=equations, parameters=parameters, rhs=rhs, weights=weights,
                vector_a=vector_a, vector_b=vector_b, chunk_size=chunk_size, name=name, freeze=freeze, variation=variation)
        self._inputs(inputs)
        count = int(targets is not None) + int(equations is not None) + int(vector_a is not None)
        if count != 1: raise ValueError('select exactly one cost form: targets, equations, or vector_a')
        if vector_b is not None and vector_a is None: raise ValueError('vector_b requires vector_a')
        chunk = 0 if chunk_size is None else _integer(chunk_size, 'chunk_size', 1)
        n, outputs = inputs.sample_count, self.layers[-1]
        d = len(inputs.derivatives)
        if targets is not None:
            if parameters is not None or rhs is not None: raise ValueError('matching does not accept parameters or rhs')
            if not isinstance(targets, Data) or targets.session is not self.session: raise ValueError('targets must be Data in the same session')
            targets._check()
            if targets.shape != (d, outputs, n): raise ValueError('target shape must cover all outputs, derivatives, and samples')
            if targets.variables != inputs.variables: raise ValueError('target differentiation variable names must match inputs')
            target_keys = {tuple(row): i for i, row in enumerate(targets.derivatives)}
            if set(target_keys) != set(map(tuple, inputs.derivatives)): raise ValueError('target derivative schema does not match inputs')
            w = prep.finite(weights, 'matching weights')
            if w.shape not in ((outputs, d), (outputs, d, n)) or np.any(w < 0):
                raise ValueError(f'matching weights must be nonnegative with shape ({outputs}, {d}) or ({outputs}, {d}, {n})')
            active = np.any(w != 0, axis=0)
            if active.ndim == 2: active = active.any(axis=1)
            rows = prep.closure(inputs.derivatives, np.flatnonzero(active))
            weight_samples = 1 if w.ndim == 2 else n
            packed = w[:, rows].T[:, None, :] if w.ndim == 2 else w[:, rows, :].transpose(1, 2, 0)
            detail = Writer().array([target_keys[tuple(inputs.derivatives[i])] for i in rows], 'i4').put('i', weight_samples).array(packed, 'f4')
            kind = 0; label = name or 'matching'
        else:
            if not self.outputs or not inputs.variables: raise ValueError('name outputs and variables before constructing expressions')
            if equations is not None:
                if not isinstance(equations, dict) or not equations or any(not isinstance(k, str) or not k for k in equations):
                    raise ValueError('equations must be a nonempty mapping of names to expressions')
                parser = prep.Polynomial(inputs, self.outputs, {} if parameters is None else parameters)
                polynomials = [parser.parse(expression) for expression in equations.values()]
                rhs = {} if rhs is None else rhs; weights = {} if weights is None else weights
                if not isinstance(rhs, dict) or not isinstance(weights, dict) or set(rhs)-set(equations) or set(weights)-set(equations):
                    raise ValueError('rhs and weights must map known equation names')
                rv = np.stack([prep.sample_vector(rhs[k], n, f'rhs {k}', scalar=False) if k in rhs else np.zeros(n, np.float32) for k in equations])
                wv = np.stack([prep.sample_vector(weights.get(k, 1), n, f'weight {k}', nonnegative=True) for k in equations])
                label = name or 'equations[' + ','.join(equations) + ']'
            else:
                if parameters is not None: raise ValueError('scalar product does not accept parameters')
                if not isinstance(vector_a, str): raise ValueError('vector_a must be a comma-separated component string')
                selected = [prep.component(token.strip(), inputs, self.outputs) for token in vector_a.split(',')]
                bv = prep.finite(vector_b, 'vector_b')
                if bv.shape == (len(selected),): bv = np.broadcast_to(bv, (n, len(selected)))
                if bv.shape != (n, len(selected)): raise ValueError('vector_b must have shape (K,) or (N, K)')
                poly = {}
                for j, index in enumerate(selected): poly[(index,)] = poly.get((index,), 0) + bv[:, j]
                polynomials = [poly]
                rv = np.zeros((1, n), np.float32) if rhs is None else prep.sample_vector(rhs, n, 'rhs', scalar=False)[None, :]
                wv = prep.sample_vector(1 if weights is None else weights, n, 'weights', nonnegative=True)[None, :]
                label = name or 'scalar_product'
            used = {index // outputs for e, poly in enumerate(polynomials) if np.any(wv[e] != 0)
                    for monomial, co in poly.items() if np.any(co != 0) for index in monomial}
            rows = prep.closure(inputs.derivatives, used)
            remap = {old: new for new, old in enumerate(rows)}
            terms, coeffs = [], []
            for e, poly in enumerate(polynomials):
                if not np.any(wv[e] != 0): continue
                for monomial, co in poly.items():
                    if not np.any(co != 0): continue
                    indices = [remap[x//outputs]*outputs+x%outputs for x in monomial]
                    terms.append((e, indices[0], indices[1] if len(indices) == 2 else -1)); coeffs.append(co)
            detail = Writer().put('i', len(polynomials)).array(terms, 'i4').array(coeffs, 'f4').array(rv, 'f4').array(wv, 'f4')
            kind = 1
        semantic = Writer().put('i', kind).put('i', chunk).array(rows, 'i4').raw(detail.bytes()).bytes()
        self._context(inputs, rows, chunk)
        body = Writer().put('Q', self._id).put('Q', inputs._id).put('Q', targets._id if targets is not None else 0).raw(semantic).bytes()
        leaf = Cost(self, inputs, targets, label, body, semantic)
        leaf._rows, leaf.chunk_size = rows, chunk
        return leaf if variation is None else leaf.with_variation(variation)

    def _costs(self, cost):
        if isinstance(cost, (IntegralCost, GraphCost)):
            net, terms, _, signature = prepare_graph(cost)
            if net is not self: raise ValueError('cost belongs to another network')
            return terms, signature
        terms = cost.terms if isinstance(cost, CostSum) else (cost,)
        if not terms or any(not isinstance(c, Cost) or c.net is not self or c._closed for c in terms):
            raise ValueError('training requires live costs belonging to this network')
        for c in terms: self._inputs(c.inputs)
        signature = hashlib.sha256(''.join(c.signature for c in terms).encode()).hexdigest()
        return terms, signature

    def train(self, *, cost, epochs, optimizer=None, state='continue', reset_steps=None, reset_moments=False,
              learning_rate=None, verbose=1, log_every=100, eval_every=0, print_outputs=False, save_every=0,
              diagnostics_every=0, restore_every=0, restore_step=None):
        terms, signature = self._costs(cost)
        graph = prepare_graph(cost)[2] if isinstance(cost, (IntegralCost, GraphCost)) else None
        if graph is not None and not self.session._capabilities & 2: raise ValueError('cost graphs require CPDE/3')
        epochs = _integer(epochs, 'epochs', 1)
        if state not in ('continue', 'reset'): raise ValueError("state must be 'continue' or 'reset'")
        if optimizer is None:
            optimizer = self._last_meta['optimizer'] if self._last_meta else 'rprop'
            optimizer = optimizer or 'rprop'
        if optimizer not in ('rprop', 'adam'): raise ValueError('optimizer must be rprop or adam')
        verbose = _integer(verbose, 'verbose')
        for key, value in [('log_every', log_every), ('eval_every', eval_every), ('save_every', save_every),
                           ('diagnostics_every', diagnostics_every)]: _integer(value, key)
        if diagnostics_every and not self.session._capabilities & 1:
            raise ValueError('gradient diagnostics require a CPDE/2 executable with diagnostics support')
        steps = 0.0 if reset_steps is None else prep.positive_float32(reset_steps, 'reset_steps')
        lr = 0.0 if learning_rate is None else prep.positive_float32(learning_rate, 'learning_rate')
        if not np.isfinite(steps) or (reset_steps is not None and steps <= 0): raise ValueError('reset_steps must be positive')
        if not np.isfinite(lr) or (learning_rate is not None and lr <= 0): raise ValueError('learning_rate must be positive')
        if reset_steps is not None and optimizer != 'rprop': raise ValueError('reset_steps requires RPROP')
        if reset_moments and optimizer != 'adam': raise ValueError('reset_moments requires Adam')
        restore_every = _integer(restore_every, 'restore_every')
        if restore_every:
            if optimizer != 'rprop': raise ValueError('periodic step restoration requires RPROP')
            if restore_step is None: raise ValueError('restore_every requires restore_step')
            with np.errstate(over='ignore', under='ignore'):
                restore_value = float(np.float32(restore_step))
            if not np.isfinite(restore_value) or restore_value < np.finfo(np.float32).tiny:
                raise ValueError('restore_step must be a positive normal float32 value')
            if not self.session._capabilities & 8:
                raise ValueError('periodic RPROP restoration requires a server with restore capability; rebuild the executable')
        elif restore_step is not None:
            raise ValueError('restore_step requires positive restore_every')
        if print_outputs and not eval_every: raise ValueError('print_outputs requires eval_every > 0')
        for term in terms:
            if term.inputs is not None:
                self._context(term.inputs, term._rows, term.chunk_size, track=True)
                if print_outputs:
                    prep.checked_product(len(term.inputs.derivatives), self.layers[-1], term.inputs.sample_count,
                                         name='printed evaluation output')
        w = Writer().put('Q', self._id).array([c._id for c in terms], 'u8').put('Q', epochs)
        w.put('i', int(optimizer == 'adam')).put('i', int(state == 'reset')).put('f', steps).put('i', int(reset_moments)).put('f', lr)
        for value in (verbose, log_every, eval_every, int(print_outputs), save_every): w.put('i', value)
        w.string(signature)
        if self.session._protocol != 'CPDE/1': w.put('i', diagnostics_every)
        if graph is not None: w.raw(graph)
        if restore_every: w.put('Q', restore_every).put('f', restore_value)
        if print_outputs and any(c.inputs is None for c in terms): raise ValueError('cannot print outputs for empty integrals')
        reports = []; gradient_reports = []; interrupted = False; by_id = {c._id: c for c in terms}
        self.last_gradient_report = None
        with self.session._lock:
            self.session._send(15 if graph is not None else 8, w.bytes())
            while True:
                try:
                    kind, r = self.session._receive()
                except KeyboardInterrupt:
                    self.session.stop(); interrupted = True; continue
                if kind == 1:
                    message = r.string()
                    if 'cost index' in message:
                        message += ' (costs: ' + ', '.join(f'{i}={c.label}' for i, c in enumerate(terms)) + ')'
                    raise RuntimeError(message)
                if kind in (2, 6):
                    epoch, update, after = r.get('Q'), r.get('Q'), r.get('i'); values = r.array('f8')
                    total = r.get('d') if kind == 6 else float(values.sum()); r.end()
                    report = dict(epoch=epoch, update=update, phase='after_update' if after else 'before_update',
                                  labels=[c.label for c in terms], total=total)
                    report['leaf_values' if kind == 6 else 'contributions'] = values.tolist()
                    reports.append(report)
                    if verbose or (epoch == epochs and after):
                        line = f"epoch {epoch}: total={report['total']:.9g} ({report['phase']})"
                        if verbose >= 2: line += ' ' + ' '.join(f'{c.label}={v:.9g}' for c, v in zip(terms, values))
                        print(line, flush=True)
                elif kind == 3:
                    self.checkpoint = self._state_dict_from_blob(r.data)
                elif kind == 4:
                    c = by_id[r.get('Q')]; a = r.array('f4'); r.end()
                    shaped = a.reshape(len(c.inputs.derivatives), c.inputs.sample_count, self.layers[-1]).transpose(0, 2, 1)
                    print(f'{c.label} outputs:\n{shaped}', flush=True)
                elif kind == 5:
                    epoch, update, count = r.get('Q'), r.get('Q'), r.get('i')
                    if count != len(self.layers)-1: raise RuntimeError('invalid gradient diagnostics layer count')
                    layers = []; squared_norm = 0.0; max_abs = 0.0; nonfinite_count = 0
                    for i in range(count):
                        norm2, maximum, bad, first = r.get('d'), r.get('d'), r.get('Q'), r.get('i')
                        squared_norm += norm2; max_abs = max(max_abs, maximum); nonfinite_count += bad
                        layers.append(dict(layer=i, l2_norm=float('nan') if bad else float(np.sqrt(norm2)),
                                           max_abs=float('nan') if bad else maximum, nonfinite_count=bad,
                                           first_nonfinite_index=None if first < 0 else first))
                    r.end()
                    report = dict(epoch=epoch, update=update, phase='before_update',
                                  l2_norm=float('nan') if nonfinite_count else float(np.sqrt(squared_norm)),
                                  max_abs=float('nan') if nonfinite_count else max_abs,
                                  nonfinite_count=nonfinite_count, layers=layers)
                    gradient_reports.append(report); self.last_gradient_report = report
                    if verbose:
                        print(f"epoch {epoch}: gradient l2={report['l2_norm']:.9g} "
                              f"max_abs={report['max_abs']:.9g} nonfinite={nonfinite_count} (before_update)", flush=True)
                elif kind == 0:
                    completed, stopped = r.get('Q'), bool(r.get('i'))
                    self._last_meta = decode_metadata(r.take(r.get('Q'))); r.end()
                    self._last_blob = None
                    break
                else: raise RuntimeError('unexpected training event')
        if verbose == 0 and completed != epochs:
            scalar_reports = [x for x in reports if 'total' in x]
            if scalar_reports: print(f"epoch {completed}: total={scalar_reports[-1]['total']:.9g}", flush=True)
        return dict(completed=completed, stopped=stopped or interrupted, reports=[x for x in reports if 'total' in x],
                    gradient_reports=gradient_reports, metadata=dict(self._last_meta))

    def resume(self, *, cost, **reporting):
        state = self.state_dict(); terms, signature = self._costs(cost)
        if signature != state['metadata']['stage']: raise ValueError('resume requires the same cost description and data versions')
        remaining = state['metadata']['remaining']
        if not remaining: raise ValueError('saved training call has no remaining epochs')
        return self.train(cost=cost, epochs=remaining, optimizer=state['metadata']['optimizer'], state='continue', **reporting)

    def _state_dict_from_blob(self, blob, include_history=True):
        meta, layers = decode_state(blob); meta['outputs'] = self.outputs
        result = dict(metadata=meta, weights=[], optimizer_state=[], gradient_history=[] if include_history else None)
        for layer in layers:
            b, active = layer['buffers'], layer['active']; inactive = 3-active
            result['weights'].append(b[0])
            result['optimizer_state'].append({'m': b[3], 'v': b[inactive]} if meta['optimizer'] == 'adam' else {'steps': b[3]})
            if include_history:
                history = {'current': b[active]}
                if meta['optimizer'] != 'adam': history['previous'] = b[inactive]
                result['gradient_history'].append(history)
        return result

    def state_dict(self, *, include_history=True):
        r = self.session._request(9, Writer().put('Q', self._id).bytes()); self._last_blob = r.data
        self._last_meta = decode_state(r.data)[0]
        return self._state_dict_from_blob(r.data, include_history)

    def load_state_dict(self, state):
        meta = state['metadata']; method = meta['optimizer']
        if meta['neurons'] != self.layers or meta['linear'] != self.linear: raise ValueError('state architecture mismatch')
        if method not in (None, 'rprop', 'adam'): raise ValueError('invalid optimizer')
        w = Writer().put('I', 1).array(self.layers, 'i4').array(self.linear, 'i4').put('i', {None: -1, 'rprop': 0, 'adam': 1}[method])
        for key in ('updates', 'adam_step', 'stage_completed', 'remaining'): w.put('Q', meta[key])
        w.put('f', prep.positive_float32(meta['learning_rate'], 'learning_rate'))
        w.put('f', prep.positive_float32(meta['initial_step'], 'initial_step')).string(meta['stage'])
        history = state.get('gradient_history')
        for i, (h, width) in enumerate(zip(self.layers[1:], self.layers[:-1])):
            size = h*(width+1); zero = np.zeros(size, np.float32)
            current = history[i]['current'] if history is not None else zero
            opt = state['optimizer_state'][i]
            nonnegative = prep.finite(opt['v'] if method == 'adam' else opt['steps'], 'optimizer state')
            if np.any(nonnegative < 0):
                raise ValueError('Adam variances and RPROP steps must be nonnegative')
            inactive = opt['v'] if method == 'adam' else history[i]['previous'] if history is not None else zero
            fourth = opt['m'] if method == 'adam' else opt['steps']
            w.put('i', 1)
            for a in (state['weights'][i], current, inactive, fourth):
                a = prep.finite(a, 'state buffer')
                if a.size != size: raise ValueError('state buffer shape mismatch')
                w.array(a, 'f4')
        blob = w.bytes()
        self.session._request(10, Writer().put('Q', self._id).raw(blob).bytes()).end()
        self._last_blob = blob; self._last_meta = dict(meta)
        if meta.get('outputs'): self.name_outputs(meta['outputs'])

    def save(self, path, *, include_history=True):
        state = self.state_dict(include_history=include_history)
        meta = dict(state['metadata'], history_available=include_history)
        arrays = {'metadata': np.asarray(json.dumps(meta))}
        for i, a in enumerate(state['weights']): arrays[f'weights_{i}'] = a
        for i, values in enumerate(state['optimizer_state']):
            for key, a in values.items(): arrays[f'optimizer_{i}_{key}'] = a
        if include_history:
            for i, values in enumerate(state['gradient_history']):
                for key, a in values.items(): arrays[f'history_{i}_{key}'] = a
        with open(path, 'wb') as f: np.savez(f, **arrays)

    def load(self, path):
        with np.load(path, allow_pickle=False) as f:
            meta = json.loads(str(f['metadata'])); method = meta['optimizer']; count = len(self.layers)-1
            state = dict(metadata=meta, weights=[f[f'weights_{i}'] for i in range(count)],
                         optimizer_state=[{k: f[f'optimizer_{i}_{k}'] for k in (('m', 'v') if method == 'adam' else ('steps',))} for i in range(count)],
                         gradient_history=[{k: f[f'history_{i}_{k}'] for k in (('current',) if method == 'adam' else ('current', 'previous'))} for i in range(count)] if meta['history_available'] else None)
            self.load_state_dict(state)

    def close(self):
        if not self._closed:
            self.session._request(11, Writer().put('i', 1).put('Q', self._id).bytes()).end(); self._closed = True
