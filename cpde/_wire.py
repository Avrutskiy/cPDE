import struct
import numpy as np

class Writer:
    def __init__(self): self.parts = []
    def put(self, fmt, value): self.parts.append(struct.pack('<' + fmt, value)); return self
    def array(self, value, dtype):
        a = np.ascontiguousarray(value, dtype=np.dtype(dtype).newbyteorder('<'))
        self.put('Q', a.size); self.parts.append(a.tobytes()); return self
    def string(self, value):
        b = value.encode('utf-8'); self.put('Q', len(b)); self.parts.append(b); return self
    def raw(self, b): self.parts.append(b); return self
    def bytes(self): return b''.join(self.parts)

class Reader:
    def __init__(self, data): self.data, self.pos = data, 0
    def take(self, n):
        if self.pos + n > len(self.data): raise RuntimeError('truncated executable response')
        value = self.data[self.pos:self.pos+n]; self.pos += n; return value
    def get(self, fmt): return struct.unpack('<' + fmt, self.take(struct.calcsize('<' + fmt)))[0]
    def array(self, dtype):
        dt = np.dtype(dtype).newbyteorder('<'); n = self.get('Q')
        return np.frombuffer(self.take(n * dt.itemsize), dtype=dt).copy()
    def string(self): return self.take(self.get('Q')).decode('utf-8')
    def end(self):
        if self.pos != len(self.data): raise RuntimeError('unexpected executable response fields')

def read_metadata(r):
    if r.get('I') != 1: raise ValueError('unsupported state version')
    ns, ls = r.array('i4'), r.array('i4')
    meta = dict(neurons=ns.tolist(), linear=ls.tolist(), optimizer={-1: None, 0: 'rprop', 1: 'adam'}[r.get('i')],
                updates=r.get('Q'), adam_step=r.get('Q'), stage_completed=r.get('Q'), remaining=r.get('Q'),
                learning_rate=r.get('f'), initial_step=r.get('f'), stage=r.string())
    return meta

def decode_metadata(blob):
    r = Reader(blob); meta = read_metadata(r); r.end(); return meta

def decode_state(blob):
    r = Reader(blob); meta = read_metadata(r)
    ns = meta['neurons']
    layers = []
    for h, w in zip(ns[1:], ns[:-1]):
        active = r.get('i'); buffers = [r.array('f4') for _ in range(4)]
        if active not in (1, 2) or any(len(a) != h*(w+1) for a in buffers): raise ValueError('invalid state buffers')
        layers.append(dict(active=active, buffers=buffers))
    r.end()
    return meta, layers
