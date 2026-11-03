"""Fourier derivative-training benchmark: CUDA cPDE / PyTorch / PyTorch + JIT.

See docs/benchmarks/fourier-rtx3090.md. The cPDE mode requires a separate
server build with docs/benchmarks/fourier-timing.patch applied.
Run cPDE first to save identical inputs/weights for both PyTorch modes.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from cpde import Data, Network, Session, table
from cpde._wire import Writer

ARCH = [2, 128, 128, 128, 128, 1]


def dataset(order):
    schema = table(['x'*i+'y'*(order-i) for i in range(order+1)], variables='xy')
    x, y = np.meshgrid(np.linspace(-1, 1, 27), np.linspace(-1, 1, 27), indexing='ij')
    points = np.column_stack((x.ravel(), y.ravel())).astype(np.float32)
    x, y = points.astype(float).T
    coef = np.random.default_rng(2026).uniform(-1, 1, (4, 10, 10))
    target = np.zeros((len(schema), len(points)))
    for row, (_, dx, dy) in enumerate(schema):
        for n in range(1, 11):
            for k in range(1, 11):
                a, b, c, d = coef[:, n-1, k-1]*n**int(dx)*k**int(dy)/(n*k)
                sx, cx = np.sin(n*x+dx*np.pi/2), np.cos(n*x+dx*np.pi/2)
                sy, cy = np.sin(k*y+dy*np.pi/2), np.cos(k*y+dy*np.pi/2)
                target[row] += a*sx*sy+b*sx*cy+c*cx*sy+d*cx*cy
    weights = (1/target.std(axis=1)**2).astype(np.float32)
    return schema, points, target.astype(np.float32), weights


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def cpde_run(order, dest):
    schema, points, target, weights = dataset(order)
    arrays = np.zeros((len(schema), 2, len(points)), np.float32)
    for row, (degree, dx, dy) in enumerate(schema):
        if degree == 0: arrays[row] = points.T
        elif degree == 1: arrays[row] = np.array([dx, dy])[:, None]
    begin = time.perf_counter()
    with Session(SERVER, backend='cuda') as session:
        net = Network(ARCH, session=session, seed=3407)
        initial = net.state_dict()
        inputs = Data(arrays, schema, session=session)
        targets = Data(target[:, None], schema, session=session)
        cost = net.cost(inputs=inputs, targets=targets, weights=weights[None])
        setup = time.perf_counter()-begin
        command = Writer().put('Q', net._id).put('Q', cost._id).put('i', 20).put('i', 100).put('i', 7)
        r = session._request(90, command.bytes())
        wall, gpu, loss = r.array('f8'), r.array('f8'), r.get('d')
        grads = [r.array('f4') for _ in initial['weights']]; r.end()
        prediction = net.eval(inputs)
        values = prediction.to_numpy()[:, 0]; prediction.close()
        np.savez_compressed(dest/f'cpde-{order}-arrays.npz', schema=schema, points=points,
                            target=target, row_weights=weights, prediction=values,
                            **{f'w{i}': w for i, w in enumerate(initial['weights'])},
                            **{f'g{i}': g for i, g in enumerate(grads)})
        result = dict(backend='cpde', order=order, rows=len(schema), setup_seconds=setup,
                      wall_ms=wall.tolist(), gpu_ms=gpu.tolist(), loss=loss,
                      median_ms=float(np.median(wall)), iterations=100, batches=7,
                      status='ok', runtime_compilation_seconds=0)
    save(dest/f'cpde-{order}.json', result)
    print(json.dumps(result), flush=True)


def torch_run(order, mode, dest):
    # Each compiled order has a fresh directory and a fresh process.
    if mode == 'compiled':
        cache = dest/f'cold-cache-{order}-{time.time_ns()}'
        os.environ['TORCHINDUCTOR_CACHE_DIR'] = str(cache/'inductor')
        os.environ['TRITON_CACHE_DIR'] = str(cache/'triton')
        os.environ['TORCHINDUCTOR_FX_GRAPH_CACHE'] = '0'
        os.environ['TORCHINDUCTOR_AUTOGRAD_CACHE'] = '0'
    import torch
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    data = np.load(dest/f'cpde-{order}-arrays.npz')
    schema = data['schema']
    x = torch.tensor(data['points'], device='cuda', requires_grad=True)
    target = torch.tensor(data['target'], device='cuda')
    row_weights = torch.tensor(data['row_weights'], device='cuda')
    params = []
    for i, (a, b) in enumerate(zip(ARCH, ARCH[1:])):
        matrix = data[f'w{i}'].reshape((b, a+1), order='F')
        params.extend([torch.tensor(matrix[:, :-1].copy(), device='cuda', requires_grad=True),
                       torch.tensor(matrix[:, -1].copy(), device='cuda', requires_grad=True)])
    params = tuple(params)
    counts = [tuple(map(int, r[1:])) for r in schema]

    def step(x, target, row_weights, *params):
        y = x
        for i in range(0, len(params), 2):
            y = torch.nn.functional.linear(y, params[i], params[i+1])
            if i+2 < len(params): y = torch.sigmoid(y)
        rows = {(0, 0): y[:, 0]}
        # Reuse each lower-order derivative graph. Batch entries are independent.
        derivative_cache = {}
        for degree in range(1, order+1):
            for dx in range(degree, -1, -1):
                dy = degree-dx
                parent = (dx-1, dy) if dx else (0, dy-1)
                direction = 0 if dx else 1
                if parent not in derivative_cache:
                    derivative_cache[parent] = torch.autograd.grad(rows[parent], x,
                        grad_outputs=torch.ones_like(rows[parent]), create_graph=True)[0]
                rows[(dx, dy)] = derivative_cache[parent][:, direction]
        prediction = torch.stack([rows[c] for c in counts])
        residual = prediction-target
        loss = (row_weights[:, None]*residual*residual).sum()
        grads = torch.autograd.grad(loss, params)
        return (loss.detach(), prediction.detach(), *(g.detach() for g in grads))

    torch.cuda.synchronize()
    first = time.perf_counter()
    reference = step(x, target, row_weights, *params)
    torch.cuda.synchronize()
    eager_first_seconds = time.perf_counter()-first
    prediction = reference[1].cpu().numpy()
    grads = [np.concatenate((reference[2+2*i].cpu().numpy().T.ravel(), reference[3+2*i].cpu().numpy()))
             for i in range(len(ARCH)-1)]
    ref = np.concatenate([data[f'g{i}'] for i in range(len(ARCH)-1)]).astype(float)
    got = np.concatenate(grads).astype(float)
    normalized_output_error = (prediction-data['prediction'])*np.sqrt(data['row_weights'])[:, None]
    agreement = dict(loss_relative_error=abs(reference[0].item()/json.loads((dest/f'cpde-{order}.json').read_text())['loss']-1),
                     gradient_relative_l2=float(np.linalg.norm(got-ref)/np.linalg.norm(ref)),
                     normalized_output_max_abs=float(np.max(abs(normalized_output_error))),
                     output_relative_l2_per_row=(np.linalg.norm(prediction-data['prediction'], axis=1)/np.maximum(np.linalg.norm(data['prediction'], axis=1), 1e-15)).tolist())
    result = dict(backend=mode, order=order, rows=len(schema), torch=torch.__version__,
                  cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
                  python=sys.version, agreement=agreement, eager_first_seconds=eager_first_seconds,
                  status='validated')
    save(dest/f'{mode}-{order}.json', result)
    if agreement['gradient_relative_l2'] > 5e-4 or agreement['loss_relative_error'] > 5e-5:
        raise RuntimeError('cPDE/PyTorch numerical agreement failed: '+str(agreement))
    fn = step
    if mode == 'compiled':
        # Capture input derivatives AND parameter gradients before Inductor.
        # Compiling only the MLP would leave most derivative work in Python.
        from torch.fx.experimental.proxy_tensor import make_fx
        started = time.perf_counter()
        graph = make_fx(step, tracing_mode='fake')(x, target, row_weights, *params)
        torch.cuda.synchronize()
        result['trace_seconds'] = time.perf_counter()-started
        result['graph_nodes'] = len(list(graph.graph.nodes))
        result['status'] = 'compiling'
        save(dest/f'{mode}-{order}.json', result)
        print('GRAPH', order, result['graph_nodes'], flush=True)
        fn = torch.compile(graph, backend='inductor', fullgraph=True)
        compiled_result = fn(x, target, row_weights, *params)
        torch.cuda.synchronize()
        result['cold_compile_and_first_call_seconds'] = time.perf_counter()-started
        errors = [float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(b).clamp_min(1e-15))
                  for a, b in zip(compiled_result, reference)]
        result['compiled_relative_errors'] = errors
        if max(errors) > 5e-4: raise RuntimeError('Compiled result disagrees with eager: '+str(errors))
        import triton
        result['triton'] = triton.__version__
    for _ in range(3): fn(x, target, row_weights, *params)
    torch.cuda.synchronize()
    probe_start = time.perf_counter()
    fn(x, target, row_weights, *params)
    torch.cuda.synchronize()
    estimate = time.perf_counter()-probe_start
    iterations = max(1, min(100, int(.25/max(estimate, 1e-6))))
    wall = []; gpu = []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(7):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        t = time.perf_counter(); start.record()
        for _ in range(iterations): output = fn(x, target, row_weights, *params)
        end.record(); end.synchronize()
        wall.append((time.perf_counter()-t)*1000/iterations)
        gpu.append(start.elapsed_time(end)/iterations)
    result.update(status='ok', iterations=iterations, batches=7, wall_ms=wall, gpu_ms=gpu,
                  median_ms=float(np.median(wall)), peak_allocated_bytes=torch.cuda.max_memory_allocated())
    save(dest/f'{mode}-{order}.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['cpde', 'eager', 'compiled'])
    parser.add_argument('order', type=int, choices=range(7))
    parser.add_argument('--server', default=os.environ.get('CPDE_SERVER'), help='Timing-patched CUDA server executable (cPDE mode)')
    parser.add_argument('--output', type=Path, default=ROOT/'output/fourier-performance')
    args = parser.parse_args()
    SERVER = args.server
    if args.mode == 'cpde' and not SERVER:
        parser.error('cPDE mode requires --server or CPDE_SERVER for the timing-patched executable')
    dest = args.output.resolve(); dest.mkdir(parents=True, exist_ok=True)
    try:
        (cpde_run(args.order, dest) if args.mode == 'cpde' else torch_run(args.order, args.mode, dest))
    except Exception as error:
        path = dest/f'{args.mode}-{args.order}.json'
        result = json.loads(path.read_text()) if path.exists() else dict(backend=args.mode, order=args.order)
        result.update(status='error', error=repr(error))
        save(path, result)
        raise
