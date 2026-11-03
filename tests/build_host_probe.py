"""Build a test-only serial execution probe; this is NOT a CUDA implementation.

Launch syntax is removed in a temporary copy. Kernels run with one simulated
thread and use scalar column-major BLAS. No GPU/stream properties are tested.
"""
from pathlib import Path
import re
import subprocess
import sys

def build(destination, source=None, *, count_forwards=False):
    root = Path(__file__).resolve().parents[1]
    destination = Path(destination).resolve(); destination.mkdir(parents=True, exist_ok=True)
    for cuda_source in (root/'cuda').iterdir():
        text = cuda_source.read_text()
        text = re.sub(r'<<<.*?>>>', '', text, flags=re.S)
        if count_forwards and cuda_source.name == 'cpde_engine.cuh':
            marker = 'layer* feedforward() {'
            if text.count(marker) != 1: raise RuntimeError('Cannot instrument host forward pass')
            text = text.replace(marker, marker + '\n fprintf(stderr,"CPDE_TEST_FORWARD %d %d\\n",layers->env->S->h,layers->env->w);')
        (destination/cuda_source.name).write_text(text)
    compile_source = destination/'cpde_server.cu'
    if source is not None:
        source = Path(source)
        compile_source = destination/source.name
        compile_source.write_text(re.sub(r'<<<.*?>>>', '', source.read_text(), flags=re.S))
    exe = destination/('cpde_host_probe' if source is None else source.stem)
    result = subprocess.run(['g++', '-std=c++17', '-DCPDE_HOST_PROBE', '-O1', '-g', '-pthread', '-I'+str(root/'tests/host_shim'),
                             '-x', 'c++', str(compile_source), '-o', str(exe)], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError('Host probe compilation failed:\n' + result.stderr)
    return exe

if __name__ == '__main__': print(build(sys.argv[1]))
