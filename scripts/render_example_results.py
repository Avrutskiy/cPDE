"""Restyle the guide figures from recorded validation arrays, without training.

Run with NumPy and Matplotlib installed. The source runs live under output/examples;
they are validation artifacts and are not included in the samples archive.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from validate_examples import plot_result

ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    'fit_function_vanilla': ('rtx3090-delivery-gpu-rerun-20261102', 'fit_function_vanilla'),
    'fit_function': ('rtx3090-delivery-gpu-rerun-20261102', 'fit_function'),
    'fit_function_explicit': ('rtx3090-explicit-guide-20261103', 'fit_function_explicit'),
    'poisson_disk_vanilla': ('rtx3090-delivery-gpu-rerun-20261102', 'poisson_disk_vanilla'),
    'poisson_disk': ('rtx3090-delivery-gpu-rerun-20261102', 'poisson_disk'),
    'schrodinger_circle_vanilla': ('rtx3090-circle-dense-20261103', 'schrodinger_circle_projection_vanilla'),
    'schrodinger_circle': ('rtx3090-circle-dense-20261103', 'schrodinger_circle_projection'),
    'attached_variation': ('rtx3090-delivery-gpu-rerun-20261102', 'attached_variation'),
}


def main():
    destination = ROOT/'examples/results'
    destination.mkdir(exist_ok=True)
    provenance = {}
    for name, (folder, source_name) in RUNS.items():
        source = ROOT/'output/examples'/folder
        report_path = source/f'{source_name}.json'
        array_path = source/f'{source_name}.npz'
        report = json.loads(report_path.read_text())
        if report['status'] != 'completed':
            raise ValueError(f'Cannot present an incomplete run as a result: {report_path}')
        if name != 'attached_variation':
            with np.load(array_path, allow_pickle=False) as arrays:
                plot_result(source_name, arrays, report, destination)
        if name != source_name:
            for extension in ('png', 'pdf'):
                (destination/f'{source_name}.{extension}').replace(destination/f'{name}.{extension}')
        provenance[name] = {
            'source_run': str(source.relative_to(ROOT)),
            'source_name': source_name,
            'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (report_path, array_path)},
            'backend': report['backend'], 'status': report['status'],
            'updates': report.get('updates', 0), 'metrics': report['metrics'],
            'train_calls': report.get('train_calls', []),
            'figure_sha256': {f'{name}.{ext}': hashlib.sha256((destination/f'{name}.{ext}').read_bytes()).hexdigest()
                              for ext in ('png', 'pdf') if name != 'attached_variation'},
        }
        if 'dense_residual_evaluation' in report:
            provenance[name]['dense_residual_evaluation'] = {
                key: value for key, value in report['dense_residual_evaluation'].items()
                if key != 'server'
            }
        print(f'Prepared {name} from {folder}/{source_name}', flush=True)
    (destination/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')


if __name__ == '__main__':
    main()
