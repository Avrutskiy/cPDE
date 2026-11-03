"""CI checks at full size: short training runs and a moving-boundary gradient demo."""
import os
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cpde import Network


if __name__ == '__main__':
    executable = str(Path(sys.argv[1]).resolve())
    train = Network.train

    def one_epoch(net, **options):
        # The example has no test mode; only this CI harness shortens its run.
        options['epochs'] = 1
        return train(net, **options)

    with patch.dict(os.environ, {'CPDE_SERVER': executable}), patch.object(Network, 'train', one_epoch):
        for name in ('fit_function_explicit', 'fit_function_vanilla', 'fit_function', 'poisson_disk_vanilla', 'poisson_disk', 'schrodinger_circle', 'schrodinger_circle_vanilla', 'attached_variation'):
            print('Checking', name, flush=True)
            runpy.run_module('examples.'+name, run_name='__main__')
