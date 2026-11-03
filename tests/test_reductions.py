import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from build_host_probe import build


class ReductionTests(unittest.TestCase):
    def test_reduction_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            executable = os.environ.get('CPDE_TEST_REDUCTION_EXECUTABLE')
            if executable is None:
                executable = build(temp, Path(__file__).with_name('reduction_probe.cu'))
            result = subprocess.run([str(executable)], check=True, capture_output=True, text=True)
            self.assertIn('padding passed', result.stdout)
