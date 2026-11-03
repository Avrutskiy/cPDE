"""Independent high-order and chunk checks, without replaying known audit defects."""
import json
import sys
from audit_cuda import Session, high_order, chunk_stress
with Session(sys.argv[1]) as session:
    result = dict(high_order=high_order(session), chunk_stress=chunk_stress(session))
print(json.dumps(result, indent=2, allow_nan=False))
if not all(x['passed'] for x in result['high_order']) or not result['chunk_stress']['passed']:
    raise SystemExit(1)
