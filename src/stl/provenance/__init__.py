"""One-shot tools that carried historical leap runs across reviewed source changes.

`leap_resume.py` and `leap_recover.py` keep the bytes they had in
`src/stl/solver/`, so their allow-lists still name that directory, and
`leap_recover.py` still imports `stl.solver.leap_resume`. This package
registers the moved `leap_resume` module under that old name. The builder
hash globs `src/stl/solver/leap_*.py` alone and does not cover these files.
"""
import sys

from stl.provenance import leap_resume

sys.modules.setdefault('stl.solver.leap_resume', leap_resume)
