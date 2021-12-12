"""
usage: python runtests.py <projname> 

this script exists for easy editor integration
"""

import sys
import willutil

_overrides = {
   #   "PYTHONPATH=. python rpxdock/app/genrate_motif_scores.py TEST"
}

_file_mappings = {
   # 'sym.py': ['willutil/tests/test_homog.py'],
}

if __name__ == '__main__':
   args = willutil.runtests.get_args(sys.argv)
   willutil.runtests.main(_file_mappings, _overrides, **args)
