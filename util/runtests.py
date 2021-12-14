"""
usage: python runtests.py <projname> 

this script exists for easy editor integration
"""

import sys
import willutil

overrides = {
   #   "PYTHONPATH=. python rpxdock/app/genrate_motif_scores.py TEST"
}

file_mappings = {
   'derp_learning/aimnet/aimnet/models.py': ['derp_learning/tests/test_aimnet.py'],
   'derp_learning/aimnet/aimnet/modules.py': ['derp_learning/tests/test_aimnet.py'],
   'derp_learning/aimnet/aimnet/loaders.py': ['derp_learning/tests/test_aimnet.py'],
   'derp_learning/aimnet/aimnet/calculator.py': ['derp_learning/tests/test_aimnet.py'],
}

if __name__ == '__main__':
   args = willutil.runtests.get_args(sys.argv)
   willutil.runtests.main(
      file_mappings=file_mappings,
      overrides=overrides,
      **args,
   )
