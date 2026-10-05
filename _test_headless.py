import sys, os
sys.argv = ['demo_interactive.py', '--demo-mode', '--cpu']
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as _plt
_plt.show = lambda: None

import runpy
runpy.run_path(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'demo_interactive.py'),
    run_name='__main__'
)
print('[ALL PASS] Demo ran successfully.')
