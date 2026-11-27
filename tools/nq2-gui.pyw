# nq2-gui.pyw ... starts the NanoQuatTwo console without a console window
# (pythonw runs .pyw files windowless). Same program as nq2-gui.py.
import os
import runpy

runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "nq2-gui.py"), run_name="__main__")
