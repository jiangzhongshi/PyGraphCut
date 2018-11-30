import numpy as np
import pygraphcut
import sys,os
sys.path.append(os.path.expanduser('~/Workspace/libigl/python'))
import pyigl as igl
from iglhelpers import e2p, p2e

with np.load('/home/zhongshi/data/sig17_seg_benchmark/full_dump0/shrec/10.off.npz') as npl:
    print(dict(npl).keys())
    V,F = npl['V'], npl['F']
new_label = np.asarray(pygraphcut.sdf_values(V,F))
import pdb;pdb.set_trace()
vw = igl.glfw.Viewer()
vw.data().set_mesh(p2e(V), p2e(F))
vw.data().set_colors(p2e(new_label))
vw.launch()
print(new_label)
