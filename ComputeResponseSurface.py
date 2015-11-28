# ComputeResponseSurface.py
#
# Generate response surface for AEP using different 
#
# Author: Lewis Li (lewisli@stanford.edu)
# Original Date: November 26th 2015


from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from AEPEvaluator import EvaluateAEP
from LCOETurbineEvaluator import EvaluateLCOE

BladeLength = 50
HubHeight = 90
MaxRotSpeed = 12
EvaluateLCOE(BladeLength,HubHeight,MaxRotSpeed,True)

# # Set range of variables to evaluate
# BladeDiameters = np.arange(50, 100, 5)
# HubHeights  = np.arange(60,100,5)
# OmegaMax = np.arange(5,20,0.5)

# X,Y =  np.meshgrid(BladeDiameters, HubHeights)
# Z = np.zeros(X.shape)

# # Benchmark with Siemens's SWT-2.3-101
# #EvaluateAEP(100,80,14)

# for b in range(0,len(BladeDiameters)):
# 	BladeDiameter = BladeDiameters[b]
# 	for h in range(0,len(HubHeights)):
# 		HubHeight = HubHeights[h]
# 		Z[h,b] = EvaluateAEP(BladeDiameter,HubHeight,22)


# # Plot Response Surface
# fig = plt.figure()
# fig.suptitle('Response Surface of AEP Given Blade Diameter and Hub Height',\
# 	fontsize=16)
# ax = fig.gca(projection='3d')
# surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=True,shade=True)
# fig.colorbar(surf, shrink=0.5, aspect=5,label='AEP (MWH)')
# fig.patch.set_facecolor('white') 

# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# ax.set_xlabel('Blade Diameter (m)')
# ax.set_ylabel('Hub Height (m)')
# ax.set_zlabel('AEP (MWH)')

# #fig.savefig('ResponseSurface1.png', facecolor=fig.get_facecolor(), edgecolor='none') 

# plt.show()

