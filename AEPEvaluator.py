# AEPEvaluator.py
#
# Function used to evaluate AEP for 3 variables (blade diameter, hub height,
# and maximum rotational speed). The other major parameters are the same
#
# Author: Lewis Li (lewisli@stanford.edu)
# Original Date: November 26th 2015


import os
from math import pi
import numpy as np
import math
import matplotlib.pyplot as plt

from commonse.utilities import check_gradient_unit_test
from rotorse.rotoraero import Coefficients, SetupRunVarSpeed, \
		RegulatedPowerCurve, AEP
from rotorse.rotoraerodefaults import RotorAeroVSVPWithCCBlade,GeometrySpline, \
		CCBladeGeometry, CCBlade, CSMDrivetrain, WeibullCDF, \
		WeibullWithMeanCDF, RayleighCDF
from rotorse.rotor import RotorSE
from rotorse.precomp import Profile, Orthotropic2DMaterial, CompositeSection

from drivese.hub import HubSE
from drivese.drive import Drive4pt

from WindDistribution import CalculateAEPConstantWind
from WindDistribution import CalculateAEPWeibull

# Given mean, and shape of Weibull distriution, estimate the scale parameter
def ComputeScaleFunction(Mean, Shape):
	return Mean/math.gamma(1 + 1.0/Shape)

def GenerateWeibull(Shape,Scale,Samples):
	return Scale*np.random.weibull(Shape, Samples)


################################################################################
### 1. Aerodynamic and structural performance using RotorSE
def EvaluateAEP(Diameter, HubHeight, RPM_Max):

	# Basic Rotor Model
	cdf_type = 'weibull'

	rotor = RotorAeroVSVPWithCCBlade(cdf_type)
	rotor.configure()

	# Define blade and chord length
	rotor.B = 3 # Number of blades (Do not change)
	rotor.r_max_chord = 0.23577  # (Float): location of second control point (generally also max chord)
	rotor.chord_sub = [3.2612, 4.5709, 3.3178, 1.4621]  # (Array, m): chord at control points
	rotor.theta_sub = [13.2783, 7.46036, 2.89317, -0.0878099]  # (Array, deg): twist at control points
	rotor.Rhub = 1.5  # (Float, m): hub radius
	rotor.precone = 2.5  # (Float, deg): precone angle
	rotor.tilt = -5.0  # (Float, deg): shaft tilt
	rotor.yaw = 0.0  # (Float, deg): yaw error

	# Hub height 
	rotor.hubHt = HubHeight  # (Float, m)

	# Blade length (if not precurved or swept) otherwise length of blade before curvature
	rotor.bladeLength = Diameter/2  # (Float, m): 

	# Radius to tip
	rotor.Rtip = rotor.bladeLength + rotor.Rhub  # (Float, m): tip radius (blade radius)

	# Rotor blade aerodynamic profiles... leave the same for now...
	basepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), \
		'5MW_AFFiles')

	# load all airfoils
	airfoil_types = [0]*8
	airfoil_types[0] = basepath + os.path.sep + 'Cylinder1.dat'
	airfoil_types[1] = basepath + os.path.sep + 'Cylinder2.dat'
	airfoil_types[2] = basepath + os.path.sep + 'DU40_A17.dat'
	airfoil_types[3] = basepath + os.path.sep + 'DU35_A17.dat'
	airfoil_types[4] = basepath + os.path.sep + 'DU30_A17.dat'
	airfoil_types[5] = basepath + os.path.sep + 'DU25_A17.dat'
	airfoil_types[6] = basepath + os.path.sep + 'DU21_A17.dat'
	airfoil_types[7] = basepath + os.path.sep + 'NACA64_A17.dat'

	# place at appropriate radial stations
	af_idx = [0, 0, 1, 2, 3, 3, 4, 5, 5, 6, 6, 7, 7, 7, 7, 7, 7]

	n = len(af_idx)
	af = [0]*n
	for i in range(n):
		af[i] = airfoil_types[af_idx[i]]

	rotor.airfoil_files = af  # (List): paths to AeroDyn-style airfoil files

	# (Array, m): locations where airfoils are defined on unit radius
	rotor.r_af = np.array([0.02222276, 0.06666667, 0.11111057, 0.16666667, \
		0.23333333, 0.3, 0.36666667, 0.43333333, 0.5, 0.56666667, 0.63333333, 0.7, \
		0.76666667, 0.83333333, 0.88888943,0.93333333, 0.97777724])    
	rotor.idx_cylinder = 3  # (Int): index in r_af where cylinder section ends

	# Wind Parameters are specified here !!!!!!!!!!!!!
	rotor.rho = 1.225  # (Float, kg/m**3): density of air
	rotor.mu = 1.81206e-5  # (Float, kg/m/s): dynamic viscosity of air

	# Shear Exponent 
	rotor.shearExp = 0.143 # 0.2  # (Float): shear exponent!!!!!!!!!!!!!!!!!
	rotor.cdf_mean_wind_speed = 6.0  # (Float, m/s): mean wind speed of site cumulative distribution function
	rotor.weibull_shape_factor = 2.0  # (Float): shape factor of weibull distribution

	# Rotor fixed design parameters !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	rotor.control.Vin = 3.0  # (Float, m/s): cut-in wind speed in


	rotor.control.Vout = 30.0 #25.0  # (Float, m/s): cut-out wind speed !!!!!!!!!!!!!!


	rotor.control.ratedPower = 1.5e6  # (Float, W): rated power !!!!!!!!!!!!!!
	rotor.control.pitch = 0.0  # (Float, deg): pitch angle in region 2 (and region 3 for fixed pitch machines)
	rotor.control.minOmega = 0.0  # (Float, rpm): minimum allowed rotor rotation speed
	rotor.control.maxOmega = RPM_Max #12.0  # (Float, rpm): maximum allowed rotor rotation speed!!!!!!!!!!!!!!!!

	rotor.control.tsr = 7  # **dv** (Float): tip-speed ratio in Region 2 (should be  !!!!!!!!!!!!!!

	rotor.nSector = 4  # (Int): number of sectors to divide rotor face into in computing thrust and power

	# Calculation 
	rotor.npts_coarse_power_curve = 20  # (Int): number of points to evaluate aero analysis at
	rotor.npts_spline_power_curve = 200  # (Int): number of points to use in fitting spline to power curve
	rotor.AEP_loss_factor = 1.0  # (Float): availability and other losses (soiling, array, etc.)

	# Energy loss estimates
	rotor.tiploss = False  # (Bool): include Prandtl tip loss model
	rotor.hubloss = True  # (Bool): include Prandtl hub loss model
	rotor.wakerotation = True  # (Bool): include effect of wake rotation (i.e., tangential induction factor is nonzero)
	rotor.usecd = True  # (Bool): use drag coefficient in computing induction factors

	# No effect on AEP
	rotor.VfactorPC = 0.7  # (Float): fraction of rated speed at which the deflection is assumed to representative throughout the power curve calculation

	# Run to calculate rotor parameters
	rotor.run()

	# Get Power Curve
	PowerCurve = rotor.P/1e6
	PowerCurveVelocity = rotor.V

	# RPM Curve
	#AEP = CalculateAEPConstantWind(PowerCurve, PowerCurveVelocity, 7.5)

	WindReferenceHeight = 30
	WindReferenceMeanVelocity = 6
	ShearFactor = 0.2

	AEP = CalculateAEPWeibull(PowerCurve,PowerCurveVelocity, HubHeight, \
	 	rotor.weibull_shape_factor, WindReferenceHeight, \
	 	WindReferenceMeanVelocity, ShearFactor)


	print "Estimated with diameter %d AEP is %f MHW " %(Diameter,AEP)

	return AEP

