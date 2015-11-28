# LCOETurbineEvaluator.py
#
# Given turbine blade size, hub height and maximum rotational speed, compute the 
# Levelized Cost of Energy (in $/KWH). The annual energy is computed by 
# mutiplying the Power Curve by the Weibull Distribution.
#
#
# Author: Lewis Li (lewisli@stanford.edu)
# Original Date: November 28th 2015
#
#

import os
from math import pi
import numpy as np

from pyopt_driver.pyopt_driver import pyOptDriver

from commonse.WindWaveDrag import FluidLoads, AeroHydroLoads, TowerWindDrag, \
	TowerWaveDrag
from commonse.environment import WindBase, WaveBase  # , SoilBase
from commonse import Tube
from commonse.utilities import check_gradient_unit_test
from commonse.UtilizationSupplement import fatigue, hoopStressEurocode, \
	shellBucklingEurocode, bucklingGL, vonMisesStressUtilization

from rotorse.rotor import RotorSE 
from rotorse.rotoraero import Coefficients, SetupRunVarSpeed, \
		RegulatedPowerCurve, AEP
from rotorse.rotoraerodefaults import RotorAeroVSVPWithCCBlade,GeometrySpline, \
		CCBladeGeometry, CCBlade, CSMDrivetrain, WeibullCDF, \
		WeibullWithMeanCDF, RayleighCDF
from rotorse.precomp import Profile, Orthotropic2DMaterial, CompositeSection
from drivese.hub import HubSE
from drivese.drive import Drive4pt
from towerse.tower import TowerSE

from turbine_costsse.turbine_costsse.turbine_costsse import Turbine_CostsSE
from plant_costsse.nrel_csm_bos.nrel_csm_bos import bos_csm_assembly
from plant_costsse.nrel_csm_opex.nrel_csm_opex import opex_csm_assembly

from WindDistribution import CalculateAEPConstantWind
from WindDistribution import CalculateAEPWeibull
from WindDistribution import ComputeScaleFunction
from WindDistribution import EstimateCapacity
from WindDistribution import ComputeLCOE

from openmdao.main.api import VariableTree, Component, Assembly, set_as_top
from openmdao.main.datatypes.api import Int, Float, Array, VarTree, Bool, Slot
from openmdao.lib.casehandlers.api import DumpCaseRecorder

from fusedwind.turbine.tower import TowerFromCSProps
from fusedwind.interface import implement_base

import frame3dd
import matplotlib.pyplot as plt


def EvaluateLCOE(BladeLength, HubHeight, MaximumRotSpeed,Verbose=False):

	############################################################################
	# Define baseline paremeters used for scaling
	ReferenceBladeLength = 35;
	ReferenceTowerHeight = 95
	WindReferenceHeight = 50
	WindReferenceMeanVelocity = 3
	WeibullShapeFactor = 2.0
	ShearFactor = 0.25

	RatedPower = 1.5e6

	# Years used for analysis
	Years = 25
	DiscountRate = 0.08
	############################################################################


	############################################################################
	### 1. Aerodynamic and structural performance using RotorSE
	rotor = RotorSE()
	# -------------------

	# === blade grid ===
	# (Array): initial aerodynamic grid on unit radius
	rotor.initial_aero_grid = np.array([0.02222276, 0.06666667, 0.11111057, \
		0.16666667, 0.23333333, 0.3, 0.36666667, 0.43333333, 0.5, 0.56666667, \
		0.63333333, 0.7, 0.76666667, 0.83333333, 0.88888943, 0.93333333, \
	    0.97777724]) 

	 # (Array): initial structural grid on unit radius
	rotor.initial_str_grid = np.array([0.0, 0.00492790457512, 0.00652942887106, 
		0.00813095316699, 0.00983257273154, 0.0114340970275, 0.0130356213234, 
		0.02222276, 0.024446481932, 0.026048006228, 0.06666667, 0.089508406455,
	    0.11111057, 0.146462614229, 0.16666667, 0.195309105255, 0.23333333, 
	    0.276686558545, 0.3, 0.333640766319,0.36666667, 0.400404310407, 0.43333333, 
	    0.5, 0.520818918408, 0.56666667, 0.602196371696, 0.63333333,
	    0.667358391486, 0.683573824984, 0.7, 0.73242031601, 0.76666667, 0.83333333, 
	    0.88888943, 0.93333333, 0.97777724, 1.0]) 

	# (Int): first idx in r_aero_unit of non-cylindrical section, 
	# constant twist inboard of here
	rotor.idx_cylinder_aero = 3  

	# (Int): first idx in r_str_unit of non-cylindrical section
	rotor.idx_cylinder_str = 14  

	# (Float): hub location as fraction of radius
	rotor.hubFraction = 0.025  
	# ------------------

	# === blade geometry ===
	# (Array): new aerodynamic grid on unit radius
	rotor.r_aero = np.array([0.02222276, 0.06666667, 0.11111057, 0.2, 0.23333333, 
		0.3, 0.36666667, 0.43333333, 0.5, 0.56666667, 0.63333333, 0.64, 0.7, 
		0.83333333, 0.88888943, 0.93333333, 0.97777724])  

	# (Float): location of max chord on unit radius
	rotor.r_max_chord = 0.23577

	# (Array, m): chord at control points. defined at hub, then at linearly spaced
	# locations from r_max_chord to tip
	ReferenceChord = [3.2612, 4.5709, 3.3178, 1.4621]
	rotor.chord_sub = [x * np.true_divide(BladeLength,ReferenceBladeLength) \
		for x in ReferenceChord]

	# (Array, deg): twist at control points.  defined at linearly spaced locations 
	# from r[idx_cylinder] to tip
	rotor.theta_sub = [13.2783, 7.46036, 2.89317, -0.0878099]  

	# (Array, m): precurve at control points.  defined at same locations at chord, 
	# starting at 2nd control point (root must be zero precurve)
	rotor.precurve_sub = [0.0, 0.0, 0.0] 

	# (Array, m): adjustment to precurve to account for curvature from loading
	rotor.delta_precurve_sub = [0.0, 0.0, 0.0]  

	# (Array, m): spar cap thickness parameters
	rotor.sparT = [0.05, 0.047754, 0.045376, 0.031085, 0.0061398] 

	# (Array, m): trailing-edge thickness parameters
	rotor.teT = [0.1, 0.09569, 0.06569, 0.02569, 0.00569]  

	# (Float, m): blade length (if not precurved or swept) 
	# otherwise length of blade before curvature
	rotor.bladeLength = BladeLength  

	# (Float, m): adjustment to blade length to account for curvature from 
	# loading
	rotor.delta_bladeLength = 0.0  
	rotor.precone = 2.5  # (Float, deg): precone angle
	rotor.tilt = 5.0  # (Float, deg): shaft tilt
	rotor.yaw = 0.0  # (Float, deg): yaw error
	rotor.nBlades = 3  # (Int): number of blades
	# ------------------

	# === airfoil files ===
	basepath = os.path.join(os.path.dirname(\
		os.path.realpath(__file__)), '5MW_AFFiles')

	# load all airfoils
	airfoil_types = [0]*8
	airfoil_types[0] = os.path.join(basepath, 'Cylinder1.dat')
	airfoil_types[1] = os.path.join(basepath, 'Cylinder2.dat')
	airfoil_types[2] = os.path.join(basepath, 'DU40_A17.dat')
	airfoil_types[3] = os.path.join(basepath, 'DU35_A17.dat')
	airfoil_types[4] = os.path.join(basepath, 'DU30_A17.dat')
	airfoil_types[5] = os.path.join(basepath, 'DU25_A17.dat')
	airfoil_types[6] = os.path.join(basepath, 'DU21_A17.dat')
	airfoil_types[7] = os.path.join(basepath, 'NACA64_A17.dat')

	# place at appropriate radial stations
	af_idx = [0, 0, 1, 2, 3, 3, 4, 5, 5, 6, 6, 7, 7, 7, 7, 7, 7]

	n = len(af_idx)
	af = [0]*n
	for i in range(n):
	    af[i] = airfoil_types[af_idx[i]]
	rotor.airfoil_files = af  # (List): names of airfoil file
	# ----------------------

	# === atmosphere ===
	rotor.rho = 1.225  # (Float, kg/m**3): density of air
	rotor.mu = 1.81206e-5  # (Float, kg/m/s): dynamic viscosity of air
	rotor.shearExp = 0.25  # (Float): shear exponent
	rotor.hubHt = HubHeight  # (Float, m): hub height
	rotor.turbine_class = 'I'  # (Enum): IEC turbine class
	rotor.turbulence_class = 'B'  # (Enum): IEC turbulence class class
	rotor.cdf_reference_height_wind_speed = 30.0 
	rotor.g = 9.81  # (Float, m/s**2): acceleration of gravity
	# ----------------------

	# === control ===
	rotor.control.Vin = 3.0  # (Float, m/s): cut-in wind speed
	rotor.control.Vout = 26.0  # (Float, m/s): cut-out wind speed
	rotor.control.ratedPower = RatedPower  # (Float, W): rated power
	
	# (Float, rpm): minimum allowed rotor rotation speed

	# (Float, rpm): maximum allowed rotor rotation speed
	rotor.control.minOmega = 0.0  
	rotor.control.maxOmega = MaximumRotSpeed

	# (Float): tip-speed ratio in Region 2 (should be optimized externally)
	rotor.control.tsr = 7
	# (Float, deg): pitch angle in region 2 (and region 3 for fixed pitch machines)
	rotor.control.pitch = 0.0  
	# (Float, deg): worst-case pitch at survival wind condition
	rotor.pitch_extreme = 0.0  

	# (Float, deg): worst-case azimuth at survival wind condition
	rotor.azimuth_extreme = 0.0  

	# (Float): fraction of rated speed at which the deflection is assumed to 
	# representative throughout the power curve calculation
	rotor.VfactorPC = 0.7  
	# ----------------------

	# === aero and structural analysis options ===

	# (Int): number of sectors to divide rotor face into in computing thrust and power
	rotor.nSector = 4 

	# (Int): number of points to evaluate aero analysis at
	rotor.npts_coarse_power_curve = 20  

	# (Int): number of points to use in fitting spline to power curve
	rotor.npts_spline_power_curve = 200  

	# (Float): availability and other losses (soiling, array, etc.)
	rotor.AEP_loss_factor = 1.0 
	rotor.drivetrainType = 'geared'  # (Enum)

	# (Int): number of natural frequencies to compute
	rotor.nF = 5 

	# (Float): a dynamic amplification factor to adjust the static deflection 
	# calculation
	rotor.dynamic_amplication_tip_deflection = 1.35 
	# ----------------------

	# === materials and composite layup  ===
	basepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), \
		'5MW_PrecompFiles')

	materials = Orthotropic2DMaterial.listFromPreCompFile(os.path.join(basepath,\
	 'materials.inp'))

	ncomp = len(rotor.initial_str_grid)
	upper = [0]*ncomp
	lower = [0]*ncomp
	webs = [0]*ncomp
	profile = [0]*ncomp

	# (Array): array of leading-edge positions from a reference blade axis 
	# (usually blade pitch axis). locations are normalized by the local chord 
	# length. e.g. leLoc[i] = 0.2 means leading edge is 0.2*chord[i] from reference 
	# axis.  positive in -x direction for airfoil-aligned coordinate system
	rotor.leLoc = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.498, 0.497, 
		0.465, 0.447, 0.43, 0.411, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 
		0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4])  

	# (Array): index of sector for spar (PreComp definition of sector)
	rotor.sector_idx_strain_spar = [2]*ncomp  

	# (Array): index of sector for trailing-edge (PreComp definition of sector)
	rotor.sector_idx_strain_te = [3]*ncomp  

	web1 = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.4114, 0.4102, 
		0.4094, 0.3876, 0.3755, 0.3639, 0.345, 0.3342, 0.3313, 0.3274, 0.323, 
		0.3206, 0.3172, 0.3138, 0.3104, 0.307, 0.3003, 0.2982, 0.2935, 0.2899, 
		0.2867, 0.2833, 0.2817, 0.2799, 0.2767, 0.2731, 0.2664, 0.2607, 0.2562, 
		0.1886, -1.0])

	web2 = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.5886, 0.5868, 
		0.5854, 0.5508, 0.5315, 0.5131, 0.4831, 0.4658, 0.4687, 0.4726, 0.477, 
		0.4794, 0.4828, 0.4862, 0.4896, 0.493, 0.4997, 0.5018, 0.5065, 0.5101, 
		0.5133, 0.5167, 0.5183, 0.5201, 0.5233, 0.5269, 0.5336, 0.5393, 0.5438, 
		0.6114, -1.0])
	web3 = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 
		-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
		1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 
		-1.0, -1.0])

	# (Array, m): chord distribution for reference section, thickness of structural 
	# layup scaled with reference thickness (fixed t/c for this case)
	rotor.chord_str_ref = np.array([3.2612, 3.3100915356, 3.32587052924, 
		3.34159388653, 3.35823798667, 3.37384375335, 3.38939112914, 3.4774055542, 
		3.49839685, 3.51343645709, 3.87017220335, 4.04645623801, 4.19408216643,
	    4.47641008477, 4.55844487985, 4.57383098262, 4.57285771934, 4.51914315648, 
	    4.47677655262, 4.40075650022, 4.31069949379, 4.20483735936, 4.08985563932, 
	    3.82931757126, 3.74220276467, 3.54415796922, 3.38732428502, 3.24931446473, 
	    3.23421422609, 3.22701537997, 3.21972125648, 3.08979310611, 2.95152261813, 
	    2.330753331, 2.05553464181, 1.82577817774, 1.5860853279, 1.4621])* \
		np.true_divide(BladeLength,ReferenceBladeLength)


	for i in range(ncomp):
	    webLoc = []
	    if web1[i] != -1:
	        webLoc.append(web1[i])
	    if web2[i] != -1:
	        webLoc.append(web2[i])
	    if web3[i] != -1:
	        webLoc.append(web3[i])

	    upper[i], lower[i], webs[i] = CompositeSection.initFromPreCompLayupFile\
	    (os.path.join(basepath, 'layup_' + str(i+1) + '.inp'), webLoc, materials)
	    profile[i] = Profile.initFromPreCompFile(os.path.join(basepath, 'shape_' \
	    	+ str(i+1) + '.inp'))

	# (List): list of all Orthotropic2DMaterial objects used in 
	# defining the geometry
	rotor.materials = materials 

	# (List): list of CompositeSection objections defining the properties for 
	# upper surface
	rotor.upperCS = upper  

	# (List): list of CompositeSection objections defining the properties for 
	# lower surface
	rotor.lowerCS = lower  

	# (List): list of CompositeSection objections defining the properties for 
	# shear webs
	rotor.websCS = webs  

	# (List): airfoil shape at each radial position
	rotor.profile = profile  
	# --------------------------------------


	# === fatigue ===

	# (Array): nondimensional radial locations of damage equivalent moments
	rotor.rstar_damage = np.array([0.000, 0.022, 0.067, 0.111, 0.167, 0.233, 0.300,
	 0.367, 0.433, 0.500, 0.567, 0.633, 0.700, 0.767, 0.833, 0.889, 0.933, 0.978]) 


	# (Array, N*m): damage equivalent moments about blade c.s. x-direction
	rotor.Mxb_damage = 1e3*np.array([2.3743E+003, 2.0834E+003, 1.8108E+003, 
		1.5705E+003, 1.3104E+003, 1.0488E+003, 8.2367E+002, 6.3407E+002, 
		4.7727E+002, 3.4804E+002, 2.4458E+002, 1.6339E+002, 1.0252E+002,
		 5.7842E+001, 2.7349E+001, 1.1262E+001, 3.8549E+000, 4.4738E-001])  

	# (Array, N*m): damage equivalent moments about blade c.s. y-direction
	rotor.Myb_damage = 1e3*np.array([2.7732E+003, 2.8155E+003, 2.6004E+003, 
		2.3933E+003, 2.1371E+003, 1.8459E+003, 1.5582E+003, 1.2896E+003, 
		1.0427E+003, 8.2015E+002, 6.2449E+002, 4.5229E+002, 3.0658E+002, 
		1.8746E+002, 9.6475E+001, 4.2677E+001, 1.5409E+001, 1.8426E+000])  

	rotor.strain_ult_spar = 1.0e-2  # (Float): ultimate strain in spar cap

	# (Float): uptimate strain in trailing-edge panels, note that I am putting a 
	# factor of two for the damage part only.
	rotor.strain_ult_te = 2500*1e-6 * 2  
	rotor.eta_damage = 1.35*1.3*1.0  # (Float): safety factor for fatigue
	rotor.m_damage = 10.0  # (Float): slope of S-N curve for fatigue analysis

	# (Float): number of cycles used in fatigue analysis  
	rotor.N_damage = 365*24*3600*20.0  
	# ----------------

	# from myutilities import plt

	# === run and outputs ===
	rotor.run()

	# Evaluate AEP Using Lewis' Functions
	# Weibull Wind Parameters
	WindReferenceHeight = 50
	WindReferenceMeanVelocity = 7.5
	WeibullShapeFactor = 2.0
	ShearFactor = 0.25

	PowerCurve = rotor.P/1e6
	PowerCurveVelocity = rotor.V

	HubHeight = rotor.hubHt

	AEP,WeibullScale = CalculateAEPWeibull(PowerCurve,PowerCurveVelocity, HubHeight, \
	  	BladeLength,WeibullShapeFactor, WindReferenceHeight, \
	  	WindReferenceMeanVelocity, ShearFactor)

	NamePlateCapacity = EstimateCapacity(PowerCurve,PowerCurveVelocity, \
		rotor.ratedConditions.V)

	# AEP At Constant 7.5m/s Wind used for benchmarking...
	#AEP = CalculateAEPConstantWind(PowerCurve, PowerCurveVelocity, 7.5)

	if (Verbose ==True):
		print '###################     ROTORSE   ######################'
		print 'AEP = %d MWH' %(AEP)
		print 'NamePlateCapacity = %fMW' %(NamePlateCapacity)
		print 'diameter =', rotor.diameter
		print 'ratedConditions.V =', rotor.ratedConditions.V
		print 'ratedConditions.Omega =', rotor.ratedConditions.Omega
		print 'ratedConditions.pitch =', rotor.ratedConditions.pitch
		print 'mass_one_blade =', rotor.mass_one_blade
		print 'mass_all_blades =', rotor.mass_all_blades
		print 'I_all_blades =', rotor.I_all_blades
		print 'freq =', rotor.freq
		print 'tip_deflection =', rotor.tip_deflection
		print 'root_bending_moment =', rotor.root_bending_moment
		print '#########################################################'

	#############################################################################
	### 2. Hub Sizing 
	# Specify hub parameters based off rotor

	# Load default hub model
	hubS = HubSE()
	hubS.rotor_diameter = rotor.Rtip*2 # m
	hubS.blade_number  = rotor.nBlades
	hubS.blade_root_diameter = rotor.chord_sub[0]*1.25
	hubS.L_rb = rotor.hubFraction*rotor.diameter
	hubS.MB1_location = np.array([-0.5, 0.0, 0.0])
	hubS.machine_rating = rotor.control.ratedPower
	hubS.blade_mass = rotor.mass_one_blade
	hubS.rotor_bending_moment = rotor.root_bending_moment

	hubS.run()

	RotorTotalWeight = rotor.mass_all_blades + hubS.spinner.mass + \
	hubS.hub.mass + hubS.pitchSystem.mass

	if (Verbose==True):
		print '##################### Hub SE ############################'
		print "Estimate of Hub Component Sizes:"
		print "Hub Components"
		print '  Hub: {0:8.1f} kg'.format(hubS.hub.mass)
		print '  Pitch system: {0:8.1f} kg'.format(hubS.pitchSystem.mass) 
		print '  Nose cone: {0:8.1f} kg'.format(hubS.spinner.mass)
		print 'Rotor Total Weight = %d kg' %RotorTotalWeight
		print '#########################################################'


	############################################################################
	### 3. Drive train + Nacelle Mass estimation
	nace = Drive4pt()
	nace.rotor_diameter = rotor.Rtip *2 # m
	nace.rotor_speed = rotor.ratedConditions.Omega # #rpm m/s
	nace.machine_rating = hubS.machine_rating/1000
	nace.DrivetrainEfficiency = 0.95

	 # 6.35e6 #4365248.74 # Nm
	nace.rotor_torque =  rotor.ratedConditions.Q
	nace.rotor_thrust = rotor.ratedConditions.T # N
	nace.rotor_mass = 0.0 #accounted for in F_z # kg

	nace.rotor_bending_moment_x = rotor.Mxyz_0[0]
	nace.rotor_bending_moment_y = rotor.Mxyz_0[1]
	nace.rotor_bending_moment_z = rotor.Mxyz_0[2]

	nace.rotor_force_x = rotor.Fxyz_0[0] # N
	nace.rotor_force_y = rotor.Fxyz_0[1]
	nace.rotor_force_z = rotor.Fxyz_0[2] # N

	# geared 3-stage Gearbox with induction generator machine
	nace.drivetrain_design = 'geared' 
	nace.gear_ratio = 96.76 # 97:1 as listed in the 5 MW reference document
	nace.gear_configuration = 'eep' # epicyclic-epicyclic-parallel

	nace.crane = True # onboard crane present
	nace.shaft_angle = 5.0 #deg
	nace.shaft_ratio = 0.10
	nace.Np = [3,3,1]
	nace.ratio_type = 'optimal'
	nace.shaft_type = 'normal'
	nace.uptower_transformer=False
	nace.shrink_disc_mass = 333.3*nace.machine_rating/1000.0 # estimated
	nace.mb1Type = 'CARB'
	nace.mb2Type = 'SRB'
	nace.flange_length = 0.5 #m
	nace.overhang = 5.0
	nace.gearbox_cm = 0.1
	nace.hss_length = 1.5

	#0 if no fatigue check, 1 if parameterized fatigue check, 
	#2 if known loads inputs
	nace.check_fatigue = 0 
	nace.blade_number=rotor.nBlades
	nace.cut_in=rotor.control.Vin #cut-in m/s
	nace.cut_out=rotor.control.Vout #cut-out m/s
	nace.Vrated=rotor.ratedConditions.V #rated windspeed m/s
	nace.weibull_k = WeibullShapeFactor # windepeed distribution shape parameter

	# windspeed distribution scale parameter
	nace.weibull_A = WeibullScale  

	nace.T_life=20. #design life in years
	nace.IEC_Class_Letter = 'B'

	# length from hub center to main bearing, leave zero if unknown
	nace.L_rb = hubS.L_rb 

	# NREL 5 MW Tower Variables
	nace.tower_top_diameter = 3.78 # m

	nace.run()

	if (Verbose==True):
		print '##################### Drive SE ############################'
		print "Estimate of Nacelle Component Sizes"
		print 'Low speed shaft: {0:8.1f} kg'.format(nace.lowSpeedShaft.mass)
		print 'Main bearings: {0:8.1f} kg'.format(\
			nace.mainBearing.mass + nace.secondBearing.mass)
		print 'Gearbox: {0:8.1f} kg'.format(nace.gearbox.mass)
		print 'High speed shaft & brakes: {0:8.1f} kg'.format\
			(nace.highSpeedSide.mass)
		print 'Generator: {0:8.1f} kg'.format(nace.generator.mass)
		print 'Variable speed electronics: {0:8.1f} kg'.format(\
			nace.above_yaw_massAdder.vs_electronics_mass)
		print 'Overall mainframe:{0:8.1f} kg'.format(\
			nace.above_yaw_massAdder.mainframe_mass)
		print '     Bedplate: {0:8.1f} kg'.format(nace.bedplate.mass)
		print 'Electrical connections: {0:8.1f} kg'.format(\
			nace.above_yaw_massAdder.electrical_mass)
		print 'HVAC system: {0:8.1f} kg'.format(\
			nace.above_yaw_massAdder.hvac_mass )
		print 'Nacelle cover: {0:8.1f} kg'.format(\
			nace.above_yaw_massAdder.cover_mass)
		print 'Yaw system: {0:8.1f} kg'.format(nace.yawSystem.mass)
		print 'Overall nacelle: {0:8.1f} kg'.format(nace.nacelle_mass, \
			nace.nacelle_cm[0], nace.nacelle_cm[1], nace.nacelle_cm[2], \
			nace.nacelle_I[0], nace.nacelle_I[1], nace.nacelle_I[2])  
		print '#########################################################'


	############################################################################
	### 4. Tower Mass

	# --- tower setup ------
	from commonse.environment import PowerWind

	tower = set_as_top(TowerSE())

	# ---- tower ------
	tower.replace('wind1', PowerWind())
	tower.replace('wind2', PowerWind())
	# onshore (no waves)

	# --- geometry ----
	tower.z_param = [0.0, HubHeight*0.5, HubHeight]
	TowerRatio = np.true_divide(HubHeight,ReferenceTowerHeight)

	tower.d_param = [6.0*TowerRatio, 4.935*TowerRatio, 3.87*TowerRatio]
	tower.t_param = [0.027*1.3*TowerRatio, 0.023*1.3*TowerRatio, \
	0.019*1.3*TowerRatio]
	n = 10

	tower.z_full = np.linspace(0.0, HubHeight, n)
	tower.L_reinforced = 15.0*np.ones(n)  # [m] buckling length
	tower.theta_stress = 0.0*np.ones(n)
	tower.yaw = 0.0

	# --- material props ---
	tower.E = 210e9*np.ones(n)
	tower.G = 80.8e9*np.ones(n)
	tower.rho = 8500.0*np.ones(n)
	tower.sigma_y = 450.0e6*np.ones(n)

	# --- spring reaction data.  Use float('inf') for rigid constraints. ---
	tower.kidx = [0]  # applied at base
	tower.kx = [float('inf')]
	tower.ky = [float('inf')]
	tower.kz = [float('inf')]
	tower.ktx = [float('inf')]
	tower.kty = [float('inf')]
	tower.ktz = [float('inf')]

	# --- extra mass ----
	tower.midx = [n-1]  # RNA mass at top
	tower.m = [0.8]
	tower.mIxx = [1.14930678e+08]
	tower.mIyy = [2.20354030e+07]
	tower.mIzz = [1.87597425e+07]
	tower.mIxy = [0.00000000e+00]
	tower.mIxz = [5.03710467e+05]
	tower.mIyz = [0.00000000e+00]
	tower.mrhox = [-1.13197635]
	tower.mrhoy = [0.]
	tower.mrhoz = [0.50875268]
	tower.addGravityLoadForExtraMass = False
	# -----------

	# --- wind ---
	tower.wind_zref = 90.0
	tower.wind_z0 = 0.0
	tower.wind1.shearExp = 0.14
	tower.wind2.shearExp = 0.14
	# ---------------

	# # --- loading case 1: max Thrust ---
	tower.wind_Uref1 = 11.73732
	tower.plidx1 = [n-1]  # at tower top
	tower.Fx1 = [0.19620519]
	tower.Fy1 = [0.]
	tower.Fz1 = [-2914124.84400512]
	tower.Mxx1 = [3963732.76208099]
	tower.Myy1 = [-2275104.79420872]
	tower.Mzz1 = [-346781.68192839]
	# # ---------------

	# # --- loading case 2: max wind speed ---
	tower.wind_Uref2 = 70.0
	tower.plidx1 = [n-1]  # at tower top
	tower.Fx1 = [930198.60063279]
	tower.Fy1 = [0.]
	tower.Fz1 = [-2883106.12368949]
	tower.Mxx1 = [-1683669.22411597]
	tower.Myy1 = [-2522475.34625363]
	tower.Mzz1 = [147301.97023764]
	# # ---------------

	# # --- run ---
	tower.run()

	if (Verbose==True):
		print '##################### Tower SE ##########################'
		print 'mass (kg) =', tower.mass
		print 'f1 (Hz) =', tower.f1
		print 'f2 (Hz) =', tower.f2
		print 'top_deflection1 (m) =', tower.top_deflection1
		print 'top_deflection2 (m) =', tower.top_deflection2
		print '#########################################################'


	############################################################################
	## 5. Turbine captial costs analysis
	turbine = Turbine_CostsSE()

	# NREL 5 MW turbine component masses based on Sunderland model approach
	# Rotor
	# inline with the windpact estimates
	turbine.blade_mass = rotor.mass_one_blade  
	turbine.hub_mass = hubS.hub.mass
	turbine.pitch_system_mass = hubS.pitchSystem.mass
	turbine.spinner_mass = hubS.spinner.mass

	# Drivetrain and Nacelle
	turbine.low_speed_shaft_mass = nace.lowSpeedShaft.mass
	turbine.main_bearing_mass=nace.mainBearing.mass 
	turbine.second_bearing_mass = nace.secondBearing.mass
	turbine.gearbox_mass = nace.gearbox.mass
	turbine.high_speed_side_mass = nace.highSpeedSide.mass
	turbine.generator_mass = nace.generator.mass
	turbine.bedplate_mass = nace.bedplate.mass
	turbine.yaw_system_mass = nace.yawSystem.mass

	# Tower
	turbine.tower_mass = tower.mass*0.5

	# Additional non-mass cost model input variables
	turbine.machine_rating = hubS.machine_rating/1000
	turbine.advanced = False
	turbine.blade_number = rotor.nBlades
	turbine.drivetrain_design = 'geared'
	turbine.crane = False
	turbine.offshore = False

	# Target year for analysis results
	turbine.year = 2010
	turbine.month =  12

	turbine.run()

	if (Verbose==True):
		print '##################### TurbinePrice SE ####################'
		print "Overall rotor cost with 3 advanced blades is ${0:.2f} USD"\
			.format(turbine.rotorCC.cost)
		print "Blade cost is ${0:.2f} USD".format(turbine.rotorCC.bladeCC.cost)
		print "Hub cost is ${0:.2f} USD".format(turbine.rotorCC.hubCC.cost)
		print "Pitch system cost is ${0:.2f} USD".format(turbine.rotorCC.pitchSysCC.cost)
		print "Spinner cost is ${0:.2f} USD".format(turbine.rotorCC.spinnerCC.cost)
		print
		print "Overall nacelle cost is ${0:.2f} USD".format(turbine.nacelleCC.cost)
		print "LSS cost is ${0:.2f} USD".format(turbine.nacelleCC.lssCC.cost)
		print "Main bearings cost is ${0:.2f} USD".format(turbine.nacelleCC.bearingsCC.cost)
		print "Gearbox cost is ${0:.2f} USD".format(turbine.nacelleCC.gearboxCC.cost)
		print "Hight speed side cost is ${0:.2f} USD".format(turbine.nacelleCC.hssCC.cost)
		print "Generator cost is ${0:.2f} USD".format(turbine.nacelleCC.generatorCC.cost)
		print "Bedplate cost is ${0:.2f} USD".format(turbine.nacelleCC.bedplateCC.cost)
		print "Yaw system cost is ${0:.2f} USD".format(turbine.nacelleCC.yawSysCC.cost)
		print
		print "Tower cost is ${0:.2f} USD".format(turbine.towerCC.cost)
		print
		print "The overall turbine cost is ${0:.2f} USD".format(turbine.turbine_cost)
		print '#########################################################'

	############################################################################
	## 6. Operating Expenses

	# A simple test of nrel_csm_bos model
	bos = bos_csm_assembly()

	# Set input parameters
	bos = bos_csm_assembly()
	bos.machine_rating = hubS.machine_rating/1000
	bos.rotor_diameter = rotor.diameter
	bos.turbine_cost = turbine.turbine_cost
	bos.hub_height = HubHeight
	bos.turbine_number = 1
	bos.sea_depth = 0
	bos.year = 2009
	bos.month = 12
	bos.multiplier = 1.0
	bos.run()

	om = opex_csm_assembly()

	om.machine_rating = rotor.control.ratedPower/1000  
	# Need to manipulate input or underlying component will not execute
	om.net_aep = AEP*10e4
	om.sea_depth = 0
	om.year = 2009
	om.month = 12
	om.turbine_number = 100

	om.run()

	if (Verbose==True):
		print '##################### Operating Costs ####################'
		print "BOS cost per turbine: ${0:.2f} USD".format(bos.bos_costs / \
			bos.turbine_number)
		print "Average annual operational expenditures"
		print "OPEX on shore with 100 turbines ${:.2f}: USD".format(\
			om.avg_annual_opex)
		print "Preventative OPEX by turbine: ${:.2f} USD".format(\
			om.opex_breakdown.preventative_opex / om.turbine_number)
		print "Corrective OPEX by turbine: ${:.2f} USD".format(\
			om.opex_breakdown.corrective_opex / om.turbine_number)
		print "Land Lease OPEX by turbine: ${:.2f} USD".format(\
			om.opex_breakdown.lease_opex / om.turbine_number)
		print '#########################################################'

	CapitalCost = turbine.turbine_cost + bos.bos_costs / bos.turbine_number
	OperatingCost = om.opex_breakdown.preventative_opex / om.turbine_number + \
	om.opex_breakdown.lease_opex / om.turbine_number + \
	om.opex_breakdown.corrective_opex / om.turbine_number

	LCOE = ComputeLCOE(AEP, CapitalCost, OperatingCost, DiscountRate, Years)

	print '######################***********************###################'
	print "Levelized Cost of Energy over %d years \
	is $%f/kWH" %(Years,LCOE/1000)
	print '######################***********************###################'

	return LCOE/1000