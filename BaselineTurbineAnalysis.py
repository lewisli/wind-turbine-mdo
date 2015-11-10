# BaselineTurbineAnalysis.py
#
# NREL 5MW Wind Turbine Analysis
# Using parameters specified by the NREL report for the 5MW tower, perform 
# aerodynamic, structural and cost analysis on the turbine to verify that the 
# numbers are reasonable.
#
# Author: Lewis Li (lewisli@stanford.edu)
# Original Date: November 1st 2015

import os
from math import pi
import numpy as np
from pyopt_driver.pyopt_driver import pyOptDriver
from openmdao.lib.casehandlers.api import DumpCaseRecorder

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

################################################################################
### 1. Aerodynamic and structural performance using RotorSE

# Basic Rotor Model
cdf_type = 'weibull'
#rotor = RotorSE()
rotor = RotorAeroVSVPWithCCBlade(cdf_type)

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
rotor.hubHt = 100.0  # (Float, m)

# Blade length (if not precurved or swept) otherwise length of blade before curvature
rotor.bladeLength = 61.5  # (Float, m): 

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

# Wind Parameters are specified here
rotor.rho = 1.225  # (Float, kg/m**3): density of air
rotor.mu = 1.81206e-5  # (Float, kg/m/s): dynamic viscosity of air
rotor.shearExp = 0.2  # (Float): shear exponent
rotor.cdf_mean_wind_speed = 6.0  # (Float, m/s): mean wind speed of site cumulative distribution function
rotor.weibull_shape_factor = 2.0  # (Float): shape factor of weibull distribution

# Rotor fixed design parameters
rotor.control.Vin = 3.0  # (Float, m/s): cut-in wind speed
rotor.control.Vout = 25.0  # (Float, m/s): cut-out wind speed
rotor.control.ratedPower = 5e6  # (Float, W): rated power
rotor.control.pitch = 0.0  # (Float, deg): pitch angle in region 2 (and region 3 for fixed pitch machines)
rotor.control.minOmega = 0.0  # (Float, rpm): minimum allowed rotor rotation speed
rotor.control.maxOmega = 12.0  # (Float, rpm): maximum allowed rotor rotation speed
rotor.control.tsr = 7.55  # **dv** (Float): tip-speed ratio in Region 2 (should be

rotor.nSector = 4  # (Int): number of sectors to divide rotor face into in computing thrust and power
rotor.npts_coarse_power_curve = 20  # (Int): number of points to evaluate aero analysis at
rotor.npts_spline_power_curve = 200  # (Int): number of points to use in fitting spline to power curve
rotor.AEP_loss_factor = 1.0  # (Float): availability and other losses (soiling, array, etc.)
rotor.tiploss = True  # (Bool): include Prandtl tip loss model
rotor.hubloss = True  # (Bool): include Prandtl hub loss model
rotor.wakerotation = True  # (Bool): include effect of wake rotation (i.e., tangential induction factor is nonzero)
rotor.usecd = True  # (Bool): use drag coefficient in computing induction factors
rotor.VfactorPC = 0.7  # (Float): fraction of rated speed at which the deflection is assumed to representative throughout the power curve calculation

# Run to calculate rotor parameters
rotor.run()

AEP0 = rotor.AEP
print 'AEP =', rotor.AEP
print 'diameter =', rotor.diameter
print 'ratedConditions.V =', rotor.ratedConditions.V
print 'ratedConditions.Omega =', rotor.ratedConditions.Omega
print 'ratedConditions.pitch =', rotor.ratedConditions.pitch
print 'ratedConditions.T =', rotor.ratedConditions.T
print 'ratedConditions.Q =', rotor.ratedConditions.Q
#print 'mass_one_blade =', rotor.mass_one_blade
# print 'mass_all_blades =', rotor.mass_all_blades
# print 'I_all_blades =', rotor.I_all_blades
# print 'freq =', rotor.freq
# print 'tip_deflection =', rotor.tip_deflection
# print 'root_bending_moment =', rotor.root_bending_moment

################################################################################
### 2. Hub Sizing 
# Specify hub parameters based off rotor

# Load default hub model
hubS = HubSE()
hubS.rotor_diameter = rotor.Rtip*2 # m
hubS.blade_number  = rotor.B 
hubS.blade_root_diameter   = 3.542

hubS.L_rb = rotor.Rhub
hubS.gamma = 5.0
hubS.MB1_location = np.array([-0.5, 0.0, 0.0])
hubS.machine_rating = rotor.control.ratedPower/1000

hubS.run()

print "Estimate of Hub Component Sizes for the Baseline 5 MW Reference Turbine"
print "Hub Components"
print '  Hub: {0:8.1f} kg'.format(hubS.hub.mass)  # 31644.47
print '  Pitch system: {0:8.1f} kg'.format(hubS.pitchSystem.mass) # 17003.98
print '  Nose cone: {0:8.1f} kg'.format(hubS.spinner.mass) # 1810.50

################################################################################
### 3. Drive train + Nacelle Mass estimation
# NREL 5 MW Rotor Variables
nace = Drive4pt()
nace.rotor_diameter = rotor.Rtip *2 # m
nace.rotor_speed = rotor.control.maxOmega # #rpm m/s
nace.machine_rating = hubS.machine_rating
nace.DrivetrainEfficiency = 0.95

 # 6.35e6 #4365248.74 # Nm
nace.rotor_torque =  1.5 * (nace.machine_rating * 1000 / \
	nace.DrivetrainEfficiency) / (nace.rotor_speed * (pi / 30))
nace.rotor_thrust = 599610.0 # N
nace.rotor_mass = 0.0 #accounted for in F_z # kg

nace.rotor_bending_moment = -16665000.0 # Nm same as rotor_bending_moment_y
nace.rotor_bending_moment_x = 330770.0# Nm
nace.rotor_bending_moment_y = -16665000.0 # Nm
nace.rotor_bending_moment_z = 2896300.0 # Nm
nace.rotor_force_x = 599610.0 # N
nace.rotor_force_y = 186780.0 # N
nace.rotor_force_z = -842710.0 # N

# NREL 5 MW Drivetrain variables
nace.drivetrain_design = 'geared' # geared 3-stage Gearbox with induction generator machine
nace.gear_ratio = 96.76 # 97:1 as listed in the 5 MW reference document
nace.gear_configuration = 'eep' # epicyclic-epicyclic-parallel
#nace.bevel = 0 # no bevel stage
nace.crane = True # onboard crane present
nace.shaft_angle = 5.0 #deg
nace.shaft_ratio = 0.10
nace.Np = [3,3,1]
nace.ratio_type = 'optimal'
nace.shaft_type = 'normal'
nace.uptower_transformer=False
nace.shrink_disc_mass = 333.3*nace.machine_rating/1000.0 # estimated
nace.carrier_mass = 8000.0 # estimated
nace.mb1Type = 'CARB'
nace.mb2Type = 'SRB'
nace.flange_length = 0.5 #m
nace.overhang = 5.0
nace.gearbox_cm = 0.1
nace.hss_length = 1.5
nace.check_fatigue = 0 #0 if no fatigue check, 1 if parameterized fatigue check, 2 if known loads inputs
nace.blade_number=rotor.B
nace.cut_in=rotor.control.Vin #cut-in m/s
nace.cut_out=rotor.control.Vout #cut-out m/s
nace.Vrated=rotor.ratedConditions.V #rated windspeed m/s
nace.weibull_k = rotor.weibull_shape_factor # windepeed distribution shape parameter

# Might need to change this...
nace.weibull_A = rotor.cdf_mean_wind_speed  # windspeed distribution scale parameter

nace.T_life=20. #design life in years
nace.IEC_Class_Letter = 'A'
nace.L_rb = hubS.L_rb # length from hub center to main bearing, leave zero if unknown

# NREL 5 MW Tower Variables
nace.tower_top_diameter = 3.78 # m

nace.run()

print "Estimate of Nacelle Component Sizes for the NREL 5 MW Reference Turbine"
print 'Low speed shaft: {0:8.1f} kg'.format(nace.lowSpeedShaft.mass)
print 'Main bearings: {0:8.1f} kg'.format(nace.mainBearing.mass + nace.secondBearing.mass)
print 'Gearbox: {0:8.1f} kg'.format(nace.gearbox.mass)
print 'High speed shaft & brakes: {0:8.1f} kg'.format(nace.highSpeedSide.mass)
print 'Generator: {0:8.1f} kg'.format(nace.generator.mass)
print 'Variable speed electronics: {0:8.1f} kg'.format(nace.above_yaw_massAdder.vs_electronics_mass)
print 'Overall mainframe:{0:8.1f} kg'.format(nace.above_yaw_massAdder.mainframe_mass)
print '     Bedplate: {0:8.1f} kg'.format(nace.bedplate.mass)
print 'Electrical connections: {0:8.1f} kg'.format(nace.above_yaw_massAdder.electrical_mass)
print 'HVAC system: {0:8.1f} kg'.format(nace.above_yaw_massAdder.hvac_mass )
print 'Nacelle cover: {0:8.1f} kg'.format(nace.above_yaw_massAdder.cover_mass)
print 'Yaw system: {0:8.1f} kg'.format(nace.yawSystem.mass)
print 'Overall nacelle: {0:8.1f} kg'.format(nace.nacelle_mass, nace.nacelle_cm[0], nace.nacelle_cm[1], nace.nacelle_cm[2], nace.nacelle_I[0], nace.nacelle_I[1], nace.nacelle_I[2]  )


################################################################################
### 4. Tower Analysis
## Problem with TowerSE..


################################################################################
## 5. Turbine captial costs analysis
## Needs to be coupled with rest of system...

from turbine_costsse.turbine_costsse.turbine_costsse import Turbine_CostsSE

turbine = Turbine_CostsSE()

# NREL 5 MW turbine component masses based on Sunderland model approach
# Rotor
turbine.blade_mass = 17650.67  # inline with the windpact estimates
turbine.hub_mass = 31644.5
turbine.pitch_system_mass = 17004.0
turbine.spinner_mass = 1810.5

# Drivetrain and Nacelle
turbine.low_speed_shaft_mass = 31257.3
#bearingsMass = 9731.41
turbine.main_bearing_mass = 9731.41 / 2
turbine.second_bearing_mass = 9731.41 / 2
turbine.gearbox_mass = 30237.60
turbine.high_speed_side_mass = 1492.45
turbine.generator_mass = 16699.85
turbine.bedplate_mass = 93090.6
turbine.yaw_system_mass = 11878.24

# Tower
turbine.tower_mass = 434559.0

# Additional non-mass cost model input variables
turbine.machine_rating = 5000.0
turbine.advanced = True
turbine.blade_number = 3
turbine.drivetrain_design = 'geared'
turbine.crane = True
turbine.offshore = True

# Target year for analysis results
turbine.year = 2010
turbine.month =  12

turbine.run()

print "Overall rotor cost with 3 advanced blades is ${0:.2f} USD".format(turbine.rotorCC.cost)
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
print





AEP0 = rotor.AEP
print 'AEP0 = %d MWH' % (AEP0/1000)

# import matplotlib.pyplot as plt
# plt.plot(rotor.V, rotor.P/1e6)
# plt.xlabel('wind speed (m/s)')
# plt.ylabel('power (MW)')
# plt.show()
