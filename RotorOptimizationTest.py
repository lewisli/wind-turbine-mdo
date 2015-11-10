import numpy as np
from math import pi, gamma
from openmdao.main.datatypes.api import Int, Float, Array, Str, List, Enum, VarTree, Bool
from openmdao.main.api import Component, Assembly

from ccblade import CCAirfoil, CCBlade as CCBlade_PY
from commonse.utilities import sind, cosd, smooth_abs, smooth_min, hstack, vstack, linspace_with_deriv
from rotoraero import GeomtrySetupBase, AeroBase, DrivetrainLossesBase, CDFBase, \
    VarSpeedMachine, FixedSpeedMachine, RatedConditions, common_configure
from akima import Akima


from commonse.utilities import check_gradient_unit_test, check_for_missing_unit_tests
from rotorse.rotoraero import Coefficients, SetupRunVarSpeed, RegulatedPowerCurve, AEP
from rotorse.rotoraerodefaults import RotorAeroVSVPWithCCBlade,GeometrySpline, CCBladeGeometry, CCBlade, CSMDrivetrain, WeibullCDF, WeibullWithMeanCDF, RayleighCDF


optimize = True

import os

# --- instantiate rotor ----
cdf_type = 'weibull'
rotor = RotorAeroVSVPWithCCBlade(cdf_type)
# --------------------------------

# --- rotor geometry ------------
rotor.r_max_chord = 0.23577  # (Float): location of second control point (generally also max chord)
rotor.chord_sub = [3.2612, 4.5709, 3.3178, 1.4621]  # (Array, m): chord at control points
rotor.theta_sub = [13.2783, 7.46036, 2.89317, -0.0878099]  # (Array, deg): twist at control points
rotor.Rhub = 1.5  # (Float, m): hub radius
rotor.Rtip = 63.0  # (Float, m): tip radius
rotor.precone = 2.5  # (Float, deg): precone angle
rotor.tilt = -5.0  # (Float, deg): shaft tilt
rotor.yaw = 0.0  # (Float, deg): yaw error
rotor.B = 3  # (Int): number of blades
# -------------------------------------

# --- airfoils ------------
basepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), '5MW_AFFiles')

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
rotor.r_af = np.array([0.02222276, 0.06666667, 0.11111057, 0.16666667, 0.23333333, 0.3, 0.36666667,
    0.43333333, 0.5, 0.56666667, 0.63333333, 0.7, 0.76666667, 0.83333333, 0.88888943,
    0.93333333, 0.97777724])    # (Array, m): locations where airfoils are defined on unit radius
rotor.idx_cylinder = 3  # (Int): index in r_af where cylinder section ends
# -------------------------------------

# --- site characteristics --------
rotor.rho = 1.225  # (Float, kg/m**3): density of air
rotor.mu = 1.81206e-5  # (Float, kg/m/s): dynamic viscosity of air
rotor.shearExp = 0.2  # (Float): shear exponent
rotor.hubHt = 80.0  # (Float, m)
rotor.cdf_mean_wind_speed = 6.0  # (Float, m/s): mean wind speed of site cumulative distribution function
rotor.weibull_shape_factor = 2.0  # (Float): shape factor of weibull distribution
# -------------------------------------


# --- control settings ------------
rotor.control.Vin = 3.0  # (Float, m/s): cut-in wind speed
rotor.control.Vout = 25.0  # (Float, m/s): cut-out wind speed
rotor.control.ratedPower = 5e6  # (Float, W): rated power
rotor.control.pitch = 0.0  # (Float, deg): pitch angle in region 2 (and region 3 for fixed pitch machines)
rotor.control.minOmega = 0.0  # (Float, rpm): minimum allowed rotor rotation speed
rotor.control.maxOmega = 12.0  # (Float, rpm): maximum allowed rotor rotation speed
rotor.control.tsr = 7.55  # **dv** (Float): tip-speed ratio in Region 2 (should be optimized externally)
# -------------------------------------

# --- drivetrain model for efficiency --------
rotor.drivetrainType = 'geared'
# -------------------------------------


# --- analysis options ------------
rotor.nSector = 4  # (Int): number of sectors to divide rotor face into in computing thrust and power
rotor.npts_coarse_power_curve = 20  # (Int): number of points to evaluate aero analysis at
rotor.npts_spline_power_curve = 200  # (Int): number of points to use in fitting spline to power curve
rotor.AEP_loss_factor = 1.0  # (Float): availability and other losses (soiling, array, etc.)
rotor.tiploss = True  # (Bool): include Prandtl tip loss model
rotor.hubloss = True  # (Bool): include Prandtl hub loss model
rotor.wakerotation = True  # (Bool): include effect of wake rotation (i.e., tangential induction factor is nonzero)
rotor.usecd = True  # (Bool): use drag coefficient in computing induction factors
# -------------------------------------

# --- run ------------
rotor.run()

AEP0 = rotor.AEP
print 'AEP0 =', AEP0

# import matplotlib.pyplot as plt
# plt.plot(rotor.V, rotor.P/1e6)
# plt.xlabel('wind speed (m/s)')
# plt.ylabel('power (MW)')
# #plt.show()

# --------------------------


if optimize:

    # --- optimizer imports ---
    from pyopt_driver.pyopt_driver import pyOptDriver
    from openmdao.lib.casehandlers.api import DumpCaseRecorder
    # ----------------------

    # --- Setup Pptimizer ---
    rotor.replace('driver', pyOptDriver())
    rotor.driver.optimizer = 'KSOPT'
    # rotor.driver.options = {'Major feasibility tolerance': 1e-6,
    #                        'Minor feasibility tolerance': 1e-6,
    #                        'Major optimality tolerance': 1e-5,
    #                        'Function precision': 1e-8}
    # ----------------------

    # --- Objective ---
    rotor.driver.add_objective('-aep.AEP/%f' % AEP0)
    # ----------------------

    # --- Design Variables ---
    rotor.driver.add_parameter('r_max_chord', low=0.1, high=0.5)
    rotor.driver.add_parameter('chord_sub', low=0.4, high=5.3)
    rotor.driver.add_parameter('theta_sub', low=-10.0, high=30.0)
    rotor.driver.add_parameter('control.tsr', low=3.0, high=14.0)
    # ----------------------

    # --- recorder ---
    #rotor.recorders = [DumpCaseRecorder()]
    # ----------------------

    # --- Constraints ---
    rotor.driver.add_constraint('1.0 >= 0.0')  # dummy constraint, OpenMDAO bug when using pyOpt
    # ----------------------

    # --- run opt ---
    rotor.run()
    # ---------------