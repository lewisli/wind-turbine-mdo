import numpy as np
import math
import matplotlib.pyplot as plt

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx], idx


# Given mean, and shape of Weibull distriution, estimate the scale parameter
def ComputeScaleFunction(Mean, Shape):
	return Mean/math.gamma(1 + 1.0/Shape)

def GenerateWeibull(Shape,Scale,Samples):
	return Scale*np.random.weibull(Shape, Samples)

def EstimateCapacity(PowerCurve,PowerCurveVelocity, DesignWindSpeed):
	ClosestWindSpeed, Idx = find_nearest(PowerCurveVelocity, DesignWindSpeed)
	return PowerCurve[Idx]# Power in MW


# Calculate AEP assuming a Weibull distribution
def CalculateAEPWeibull(PowerCurve, PowerCurveVelocity, HubHeight, WeibullWindShape, \
	WindReferenceHeight, WindReferenceMean, ShearExponent):

	# Use Power Law to estimate mean velocity at design hub height
	HubMeanVelocity = WindReferenceMean*(np.true_divide(HubHeight,\
		WindReferenceHeight))**ShearExponent

	print "Mean Wind Velocity at %d m is %f m/s" %(HubHeight,HubMeanVelocity)

	# Calcluate corresponding scale parameter (Lambda)
	WeibullScale = ComputeScaleFunction(HubMeanVelocity,WeibullWindShape)

	HoursInYear = 365*24
	WindDistribution = GenerateWeibull(WeibullWindShape,WeibullScale,HoursInYear)
	Range = np.min(PowerCurveVelocity), np.max(PowerCurveVelocity)
	NumBins = PowerCurveVelocity.size

	# Count gives the number of hours per year at that wind speed
	count, bins, ignored = plt.hist(WindDistribution,range=Range,bins=NumBins)

	# AEP is equal to rotor.P*count (in MWH)
	AEP = np.dot(count,PowerCurve)

	return AEP,WeibullScale

# Calculate AEP for a given Power Curve assuming constant wind speed
def CalculateAEPConstantWind(PowerCurve, PowerCurveVelocity, WindSpeed):

	# Find corresponding index in PowerCurveVelocity that matches WindSpeed the 
	# closest
	ClosestWindSpeed, Idx = find_nearest(PowerCurveVelocity, WindSpeed)
	GeneratedPower = PowerCurve[Idx] # Power in MW

	HoursInYear = 24*365

	return GeneratedPower*HoursInYear

