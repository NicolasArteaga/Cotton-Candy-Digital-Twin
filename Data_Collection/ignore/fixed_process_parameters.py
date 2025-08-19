"""
Fixed Process Parameters for Cotton Candy Digital Twin

These parameters remain constant across all processes and are not included 
in the feature vector for machine learning optimization.
"""

# Sugar dispensing configuration
SUGAR_AMOUNT = 1  # Setting value (corresponds to 1 second dispensing)
SUGAR_WEIGHT_GRAMS = 12.63  # Actual weight dispensed in grams (calibrated median value)

# Cotton candy geometry (fixed measurements)
RADIUS_METERS = 0.135  # Cotton candy radius in meters
HEIGHT_METERS = 0.05   # Cotton candy height in meters

# Process constants that don't vary
FIXED_PARAMETERS = {
    'sugar_amount_setting': SUGAR_AMOUNT,
    'sugar_weight_grams': SUGAR_WEIGHT_GRAMS,
    'radius_meters': RADIUS_METERS,
    'height_meters': HEIGHT_METERS,
    'dispensing_duration_seconds': 1.0,
    'spinning_time_seconds': 3.75,
    'spins_per_run': 28,  # 105s / 3.75s = 28 spins per run
}

# Calibration notes
CALIBRATION_INFO = {
    'sugar_dispensing': {
        'measurement_trials': 10,
        'duration_0_5s': 8.50,   # grams
        'duration_1_0s': 12.63,  # grams (median, used as standard)
        'duration_1_5s': 16.64,  # grams
        'duration_2_0s': 20.58,  # grams (slightly overflows spoon)
        'relationship': 'roughly linear',
        'standard_setting': 1.0,  # seconds
        'note': 'All experiments use 1s dispensing (12.63g) for consistency'
    }
}

def get_fixed_parameter(parameter_name: str):
    """Get a fixed parameter value by name"""
    return FIXED_PARAMETERS.get(parameter_name)

def get_sugar_weight():
    """Get the standardized sugar weight in grams"""
    return SUGAR_WEIGHT_GRAMS

def get_cotton_candy_volume():
    """Calculate cotton candy volume using oblate spheroid approximation"""
    # V = (4/3) * π * a² * c, where a = radius, c = height/2
    import math
    a = RADIUS_METERS  # equatorial radius
    c = HEIGHT_METERS / 2  # polar radius (half height)
    volume = (4/3) * math.pi * (a**2) * c
    return volume

def get_geometry():
    """Get cotton candy geometry parameters"""
    return {
        'radius_meters': RADIUS_METERS,
        'height_meters': HEIGHT_METERS,
        'volume_cubic_meters': get_cotton_candy_volume()
    }

if __name__ == "__main__":
    print("Fixed Process Parameters:")
    for key, value in FIXED_PARAMETERS.items():
        print(f"  {key}: {value}")
    
    print(f"\nSugar dispensing: {SUGAR_AMOUNT} setting = {SUGAR_WEIGHT_GRAMS}g")
    print(f"Cotton candy geometry: radius={RADIUS_METERS}m, height={HEIGHT_METERS}m")
    print(f"Estimated volume: {get_cotton_candy_volume():.6f} m³")
