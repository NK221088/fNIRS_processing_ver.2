from datetime import datetime

def compute_differential_pathlength(raw_od, alpha=223.3, beta=0.05624, delta=-5.723*10**(-7), gamma = 0.8493, epsilon=0.001245, zeta=-0.9025):
    """
    Scholkmann & Wolf (2013) style DPF regression:
      DPF(λ, age) = α + β * age^γ + δ * λ^3 + ε * λ^2 + ζ * λ
    """
    
    age = datetime.now().year - raw_od.info["subject_info"]["birthday"][0]
    _lambda = tuple([float(raw_od.ch_names[:2][i][-3:]) for i in range(2)])
    dpf = ()
    for wavelength in _lambda:
        dpf += (alpha + beta * age**gamma + delta*wavelength**3 + epsilon*wavelength**2 + zeta * wavelength,)
    return dpf