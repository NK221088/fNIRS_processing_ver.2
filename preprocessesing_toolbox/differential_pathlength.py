

def compute_differential_pathlength(age, _lambda = [760, 850], alpha=223.3, beta=0.05624, delta=-5.723*10**(-7), gamma = 0.8493, epsilon=0.001245, zeta=-0.9025):
    """
    Scholkmann & Wolf (2013) style DPF regression:
      DPF(λ, age) = α + β * age^γ + δ * λ^3 + ε * λ^2 + ζ * λ
    """
    ppf = ()
    for wavelength in _lambda:
        ppf += (alpha + beta * age**gamma + delta*wavelength**3 + epsilon*wavelength**2 + zeta * wavelength,)
    return ppf