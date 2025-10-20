from datetime import datetime
from datetime import date

def compute_differential_pathlength(raw_od, alpha=223.3, beta=0.05624, delta=-5.723*10**(-7), gamma = 0.8493, epsilon=0.001245, zeta=-0.9025):
    """
    Scholkmann & Wolf (2013) style DPF regression:
      DPF(λ, age) = α + β * age^γ + δ * λ^3 + ε * λ^2 + ζ * λ
    """
    measuring_date = raw_od.info["meas_date"]
    try:
      birthday = raw_od.info["subject_info"]["birthday"]
    except:
      return 6
    if type(birthday) == tuple:
      age = (datetime.date(measuring_date) - datetime.date(datetime(*birthday))).days / 365.2425
    elif isinstance(birthday, date):
      age = (datetime.date(measuring_date) - birthday).days / 365.2425
    else:
      return 6
    # 
    _lambda = tuple([float(raw_od.ch_names[:2][i][-3:]) for i in range(2)])
    dpf = 0
    for wavelength in _lambda:
        dpf += alpha + beta * age**gamma + delta*wavelength**3 + epsilon*wavelength**2 + zeta * wavelength
    return dpf