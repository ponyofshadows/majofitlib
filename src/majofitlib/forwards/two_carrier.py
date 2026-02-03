from __future__ import annotations
from numpy.typing import NDArray

from ..constants import E_CHARGE
from ..base import registered_quantities
from ..base import Data

def two_carrier(data:Data)->Data:
    quantites = {
        ("X","magnetic field"): registered_quantities["magnetic field"],
        ("Y", "longitudinal resistivity"): registered_quantities["resistivity"],
        ("Y", "transverse resistivity"): registered_quantities["resistivity"],
        ("PARA", "carrier 1 charge"): registered_quantities["charge"],
        ("PARA", "carrier 2 charge"): registered_quantities["charge"],
        ("PARA", "carrier 1 concentration"): registered_quantities["carrier"],
    }
    H = data["X","magnetic field"]
    q1 = data["PARA","carrier 1 charge"]
    q2 = data["PARA","carrier 2 charge"]
    n1 = data["PARA",""]


    def model(self, x): 
        n1, n2, mu1, mu2 = self.to_SI(*x)
        sigma1 = self.q1 * n1 * mu1
        sigma2 = self.q2 * n2 * mu2

        def two_carrier_model(B):
            denom1 = 1 + (mu1 * B) ** 2
            denom2 = 1 + (mu2 * B) ** 2
            sigma_xx =  sigma1 / denom1 + sigma2 / denom2
            sigma_xy = B * (sigma1 * mu1 / denom1 + sigma2 * mu2 / denom2)

            D = (sigma_xx **2 + sigma_xy **2)
            rho_xx = sigma_xx / D
            rho_xy = - sigma_xy / D

            return np.vstack((rho_xx, rho_xy))
        
        return two_carrier_model
