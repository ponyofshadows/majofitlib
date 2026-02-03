from __future__ import annotations
import numpy as np
from numpy.typing import NDArray

from ..constants import E_CHARGE, VAC_PERMEABILITY
from ..base import registered_quantities, model_register
from ..base import Data

@model_register
class PolynomialTwoCarrierModel():
    """
    PolynomailTwoCarrierModel
    ===
    References
    
    J. Appl. Phys. 137, 025702 (2025); doi: 10.1063/5.0248998
    ---
    The double-carrier model can be strictly represented by four of the coefficients of its polynomial expansion:
    c0, c2; c1, c3.
    Other coefficients can be derived from these four coefficients using simple recursive formulas
    Among these coefficients, c0 and c2 directly reflect the geometric features of R_xx(B);
    And c1 and c3 directly reflect the geometric features of R_xy(B).
    
    This model uses c0, c1, c2, c32=c3/c2 as the optimization parameter
    我用R_xx, R_xy各自的多项式系数表达式的方均根构造了残差，并编写了jacobian以加速收敛。
    指定初值的方法非常自然：将多项式截断后使用线性最小二乘法拟合。
    本策略接受的是SI的B, rho_xx, rho_xy数据，接受的载流子浓度和迁移率初值为厘米单位制。
    本策略在计算过程中全程用SI单位制，并提供了计算时使用的参数(c0-c4)与输入输出参数(厘米单位制的n1,n2,mu1,mu2)。之间的转换函数

    同一组c0-c4，当载流子电荷q1,q2同号时对应两组图像完全相同的物理参数解；异号时只有一组物理参数解。

    """
    __slots__ = ("quantites","data_x","H","R_xx","R_xy","q1","q2","_R_xx_std","_R_xy_std")

    def __init__(self, data:Data):
        self.quantites = {
            ("X","magnetic field"): registered_quantities["magnetic field"],
            ("Y", "longitudinal resistivity"): registered_quantities["resistivity"],
            ("Y", "transverse resistivity"): registered_quantities["resistivity"],
            ("PARA", "carrier 1 charge"): registered_quantities["charge"],
            ("PARA", "carrier 2 charge"): registered_quantities["charge"],
        }
        self.H = data.get_array(self.quantites, ("X", "magnetic field"),"T")
        self.data_x = self.H

        self.R_xx = data.get_array(self.quantites, ("Y", "longitudinal resistivity"), "Ohm*cm")
        self.R_xy = data.get_array(self.quantites, ("Y", "transverse resistivity"), "Ohm*cm")

        self.q1 = data.get_array(self.quantites, ("PARA", "carrier 1 charge"), "C")
        self.q2 = data.get_array(self.quantites, ("PARA", "carrier 2 charge"), "C")

        if np.abs(self.q1) < E_CHARGE:
            raise ValueError(f"Carrier 1 charge cannot be smaller than the elementary charge")
        if np.abs(self.q2) < E_CHARGE:
            raise ValueError(f"Carrier 2 charge cannot be smaller than the elementary charge")
        
        self._R_xx_std = np.std(self.R_xx)
        self._R_xy_std = np.std(self.R_xy)


    def forward_factory(self, fit_x):
        n1, n2, mu1, mu2 = fit_x
        def forward(H: NDArray)->Data:
            # For each carrier, we get
            sigma1 = self.q1 * n1 * mu1
            sigma2 = self.q2 * n2 * mu2
            # the relation betwean B-field and H-field is
            #   B = mu0 ( H + M )
            # For most materials, the absolute value of susceptibility |chi| = |M/H| < 1E-4.
            # So below is reasonable in transport measurement:
            B = VAC_PERMEABILITY * H
            denom1 = 1 + (mu1 * B) ** 2
            denom2 = 1 + (mu2 * B) ** 2
            sigma_xx =  sigma1 / denom1 + sigma2 / denom2
            sigma_xy = H * (sigma1 * mu1 / denom1 + sigma2 * mu2 / denom2)

            D = (sigma_xx **2 + sigma_xy **2)
            R_xx = sigma_xx / D
            R_xy = - sigma_xy / D

            return np.vstack((rho_xx, rho_xy))
        
        return forward

    def residual(self,x):
        # Convert log-parameters back to actual parameters (ensure positivity)
        c0,c1,c2,c32 = x
        c3 = c32 * c2
        B = self.H
        rho_fit_denom = 1 + (c32*B)**2
        
        rho_xx_fit = c0 + c2 * B**2 / rho_fit_denom
        rho_xy_fit = c1*B + c3 * B**3 / rho_fit_denom

        res_xx = rho_xx_fit - self.R_xx
        res_xy = rho_xy_fit - self.R_xy

        return np.concatenate([res_xx / self._R_xx_std, res_xy / self._R_xy_std])

    def jacobian(self, x):
        """
        Analytic Jacobian of residual vector w.r.t (c0, c1, c2, c32),
        where c32 = c3/c2.

        Model:
            den    = 1 + (c32 * B)^2
            rho_xx = c0 + c2 * B^2 / den
            rho_xy = c1 * B + (c32 * c2) * B^3 / den

        Residual:
            r_xx = (rho_xx - rho_xx_data) / std_xx
            r_xy = (rho_xy - rho_xy_data) / std_xy

        Returns:
            J with shape (2*N, 4)
        """
        c0, c1, c2, c32 = x
        B = self.H

        std_xx = self._R_xx_std
        std_xy = self._R_xy_std

        # Common terms
        aB = c32 * B
        den = 1.0 + aB**2
        den2 = den**2

        B2 = B**2
        B3 = B**3
        B4 = B**4

        N = B.size
        J = np.zeros((2 * N, 4), dtype=float)

        # ======================
        # rho_xx block
        # rho_xx = c0 + c2 * B^2 / den
        # ======================

        # ∂rho_xx/∂c0 = 1
        J[:N, 0] = 1.0 / std_xx

        # ∂rho_xx/∂c1 = 0
        J[:N, 1] = 0.0

        # ∂rho_xx/∂c2 = B^2 / den
        J[:N, 2] = (B2 / den) / std_xx

        # ∂rho_xx/∂c32:
        # den = 1 + (c32 B)^2 => ∂den/∂c32 = 2*c32*B^2
        # rho_xx = c0 + c2*B^2*den^{-1}
        # => ∂rho_xx/∂c32 = c2*B^2 * (-1)*den^{-2} * ∂den/∂c32
        #                = -2*c2*c32*B^4 / den^2
        J[:N, 3] = (-2.0 * c2 * c32 * B4 / den2) / std_xx

        # ======================
        # rho_xy block
        # rho_xy = c1*B + (c32*c2)*B^3 / den
        # ======================

        # ∂rho_xy/∂c0 = 0
        J[N:, 0] = 0.0

        # ∂rho_xy/∂c1 = B
        J[N:, 1] = (B) / std_xy

        # ∂rho_xy/∂c2:
        # second term = (c32*c2)*B^3/den => derivative w.r.t c2 is c32*B^3/den
        J[N:, 2] = (c32 * B3 / den) / std_xy

        # ∂rho_xy/∂c32:
        # term T = (c32*c2)*B^3 * den^{-1}
        # dT/dc32 = c2*B^3*den^{-1} + (c32*c2)*B^3 * (-1)*den^{-2} * dden/dc32
        #         = c2*B^3/den - (c32*c2)*B^3 * (2*c32*B^2)/den^2
        #         = c2*B^3/den - 2*c2*c32^2*B^5/den^2
        B5 = B**5
        J[N:, 3] = (c2 * B3 / den - 2.0 * c2 * (c32**2) * B5 / den2) / std_xy

        return J


    
    
    def initial_guess(self):
        """
        将多项式进行截断，然后通过无需迭代的线性最小二乘法得到c0-c3初值
        因为约定mu与对应q同号，n*mu*q必须为正0，所以可推出c0和c2也必须为正。
        """
        B = self.H
        rho_xx_base = np.column_stack([
            np.ones_like(B),
            B ** 2
        ])
        coeff_xx, residuals_xx, rank_xx, s_xx = np.linalg.lstsq(rho_xx_base, self.R_xx)
        c0, c2 = coeff_xx

        c0 = 1e-9 if c0 < 0 else c0
        c2 = 1e-12 if c2 < 0 else c2

        rho_xy_base = np.column_stack([
            B,
            - B ** 3
        ])
        coeff_xy, residuals_xy, rank_xy, s_xy = np.linalg.lstsq(rho_xy_base, self.R_xy)
        c1, c3 = coeff_xy

        c32 = c3 / c2

        return c0, c1, c2, c32
        
    
    def from_physics(self,x):
        n1, n2, mu1, mu2 = self.to_SI(*x)
        sigma1 = self.q1 * n1 * mu1
        sigma2 = self.q2 * n2 * mu2
        sigma = sigma1 + sigma2
        s1s2 = sigma1 * sigma2
        
        c0 = 1 / sigma
        c2 = s1s2 * (mu1-mu2)**2 / sigma**3
        c1 = - (sigma1 * mu1 + sigma2 * mu2) / sigma**2
        c3 = s1s2 * (sigma1*mu2+sigma2*mu1) * (mu1-mu2)**2 / sigma**4

        c32 = c3 / c2
        
        return c0, c1, c2, c32

    def to_physics(self,x):
        c0, c1, c2, c32 = x
        q1, q2 = self.q1, self.q2
        x_set = set()

        half_neg_b = 1/2*(c32 - c1 / c0)
        half_sqrt_delta = np.sqrt(half_neg_b**2 + (c2/c0+(c1*c32)/(c0)))
        mu1 = half_neg_b + half_sqrt_delta
        mu2 = half_neg_b - half_sqrt_delta
        f = lambda mu1,mu2: (
            -(c1 / c0**2 + mu2/c0)/(mu1-mu2)/(q1*mu1),
            (c1 / c0**2 + mu1/c0)/(mu1-mu2)/(q2*mu2),
            mu1,
            mu2
        )
        if q1 * q2 > 0:
            x_set.add(
                self.from_SI(*f(mu1,mu2))
            )
            x_set.add(
                self.from_SI(*f(mu2,mu1))
            )
        else:
            if q1 > 0:
                x_set.add(
                    self.from_SI(*f(mu1,mu2))
                )
            else:
                x_set.add(
                    self.from_SI(*f(mu2,mu1))
                )
        return x_set