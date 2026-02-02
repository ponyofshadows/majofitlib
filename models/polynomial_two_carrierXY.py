from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray


E_CHARGE = 1.602176634e-19

@dataclass(slots=True)
class PolynomialTwoCarrierXYStrategy():
    """
    根据10.1063/5.0248998提供的多项式方法，双载流子模型能用展开为多项式时的4个系数严格表示。
    前四个多项式系数直接反映了图像特征。c0,c2对应R_(xx)(B)图像的纵截距和开口狭窄度；c1,c3对应R_{xy}(B)图像的线性斜率与三次项在B=1T附近的起伏强度。
    在物理意义上，c0对应单载流子模型的R_(sh)，c1对应单载流子模型的R_H
    更高次项的系数满足一个简单的递推关系：c_(j>=2)= c_3^(j-2)/c_2^(j-3)，这可以理解成：双载流子模型中，磁阻和霍尔电阻率只在四次项及以上存在统计关联，而且这种关联与线性项无关。
    
    我用c0,c1,c2,c32=c3/c2作为拟合参数
    我用R_xx, R_xy各自的多项式系数表达式的方均根构造了残差，并编写了jacobian以加速收敛。
    指定初值的方法非常自然：将多项式截断后使用线性最小二乘法拟合。
    本策略接受的是SI的B, rho_xx, rho_xy数据，接受的载流子浓度和迁移率初值为厘米单位制。
    本策略在计算过程中全程用SI单位制，并提供了计算时使用的参数(c0-c4)与输入输出参数(厘米单位制的n1,n2,mu1,mu2)。之间的转换函数

    同一组c0-c4，当载流子电荷q1,q2同号时对应两组图像完全相同的物理参数解；异号时只有一组物理参数解。
    ---
    但是，当磁阻的曲线不理想时，可以对磁阻特性装聋作哑，只用零场电组+霍尔曲线(XY)来拟合。
    """
    B: NDArray[np.floating]
    R_sh: np.floating
    rho_xy: NDArray[np.floating]
    q1: float
    q2: float


    def __post_init__(self):
        if np.abs(self.q1) < E_CHARGE:
            raise ValueError(f"q1 cannot be smaller than the elementary charge")
        if np.abs(self.q2) < E_CHARGE:
            raise ValueError(f"q2 cannot be smaller than the elementary charge")
        
        if self.R_sh == 0:
            raise ValueError("TwoCarrierModel is not suit for superconductor!")
        elif self.R_sh < 0:
            raise ValueError("R_sh should be positive.")
        
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
            rho_xy = - sigma_xy / D

            return rho_xy
        
        return two_carrier_model

    def residual(self,x):
        # Convert log-parameters back to actual parameters (ensure positivity)
        c1,c2,c32 = x
        c3 = c32 * c2
        B = self.B
        rho_fit_denom = 1 + (c32*B)**2
        
        rho_xy_fit = c1*B + c3 * B**3 / rho_fit_denom

        res_xy = rho_xy_fit - self.rho_xy

        return  res_xy

    def jacobian(self, x):
        """
        Analytic Jacobian of residual vector w.r.t (c0, c1, c2, c32),
        where c32 = c3/c2.

        Model:
            den    = 1 + (c32 * B)^2
            rho_xx = c0 + c2 * B^2 / den  <- not used
            rho_xy = c1 * B + (c32 * c2) * B^3 / den

        Residual:
            r_xx = (rho_xx - rho_xx_data) <- not used
            r_xy = (rho_xy - rho_xy_data)

        Returns:
            J with shape (2*N, 4)
        """
        c1, c2, c32 = x
        B = self.B
        ...

        return J


    
    
    def initial_guess(self):
        """
        将多项式进行截断，然后通过无需迭代的线性最小二乘法得到c0-c3初值
        因为约定mu与对应q同号，n*mu*q必须为正0，所以可推出c0和c2也必须为正。
        """
        B = self.B
        rho_xx_base = np.column_stack([
            np.ones_like(B),
            B ** 2
        ])
        coeff_xx, residuals_xx, rank_xx, s_xx = np.linalg.lstsq(rho_xx_base, self.rho_xx)
        c0, c2 = coeff_xx

        c0 = 1e-9 if c0 < 0 else c0
        c2 = 1e-12 if c2 < 0 else c2

        rho_xy_base = np.column_stack([
            B,
            - B ** 3
        ])
        coeff_xy, residuals_xy, rank_xy, s_xy = np.linalg.lstsq(rho_xy_base, self.rho_xy)
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


    @staticmethod
    def from_SI(n1SI,n2SI,mu1SI,mu2SI):
        return n1SI*1e-6, n2SI*1e-6, mu1SI*1e4, mu2SI*1e4 

    @staticmethod
    def to_SI(n1cm,n2cm,mu1cm,mu2cm):
        return n1cm*1e6, n2cm*1e6, mu1cm*1e-4, mu2cm*1e-4 

