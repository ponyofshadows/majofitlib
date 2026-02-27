"""
Eeduction
------
Review of basic concepts in electromagnetism (in SI).

Some particles can carray charge, which makes these particles acted upon by Lorentz force
when they are placed in two kinds of fields: electric field and magnetic field.
    F_(Lorentz) = qE + q v xx B
The change in the amount of charge satisfies the continuity equation.
To express this, we can further define the charge density as the amount of charge in unit volume,
and the electric current density as the amount of charge passing through a unit area element within a unit of time.
    rho = Q // Delta V
    J = Delta Q // (Delta S * Delta t)
continuity equation:
    drho/dt + nabla*J = 0
electric field and magnetic field satisfy Maxwell's equations:
    nabla*E = 1/(epsilon_0) rho
    nabla*B = 0
    nabla xx E = - delB/delt
    nabla xx B = mu0 J + mu_0 epsilon_0 delE/delt
Based on these, considering the speed light in vaccum, we can get:
    mu_0 epsilon_0 = 1/ c^2
The vacuum permeability mu0 is a measured quantity.
Substituting mu_0 into the above relationship yields the vaccum permitivity epsilon_0.

To simplify the Maxwell's equations to a no-constant form, we define electric displacement and magnetic intensity:
    D = mu0 E
    H = 1/(epsilon_0) B
Now we get:
    nabla*D = rho
    nabla*B = 0
    nabla xx E = - delB/delt
    nabla xx H = J + delD/delt
In matter, rho can be devided into free charge density rho_f and bound charge density;
J can be devided into free currents J_f, bound current J_b, and polarization current J_p.
    rho = rho_f + rho_b
    J = J_f + J_b + J_p
We define polarization P and magnetization M:
    rho_b -= - nabla*P  (NOTE THE NEGATIVE SIGN)
    J_p = delP/delt
    J_b = nabla xx M
Then we let:
    D = mu0 E + P
    H = 1/(epsilon_0) B - M
Now we get the Maxwell's equations in materials (or macroscopic Maxwell's equation):
    nabla*D = rho_f
    nabla*B = 0
    nabla xx E = - delB/delt
    nabla xx H = J_f + delD/delt
The following linar reactive expressions are commonly used:
    D = mu E
    H = epsilon B
Mu and epsilon can be scalars or tensors, depending on whether the material is isotropic
------
The principle under the same system of units
    1. Every physical law must be dimensionally consistent: [LHS] = [RHS]
    2. Conversion constants are just bookkeeping undet certain unit system, multiplicity of units is human-centered.
The principle under different systems of units:
    Suppose we are considering (n+1) physical quantities.
    Among them, the conversion coefficient of q_1,q_2,...,q_n in thete unit system is know.
    To get the unknown conversion coefficient of q_x between system A and B, we can find a equation:
        F(q_1,q_2,...,q_n; q_x) = 0
    In two unit system:
        F_A(q_1A,q_2A,...,q_nA; q_xA) = 0
        F_B(q_1B,q_2B,...,q_B; q_xB) = 0
    Then substitute the different values of the same quantities  q_1, q_2, ..., q_n into the above equations without units,
    we obtain the values q_xA and q_xB.
    q_1, q_2, ..., q_n remain same quantities in two equations, which means q_xA and q_xB are the same when we write the values with units:
        q_xA [Unit of q_xA] = q_xB [Unit of q_xB]
    So
        [Unit of q_xB] // [Unit of q_xA] = q_xA // q_xB
"""
from .base import unit_register
from .constants import VAC_PERMEABILITY, VAC_PERMITIVITY
import numpy as np

# Derived from Coulomb's law
#       SI: F_(Coulomb) = 1/(4*pi*epslion_0) (q1*q2)/r^2
# Gaussian: F_(Coulomb) = (q1*q2)/r^2
## 1 kg*m*s^(-2) = 1E5 g*cm*s^(-2)
## 1 m = 1E2 cm
## Let q1 = q2 = q
unit_register("charge", "C", multiplier=1, ascii_name="C")
unit_register("charge", "statC", multiplier=np.sqrt(4 * np.pi * VAC_PERMITIVITY * 1E-9), ascii_name="statC")


# Derived from Lorentz's formula
#       SI: F_(Lorntz) = qE + qvB
# Gaussian: F_(Lorentz) = qE + 1/c*qvB
## 1 kg*m*s^(-2) = 1E5 g*cm*s^(-2)
## 1 m*s^(-1) = 1E2 cm*s^(-1)
unit_register("B-field", "T", multiplier=1, ascii_name="T")
unit_register("B-field", "G", multiplier=np.sqrt(VAC_PERMEABILITY*0.1/(4*np.pi)), ascii_name="G")
unit_register("B-field", "Gs", multiplier=np.sqrt(VAC_PERMEABILITY*0.1/(4*np.pi)), ascii_name="Gs")
#       SI: B = mu0(H + M)
# Gaussian: B = H + M
unit_register("H-field", "A/m", multiplier=1, ascii_name="A/m")
unit_register("H-field", "Oe", multiplier=np.sqrt(0.1/(4*np.pi*VAC_PERMEABILITY)), ascii_name="Oe")
# M is often very small. At this time, H can be directly used as B
unit_register("B-field", "A/m", multiplier=VAC_PERMEABILITY, ascii_name="A/m")
unit_register("B-field", "Oe", multiplier=np.sqrt(VAC_PERMEABILITY*0.1/(4*np.pi)), ascii_name="Oe")


unit_register("resistivity", "Ohm*m", multiplier=1, ascii_name="Ω·m")
unit_register("resistivity", "Ohm*cm", multiplier=1E-2, ascii_name="Ω·cm")
unit_register("resistivity", "mOhm*cm", multiplier=1E-5, ascii_name="mΩ·cm")
unit_register("resistivity", "muOhm*cm", multiplier=1E-8,ascii_name="μΩ·m")


unit_register("carrier concentration", "m^-3", multiplier=1,ascii_name="m⁻³")
unit_register("carrier concentration", "cm^-3", multiplier=1E-6,ascii_name="cm⁻³")

unit_register("mobility", "m^2/(V*s)", multiplier=1,ascii_name="m²/(V·s)")
unit_register("mobility", "cm^2/(V*s)", multiplier=1E-4,ascii_name="cm²/(V·s)")