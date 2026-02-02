import numpy as np
import matplotlib.pyplot as plt

from base import ModelStrategy
from models.polynomial_two_carrier import PolynomialTwoCarrierStrategy
from optimize import least_squares_optimizer

def main():

    # Load data from CSV files (assume comma or whitespace delimiter, no header)
    data_xx = np.loadtxt("./data/80KMR.txt", delimiter='\t')
    data_xy = np.loadtxt("./data/80Khall.txt", delimiter='\t')
    B, rho_xx_raw = data_xx[:, 0], data_xx[:, 1]
    rho_xy_raw = data_xy[:, 1]

    rho_xx = (rho_xx_raw + rho_xx_raw[::-1])/2
    rho_xy = (rho_xy_raw - rho_xy_raw[::-1])/2


    # Init the Model
    e = 1.602176634e-19
    model_strategy = PolynomialTwoCarrierStrategy(B, rho_xx, rho_xy, e, -e)

    #x0 = model_strategy.initial_guess()
    x0 = model_strategy.from_physics((1e16, 1e14, 1e06, -1e03))
    print(f"initial guess c: {x0}")
    print(f"residual of initial guess c")
    residual_diagnose(model_strategy,x0)
    result = least_squares_optimizer(model_strategy, 
                                x0=x0,
                                lower_bound=(-np.inf,-np.inf,-np.inf,-np.inf),
                                upper_bound=(np.inf,np.inf,np.inf,np.inf)
                            )

    # Convert to desired output units
    c0, c1, c2, c32 = result.x
    print(f"residual of optimical result:")
    residual_diagnose(model_strategy,result.x)
    print(f"{result.message}. Optimal parameters:")
    print(f"  c0 = {c0:.6g}")
    print(f"  c1 = {c1:.6g}")
    print(f"  c2 = {c2:.6g}")
    print(f"  c32 = {c32:.6g}")
    print("physics parameters:")
    for i,sol in enumerate(result.set):
        print(f"######## solution {i} ########")
        n_h_opt, n_e_opt, mu_h_opt, mu_e_opt = sol

        print(f"  n_h = {n_h_opt:.6g} cm^-3")
        print(f"  n_e = {n_e_opt:.6g} cm^-3")
        print(f"  mu_h = {mu_h_opt:.6g} cm^2/V·s")
        print(f"  mu_e = {mu_e_opt:.6g} cm^2/V·s")
    # Print final cost or goodness of fit information
    print(f"Final cost (sum of squared residuals) = {result.cost:.6e},   iterations = {result.nfev}")
    cond_J = np.linalg.cond(result.jac)
    print("Jacobian condition number =", cond_J)
    print("Result Report:")
    print(result)

    # Plot
    model = model_strategy.model(sol)
    fit_rho = model(B)
    fit_rho_xx, fit_rho_xy = fit_rho[0,:], fit_rho[1,:]
    print(f"原数据rho_xy的a1*x+a3*x^3+a5*x^5拟合分量：")
    print(fit_poly_coeffs(B,rho_xy,(1,3,5)))
    print(f"拟合数据fit_rho_xy的a1*x+a3*x^3+a5*x^5拟合分量：")
    print(fit_poly_coeffs(B,fit_rho_xy,(1,3,5)))
 

    #c0, c1, c2, c3 = model_strategy.initial_guess()
    c0, c1, c2, c32 = x0
    rho_fit_denom = 1 + (c32*B)**2
    rho_xx_guess = c0 + c2 * B**2 / rho_fit_denom
    rho_xy_guess = c1*B + c32*c2 * B**3 / rho_fit_denom

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10,4))
    ax1.set_title("80K MR Fit")
    ax1.set_xlabel("B (T)")
    ax1.set_ylabel("ρ_xx (Ω*m)")
    ax1.plot(B, rho_xx_raw, linestyle="-.", color="black", marker="s",label="rho_xx Data")
    #ax1.plot(B, rho_xx_guess, linestyle="-", color="blue", marker="^", label="rho_xx Initial")
    ax1.plot(B, fit_rho_xx, linestyle="--", color="red", marker="o", label="rho_xx Fit")
    ax1.legend()

    ax2.set_title("80K Hall Fit")
    ax2.set_xlabel("B (T)")
    ax2.set_ylabel("ρ_xy (Ω*m)")
    ax2.plot(B, rho_xy_raw, linestyle="-.", color="black", marker="s", label="rho_xy Data")
    #ax2.plot(B, rho_xy_guess, linestyle="-", color="blue", marker="^", label="rho_xy Initial")
    ax2.plot(B, fit_rho_xy, linestyle="--", color="red", marker="o", label="rho_xy Fit")
    ax2.legend()
    plt.show()

    if not result.success:
        print("Optimizer did not converge:", result.message)


def residual_diagnose(model_strategy, x):
    # split residual into xx and xy blocks and locate worst points.
    r = model_strategy.residual(x)
    N = model_strategy.B.shape[0]
    r_xx = r[:N]
    r_xy = r[N:]

    rmse_xx = np.sqrt(np.mean(r_xx**2))
    rmse_xy = np.sqrt(np.mean(r_xy**2))
    print("Normalized RMSE:")
    print("  rmse_xx =", rmse_xx)
    print("  rmse_xy =", rmse_xy)

    kx = int(np.argmax(np.abs(r_xx)))
    ky = int(np.argmax(np.abs(r_xy)))
    print("Worst points:")
    print("  xx: B =", model_strategy.B[kx], "r =", r_xx[kx])
    print("  xy: B =", model_strategy.B[ky], "r =", r_xy[ky])

import numpy as np

def fit_poly_coeffs(B: np.ndarray, y: np.ndarray, orders:tuple[int]):
    """
    Fit y(B) ≈ Σ a_k * B^k over given orders (least squares).
    Returns dict {k: a_k}.
    """
    B = np.asarray(B, float)
    y = np.asarray(y, float)

    A = np.column_stack([B**k for k in orders])
    coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
    return {k: float(a) for k, a in zip(orders, coeffs)}

if __name__ == "__main__":
    main()
