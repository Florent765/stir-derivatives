import numpy as np
import pandas as pd
import scipy.stats as si
from scipy.optimize import minimize


F0 = 98.11  # ERU6 Future
T_EXP = 11.0 / 12.0  # 11 Months
RISK_FREE = 0.03  # 3% Rate

# We calibrate to these exact market observations from the text.
# ATM Put Fly (98.18 / 98.06 / 97.93) ~ 0.75 ticks (Mid of 0.50/1.00)
# Downside Put Fly (98.06 / 97.93 / 97.81) ~ 4.50 ticks (Mid of 4.25/4.75)
TARGET_ATM_FLY = 0.75 / 100.0
TARGET_SKEW_FLY = 4.50 / 100.0


def bachelier_price(F, K, T, r, sigma, otype="put"):
    d = (F - K) / (sigma * np.sqrt(T))
    disc = np.exp(-r * T)
    n_d = si.norm.pdf(d)
    N_d = si.norm.cdf(d)

    if otype == "call":
        return disc * ((F - K) * N_d + sigma * np.sqrt(T) * n_d)
    else:
        return disc * ((K - F) * (1 - N_d) + sigma * np.sqrt(T) * n_d)


def sabr_normal_vol(K, F, T, alpha, beta, rho, nu):
    if K <= 0 or F <= 0:
        return alpha  # Fallback

    if abs(F - K) < 1e-5:
        return alpha * (1 + ((2 - 3 * rho**2) / 24 * nu**2) * T)

    z = (nu / alpha) * (F - K)
    discriminant = 1 - 2 * rho * z + z**2
    if discriminant <= 0:
        discriminant = 1e-5

    chi_z = np.log((np.sqrt(discriminant) + z - rho) / (1 - rho))

    if abs(chi_z) < 1e-5:
        return alpha

    vol = alpha * (z / chi_z) * (1 + ((2 - 3 * rho**2) / 24 * nu**2) * T)
    return max(1e-4, vol)  # Floor at 1bp to prevent zeroes


class MarketCalibrator:
    def __init__(self):
        self.params = self.calibrate()

    def price_fly(self, strikes, alpha, rho, nu):
        k1, k2, k3 = strikes
        p1 = self._p(k1, alpha, rho, nu)
        p2 = self._p(k2, alpha, rho, nu)
        p3 = self._p(k3, alpha, rho, nu)
        return p1 - 2 * p2 + p3

    def _p(self, K, alpha, rho, nu):
        vol = sabr_normal_vol(K, F0, T_EXP, alpha, 0.0, rho, nu)
        return bachelier_price(F0, K, T_EXP, RISK_FREE, vol, "put")

    def calibrate(self):
        print("STARTING CALIBRATION")

        def objective(params):
            alpha, rho, nu = params
            if not (0.0001 < alpha < 0.5):
                return 1e9
            if not (-0.999 < rho < 0.999):
                return 1e9
            if not (0.01 < nu < 5.0):
                return 1e9

            strikes_atm = [98.185, 98.06, 97.935]
            model_atm = self.price_fly(strikes_atm, alpha, rho, nu)

            strikes_skew = [98.06, 97.935, 97.81]
            model_skew = self.price_fly(strikes_skew, alpha, rho, nu)

            # Squared Error (Weighted to prioritize the skew fit)
            error = ((model_atm - TARGET_ATM_FLY) * 1000) ** 2 + (
                (model_skew - TARGET_SKEW_FLY) * 10
            ) ** 2
            return error

        # Initial Guess: Normal-ish markets (Alpha=20bps, Rho=-0.5, Nu=0.5)
        x0 = [0.20, -0.5, 0.5]

        res = minimize(objective, x0, method="Nelder-Mead", tol=1e-4)

        print(f"converged: {res.success}")
        print(f"Alpha: {res.x[0]:.4f} | Rho: {res.x[1]:.4f} | Nu: {res.x[2]:.4f}")
        return res.x


class Strategy:
    def __init__(self, name, legs):
        self.name = name
        self.legs = legs

    def evaluate(self, calibrator, view_pdf, spot_grid):
        alpha, rho, nu = calibrator.params

        cost_agg = 0
        payoff_grid = np.zeros_like(spot_grid)

        # Calculate Price & Payoff
        for K, otype, qty in self.legs:
            vol = sabr_normal_vol(K, F0, T_EXP, alpha, 0.0, rho, nu)
            p = bachelier_price(F0, K, T_EXP, RISK_FREE, vol, otype)
            cost_agg += p * qty

            if otype == "put":
                payoff_grid += np.maximum(K - spot_grid, 0) * qty
            else:
                payoff_grid += np.maximum(spot_grid - K, 0) * qty

        # We enforce a minimum Bid-Ask spread cost of 0.50 ticks for complex structures.
        calc_ticks = cost_agg * 100
        real_cost = max(calc_ticks, 0.50)

        # PnL & EV
        pnl_grid = (payoff_grid * 100) - real_cost
        ev = np.sum(pnl_grid * view_pdf)

        pos_idxs = np.where(pnl_grid > 0)[0]
        width = 0
        if len(pos_idxs) > 0:
            width = (spot_grid[pos_idxs[-1]] - spot_grid[pos_idxs[0]]) * 100

        return {
            "Name": self.name,
            "Cost (Calc)": round(calc_ticks, 2),
            "Cost (Real)": round(real_cost, 2),
            "EV": round(ev, 2),
            "Width": round(width, 1),
            "ROI": round(ev / real_cost, 2) if real_cost > 0 else 0,
        }


spot_grid = np.linspace(97.50, 98.50, 2000)
# Box View: 97.98 to 98.14 (Targeting 98.06 +/- 0.08)
view_pdf = np.zeros_like(spot_grid)
mask = (spot_grid >= 97.98) & (spot_grid <= 98.14)
view_pdf[mask] = 1.0
view_pdf /= np.sum(view_pdf)

calib = MarketCalibrator()

# Generate Strategies
strategies = []
centers = np.arange(97.90, 98.12, 0.01)

for center in centers:
    center = round(center, 2)

    for w in [0.10, 0.125, 0.15]:
        # Fly
        legs_fly = [(center + w, "put", 1), (center, "put", -2), (center - w, "put", 1)]
        strategies.append(Strategy(f"Fly {center} (w={w})", legs_fly))

        # Condor (Body = 4 ticks)
        body = 0.04
        k1, k2 = center + body / 2, center - body / 2
        legs_condor = [
            (k1 + w, "put", 1),
            (k1, "put", -1),
            (k2, "put", -1),
            (k2 - w, "put", 1),
        ]
        strategies.append(Strategy(f"Condor {center} (w={w})", legs_condor))

df = pd.DataFrame([s.evaluate(calib, view_pdf, spot_grid) for s in strategies])
df_sorted = df.sort_values("EV", ascending=False)

print("\nRESULTS (With Liquidity Floor)")
print(df_sorted.head(5))

winner = df_sorted.iloc[0]
print(f"\nWINNER: {winner['Name']} with EV: {winner['EV']}")
