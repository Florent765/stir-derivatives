import numpy as np
import pandas as pd
import scipy.stats as si
from scipy.optimize import brentq


F0 = 96.16
TARGET_LANDING = 96.35
TARGET_BREAKOUT = 96.41
RISK_FREE = 0.045
LEG_PENALTY = 0.25

DAYS_TO_X5 = 26
DAYS_TO_Z5 = 54


def bachelier_price(F, K, T, r, sigma, otype="call"):
    d = (F - K) / (sigma * np.sqrt(T))
    disc = np.exp(-r * T)
    n_d = si.norm.pdf(d)
    N_d = si.norm.cdf(d)
    if otype == "call":
        return disc * ((F - K) * N_d + sigma * np.sqrt(T) * n_d)
    else:
        return disc * ((K - F) * (1 - N_d) + sigma * np.sqrt(T) * n_d)


def bachelier_greeks(F, K, T, r, sigma, otype="call"):
    d = (F - K) / (sigma * np.sqrt(T))
    disc = np.exp(-r * T)
    n_d = si.norm.pdf(d)
    N_d = si.norm.cdf(d)
    delta = disc * N_d if otype == "call" else disc * (N_d - 1)
    gamma = (disc * n_d) / (sigma * np.sqrt(T))
    return delta, gamma


def calibrate_volatility():
    target_price = 1.25 / 100.0
    T = DAYS_TO_X5 / 365.0

    def objective(sigma):
        c1 = bachelier_price(F0, 96.25, T, RISK_FREE, sigma, "call")
        c2 = bachelier_price(F0, 96.35, T, RISK_FREE, sigma, "call")
        return (c1 - c2) - target_price

    return brentq(objective, 0.10, 1.00)


class Strategy:
    def __init__(self, name, expiry_code, T_days, legs):
        self.name = name
        self.expiry_code = expiry_code
        self.T = T_days / 365.0
        self.days = T_days
        self.legs = legs

    def evaluate(self, model):
        cost_mid = 0
        gamma_tgt = 0

        for K, otype, qty in self.legs:
            p = bachelier_price(model.F0, K, self.T, RISK_FREE, model.sigma, otype)
            _, g = bachelier_greeks(
                TARGET_LANDING, K, self.T, RISK_FREE, model.sigma, otype
            )
            cost_mid += p * qty
            gamma_tgt += g * qty

        mid_ticks = max(cost_mid * 100, 0.50)
        total_cost = mid_ticks + (len(self.legs) * LEG_PENALTY)

        # Scenario A: Cut happens EARLY (Nov) - 20% Prob
        # Scenario B: Cut happens LATE (Dec) - 80% Prob

        payoff_A = 0  # Early Cut Payoff
        payoff_B = 0  # Late Cut Payoff

        for K, otype, qty in self.legs:
            val_at_cut = (
                max(TARGET_BREAKOUT - K, 0)
                if otype == "call"
                else max(K - TARGET_BREAKOUT, 0)
            )
            payoff_A += val_at_cut * qty
            payoff_B += val_at_cut * qty

        if self.expiry_code == "X5":
            payoff_B = 0  # Dies before December meeting

        # EV
        prob_early = 0.20
        prob_late = 0.80
        expected_payoff = (payoff_A * prob_early) + (payoff_B * prob_late)

        ev_ticks = expected_payoff * 100
        score = ev_ticks / total_cost if total_cost > 0 else 0

        # Dynamic Check
        is_dynamic = "YES" if gamma_tgt > 0 else "NO"

        return {
            "Name": self.name,
            "Expiry": self.expiry_code,
            "Cost": round(total_cost, 2),
            "EV (Weighted)": round(ev_ticks, 2),
            "Score (EV/Cost)": round(score, 2),
            "Dynamic": is_dynamic,
        }


# Generate universe
def generate_strategies():
    strats = []
    strikes = np.arange(96.15, 96.45, 0.05)

    for expiry, days in [("X5", DAYS_TO_X5), ("Z5", DAYS_TO_Z5)]:
        for k in strikes:
            k = round(k, 2)
            # Call Spreads
            for w in [0.10, 0.15, 0.20, 0.25]:
                k_up = round(k + w, 2)
                legs = [(k, "call", 1), (k_up, "call", -1)]
                strats.append(Strategy(f"CS {k}/{k_up}", expiry, days, legs))

            # Ratio Backspreads (1x2)
            for w in [0.10, 0.15]:
                k_up = round(k + w, 2)
                legs = [(k, "call", -1), (k_up, "call", 2)]
                strats.append(Strategy(f"Ratio 1x2 {k}/{k_up}", expiry, days, legs))

    return strats


class MarketModel:
    def __init__(self, F0):
        self.F0 = F0
        self.sigma = calibrate_volatility()
        print(f"CALIBRATED VOL: {self.sigma:.4f} ({self.sigma * 100:.1f} bps)")


model = MarketModel(F0)
universe = generate_strategies()
df = pd.DataFrame([s.evaluate(model) for s in universe])
df_sorted = df.sort_values("Score (EV/Cost)", ascending=False)

print("\n--- FINAL STRATEGIC RANKING (Weighted Scenarios) ---")
print(df_sorted.head(10))
