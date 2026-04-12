import numpy as np


def rate_3alpha(T: float) -> float:
    T8 = T / 1.0e8
    return 5.09e11 * (T8 ** (-3.0)) * np.exp(-44.027 / T8)


def finite_diff_dq_dT(T0: float, h: float = 1e-8) -> float:
    h_abs = h * T0
    return (rate_3alpha(T0 + h_abs) - rate_3alpha(T0)) / h_abs
    # TODO A1: 使用前向差分实现 dq/dT
    raise NotImplementedError("TODO A1")


def sensitivity_nu(T0: float, h: float = 1e-8) -> float:
    # TODO A2: 根据 nu = (T/q) * dq/dT 计算温度敏感性指数
    q = rate_3alpha(T0)
    dq_dT = finite_diff_dq_dT(T0, h)
    return (T0 / q) * dq_dT
    raise NotImplementedError("TODO A2")


def nu_table(T_values, h: float = 1e-8):
    # TODO A3: 返回 [(T, nu(T)), ...]
    return [(T, sensitivity_nu(T, h)) for T in T_values]
    raise NotImplementedError("TODO A3")
if __name__ == "__main__":
    T_values = [1.0e8, 2.5e8, 5.0e8, 1.0e9, 2.5e9, 5.0e9]
    results = nu_table(T_values)
    for T, nu in results:
        print(f"{T:.3e} K : nu = {nu:.2f}")