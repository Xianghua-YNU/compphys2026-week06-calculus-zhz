import math


def debye_integrand(x: float) -> float:
    if abs(x) < 1e-12:
        return 0.0
    ex = math.exp(x)
    return (x**4) * ex / ((ex - 1.0) ** 2)


def trapezoid_composite(f, a: float, b: float, n: int) -> float:
    # TODO B1: 实现复合梯形积分
    h = (b - a) / n
    result = 0.5 * (f(a) + f(b))  # 端点贡献
    
    for i in range(1, n):
        x_i = a + i * h
        result += f(x_i)
    
    return result * h

def simpson_composite(f, a: float, b: float, n: int) -> float:
    # TODO B2: 实现复合 Simpson 积分，并检查 n 为偶数
    if n % 2 != 0:
        raise ValueError(f"Simpson 积分要求 n 为偶数，当前 n={n}")
    
    h = (b - a) / n
    result = f(a) + f(b)  # 端点
    
    # 奇数节点 (系数 4)
    for i in range(1, n, 2):
        x_i = a + i * h
        result += 4 * f(x_i)
    
    # 偶数节点 (系数 2)
    for i in range(2, n, 2):
        x_i = a + i * h
        result += 2 * f(x_i)
    
    return result * h / 3

def debye_integral(T: float, theta_d: float = 428.0, method: str = "simpson", n: int = 200) -> float:
    # TODO B3: 计算 Debye 积分 I(theta_d/T)
    if T <= 0:
        raise ValueError("温度 T 必须为正数")
    
    y = theta_d / T  # 积分上限
    
    # 选择积分方法
    if method == "trapezoid":
        return trapezoid_composite(debye_integrand, 0.0, y, n)
    elif method == "simpson":
        return simpson_composite(debye_integrand, 0.0, y, n)
    else:
        raise ValueError(f"未知方法: {method}，请选择 'trapezoid' 或 'simpson'")

def reference_integral(y: float) -> float:
    """
    使用高精度参考值（通过增大 n 的 Simpson 方法获得）
    Debye 积分的解析性质：当 y→∞ 时，I(∞) = 4π^4/15 ≈ 25.9757
    """
    # 使用非常大的 n 获得参考值
    return simpson_composite(debye_integrand, 0.0, y, 100000)


def compare_methods():
    """比较梯形法和 Simpson 法的精度"""
    print("=" * 60)
    print("Debye 积分数值方法比较")
    print("=" * 60)
    
    test_cases = [
        (50, 428),   # 低温 T=50K, y=8.56
        (100, 428),  # 中温 T=100K, y=4.28
        (300, 428),  # 室温 T=300K, y=1.43
        (1000, 428), # 高温 T=1000K, y=0.428
    ]
    
    n_values = [20, 50, 100, 200, 500]
    
    for T, theta_d in test_cases:
        y = theta_d / T
        ref = reference_integral(y)
        
        print(f"\n温度 T = {T}K, Debye 温度 θ_D = {theta_d}K")
        print(f"积分上限 y = θ_D/T = {y:.4f}")
        print(f"参考值 (n=100000 Simpson): I(y) = {ref:.10f}")
        print("-" * 50)
        print(f"{'n':>6} | {'梯形法':>15} | {'误差':>12} | {'Simpson':>15} | {'误差':>12}")
        print("-" * 50)
        
        for n in n_values:
            # 确保 n 为偶数用于 Simpson
            n_simpson = n if n % 2 == 0 else n + 1
            
            trap_result = debye_integral(T, theta_d, "trapezoid", n)
            trap_error = abs(trap_result - ref)
            
            simp_result = debye_integral(T, theta_d, "simpson", n_simpson)
            simp_error = abs(simp_result - ref)
            
            print(f"{n:>6} | {trap_result:>15.8f} | {trap_error:>12.2e} | "
                  f"{simp_result:>15.8f} | {simp_error:>12.2e}")
    
    # 高温极限验证
    print("\n" + "=" * 60)
    print("高温极限验证 (T → ∞, y → 0)")
    print("=" * 60)
    print("理论极限: I(∞) = 4π⁴/15 =", 4 * math.pi**4 / 15)
    
    for T in [1000, 2000, 5000, 10000]:
        y = 428 / T
        I_y = debye_integral(T, 428, "simpson", 1000)
        print(f"T = {T:>5}K, y = {y:.4f}, I(y) = {I_y:.6f}")


if __name__ == "__main__":
    compare_methods()