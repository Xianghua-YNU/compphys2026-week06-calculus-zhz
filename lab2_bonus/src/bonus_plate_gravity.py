import numpy as np
from scipy.special import roots_legendre

G = 6.674e-11


def gauss_legendre_2d(func, ax: float, bx: float, ay: float, by: float, n: int = 40) -> float:
    """
    使用二维高斯-勒让德积分计算二重积分
    ∫_{ax}^{bx} ∫_{ay}^{by} func(x, y) dy dx
    
    参数:
    - func: 被积函数，接受(x, y)返回函数值
    - ax, bx: x方向的积分上下限
    - ay, by: y方向的积分上下限
    - n: 高斯-勒让德积分点的数量
    
    返回:
    - 积分结果
    """
    # 获取高斯-勒让德积分点和权重
    xi, wi = roots_legendre(n)
    
    # 坐标变换：从[-1, 1]变换到[ax, bx]和[ay, by]
    x_mid = 0.5 * (bx + ax)
    x_half = 0.5 * (bx - ax)
    
    y_mid = 0.5 * (by + ay)
    y_half = 0.5 * (by - ay)
    
    # 计算二维积分
    result = 0.0
    
    for i in range(n):
        # x方向的积分点和权重
        x_i = x_mid + x_half * xi[i]
        wx_i = wi[i]
        
        for j in range(n):
            # y方向的积分点和权重
            y_j = y_mid + y_half * xi[j]
            wy_j = wi[j]
            
            # 计算函数值并加权
            result += wx_i * wy_j * func(x_i, y_j)
    
    # 乘以Jacobian因子
    return result * x_half * y_half


def plate_force_z(z: float, L: float = 10.0, M_plate: float = 1.0e4, m_particle: float = 1.0, n: int = 40) -> float:
    """
    计算方形薄板中心正上方z位置的Fz（垂直方向的引力）
    
    参数:
    - z: 质点距离板中心的垂直距离
    - L: 方形薄板的边长
    - M_plate: 薄板的总质量
    - m_particle: 质点的质量
    - n: 高斯积分点的数量
    
    返回:
    - 垂直方向的引力Fz
    """
    if z <= 0:
        raise ValueError("z必须是正数（质点在薄板上方）")
    
    # 面密度
    sigma = M_plate / (L * L)
    
    def integrand(x, y):
        """
        被积函数：方形薄板上一点(x, y)对质点的垂直引力分量
        质点位于(0, 0, z)，薄板在z=0平面
        """
        r_squared = x*x + y*y + z*z
        r = np.sqrt(r_squared)
        
        # 引力公式：dFz = G * (sigma * dx * dy) * m_particle * z / r^3
        # 由于sigma是面密度，dx*dy是面积元
        return G * sigma * m_particle * z / (r * r_squared)
    
    # 积分区域：[-L/2, L/2] × [-L/2, L/2]
    half_L = L / 2.0
    ax, bx = -half_L, half_L
    ay, by = -half_L, half_L
    
    # 计算二重积分
    Fz = gauss_legendre_2d(integrand, ax, bx, ay, by, n)
    
    return Fz


def force_curve(z_values, L: float = 10.0, M_plate: float = 1.0e4, m_particle: float = 1.0, n: int = 40):
    """
    返回z_values对应的Fz数组
    
    参数:
    - z_values: 距离值的数组
    - L: 方形薄板的边长
    - M_plate: 薄板的总质量
    - m_particle: 质点的质量
    - n: 高斯积分点的数量
    
    返回:
    - Fz数组，对应每个z值的引力
    """
    Fz_values = np.zeros_like(z_values)
    
    for i, z in enumerate(z_values):
        Fz_values[i] = plate_force_z(z, L, M_plate, m_particle, n)
    
    return Fz_values


# 测试代码
if __name__ == "__main__":
    # 测试1: 计算单个点的引力
    z_test = 5.0
    M_plate = 1.0e4  # 定义M_plate变量
    m_particle = 1.0
    
    Fz_test = plate_force_z(z_test, L=10.0, M_plate=M_plate, m_particle=m_particle, n=20)
    print(f"在z={z_test}m处，Fz = {Fz_test:.2e} N")
    
    # 测试2: 计算引力曲线
    z_array = np.linspace(1, 20, 10)
    Fz_array = force_curve(z_array, L=10.0, M_plate=M_plate, m_particle=m_particle, n=20)
    
    print("\n引力曲线:")
    for z, Fz in zip(z_array, Fz_array):
        print(f"z={z:.1f}m: Fz={Fz:.2e} N")
    
    # 验证：当z远大于L时，应近似于点质量引力
    z_far = 1000.0
    Fz_far = plate_force_z(z_far, L=10.0, M_plate=M_plate, m_particle=m_particle, n=20)
    Fz_point = G * M_plate * m_particle / (z_far * z_far)  # 点质量近似
    print(f"\n远场验证:")
    print(f"数值积分: Fz = {Fz_far:.2e} N")
    print(f"点质量近似: Fz = {Fz_point:.2e} N")
    print(f"相对误差: {abs(Fz_far - Fz_point)/Fz_point*100:.2f}%")