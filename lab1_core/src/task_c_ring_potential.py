import numpy as np
import matplotlib.pyplot as plt


def ring_potential_point(
    x: float,
    y: float,
    z: float,
    a: float = 1.0,
    q: float = 1.0,
    n_phi: int = 720
) -> float:
    """
    用离散积分计算圆环电荷在单个点 (x, y, z) 的电势
    圆环位于 xy 平面，半径 a，总电荷 q
    """
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    dphi = 2.0 * np.pi / n_phi

    # 圆环上电荷元位置
    xr = a * np.cos(phi)
    yr = a * np.sin(phi)

    # 场点到电荷元距离
    r = np.sqrt((x - xr) ** 2 + (y - yr) ** 2 + z ** 2)

    # 避免除零（理论上测试不会正好取到环上）
    r = np.where(r < 1e-12, 1e-12, r)

    # 数值积分
    V = (q / (2.0 * np.pi)) * np.sum(1.0 / r) * dphi

    return float(V)


def ring_potential_grid(
    y_grid,
    z_grid,
    x0: float = 0.0,
    a: float = 1.0,
    q: float = 1.0,
    n_phi: int = 720
):
    """
    在 x = x0 截面上的 yz 网格中计算电势矩阵

    支持两种输入：
    1. y_grid, z_grid 为一维数组 -> 自动生成 meshgrid
    2. y_grid, z_grid 已经是二维 meshgrid
    """
    y = np.asarray(y_grid)
    z = np.asarray(z_grid)

    # 如果输入是一维数组，则生成二维网格
    if y.ndim == 1 and z.ndim == 1:
        Y, Z = np.meshgrid(y, z)
    else:
        # 如果已经是二维网格，则直接广播
        Y, Z = np.broadcast_arrays(y, z)

    V = np.zeros_like(Y, dtype=float)

    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            V[i, j] = ring_potential_point(
                x=x0,
                y=Y[i, j],
                z=Z[i, j],
                a=a,
                q=q,
                n_phi=n_phi
            )

    return V


def axis_potential_analytic(
    z: float,
    a: float = 1.0,
    q: float = 1.0
) -> float:
    """
    z 轴上 (0, 0, z) 的解析解
    """
    return q / np.sqrt(a**2 + z**2)


def compute_electric_field(V, y, z):
    """
    用数值差分计算 yz 平面中的电场
    Ey = -∂V/∂y
    Ez = -∂V/∂z
    """
    dy = y[1] - y[0]
    dz = z[1] - z[0]

    Ey = np.zeros_like(V)
    Ez = np.zeros_like(V)

    # 中心差分
    Ey[1:-1, :] = -(V[2:, :] - V[:-2, :]) / (2 * dy)
    Ez[:, 1:-1] = -(V[:, 2:] - V[:, :-2]) / (2 * dz)

    # 边界：前向 / 后向差分
    Ey[0, :] = -(V[1, :] - V[0, :]) / dy
    Ey[-1, :] = -(V[-1, :] - V[-2, :]) / dy

    Ez[:, 0] = -(V[:, 1] - V[:, 0]) / dz
    Ez[:, -1] = -(V[:, -1] - V[:, -2]) / dz

    return Ey, Ez


def visualize_ring_potential():
    """
    绘制圆环电势与电场
    """
    a = 1.0
    q = 1.0

    y_vals = np.linspace(-3.0, 3.0, 100)
    z_vals = np.linspace(-3.0, 3.0, 100)

    print("正在计算电势... ⏳")
    V = ring_potential_grid(
        y_vals,
        z_vals,
        x0=0.0,
        a=a,
        q=q,
        n_phi=720
    )

    # 用于绘图的 meshgrid
    Y, Z = np.meshgrid(y_vals, z_vals, indexing="ij")

    print("正在计算电场... ⚡")
    Ey, Ez = compute_electric_field(V, y_vals, z_vals)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ===== 左图：等势线 + 电场箭头 =====
    ax1 = axes[0]

    contourf = ax1.contourf(Y, Z, V, levels=50, cmap="RdYlBu_r")
    plt.colorbar(contourf, ax=ax1, label="Potential V")

    contours = ax1.contour(Y, Z, V, colors="black", linewidths=0.5)
    ax1.clabel(contours, inline=True, fontsize=8)

    skip = 6
    ax1.quiver(
        Y[::skip, ::skip],
        Z[::skip, ::skip],
        Ey[::skip, ::skip],
        Ez[::skip, ::skip]
    )

    ax1.plot([a, -a], [0, 0], "ro", label="Ring projection")
    ax1.set_xlabel("y")
    ax1.set_ylabel("z")
    ax1.set_title("Potential & Electric Field")
    ax1.set_aspect("equal")
    ax1.legend()

    # ===== 右图：流线 =====
    ax2 = axes[1]

    ax2.contour(Y, Z, V, colors="black", linewidths=0.5)

    stream = ax2.streamplot(
        y_vals,
        z_vals,
        Ey.T,
        Ez.T,
        density=1.5
    )

    ax2.plot([a, -a], [0, 0], "ro", label="Ring projection")
    ax2.set_xlabel("y")
    ax2.set_ylabel("z")
    ax2.set_title("Electric Field Streamlines")
    ax2.set_aspect("equal")
    ax2.legend()

    plt.tight_layout()
    plt.show()

    # ===== 验证解析解 =====
    print("\n验证 z 轴解析解... 🔍")

    z_axis = np.linspace(0.0, 3.0, 50)

    V_numeric = np.array([
        ring_potential_point(0.0, 0.0, z, a, q)
        for z in z_axis
    ])

    V_exact = axis_potential_analytic(z_axis, a, q)

    max_error = np.max(np.abs(V_numeric - V_exact))
    print(f"最大误差 = {max_error:.3e} ✅")

    plt.figure(figsize=(8, 5))
    plt.plot(z_axis, V_numeric, label="Numeric", linewidth=2)
    plt.plot(z_axis, V_exact, "--", label="Analytic", linewidth=2)
    plt.xlabel("z")
    plt.ylabel("V(0,0,z)")
    plt.title("Potential on z-axis")
    plt.grid(True)
    plt.legend()
    plt.show()

    return V, Ey, Ez


if __name__ == "__main__":
    V, Ey, Ez = visualize_ring_potential()