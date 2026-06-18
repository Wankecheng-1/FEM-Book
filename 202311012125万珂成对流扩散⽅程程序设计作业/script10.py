import warnings
# 屏蔽matplotlib字体警告
warnings.filterwarnings("ignore", category=UserWarning)
import numpy as np
import matplotlib.pyplot as plt

# Windows中文字体配置，图表中文正常显示
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# ===================== 任务1要求的三个标准函数 =====================
def alpha_supg(Pe):
    """SUPG最优稳定参数 alpha_opt = coth(Pe) - 1/Pe"""
    if abs(Pe) < 1e-10:
        return 0.0
    return np.cosh(Pe) / np.sinh(Pe) - 1.0 / Pe

def element_matrix(kappa, v, le, alpha):
    """
    生成2节点线性单元对流扩散单元矩阵 Ke
    kappa_bar = kappa + alpha * v * le / 2
    """
    kappa_bar = kappa + alpha * v * le / 2.0
    K_diff = (kappa_bar / le) * np.array([[1, -1],
                                         [-1, 1]])
    K_conv = (v / 2.0) * np.array([[-1, 1],
                                  [-1, 1]])
    Ke = K_diff + K_conv
    return Ke

def solve_advection_diffusion(nel, L, v, kappa, alpha):
    """
    有限元求解器主函数
    返回：节点坐标x, 数值解theta_num, 精确解theta_exact, 总刚矩阵K
    """
    # 1. 生成均匀网格
    nnodes = nel + 1
    le = L / nel
    x = np.linspace(0, L, nnodes)

    # 2. 组装总体刚度矩阵
    K = np.zeros((nnodes, nnodes))
    for e in range(nel):
        i = e
        j = e + 1
        Ke = element_matrix(kappa, v, le, alpha)
        K[i, i] += Ke[0, 0]
        K[i, j] += Ke[0, 1]
        K[j, i] += Ke[1, 0]
        K[j, j] += Ke[1, 1]

    # 3. 施加Dirichlet边界条件 theta(0)=0, theta(L)=1
    rhs = np.zeros(nnodes)
    K[0, :] = 0.0
    K[0, 0] = 1.0
    rhs[0] = 0.0
    K[-1, :] = 0.0
    K[-1, -1] = 1.0
    rhs[-1] = 1.0

    # 4. 求解线性方程组
    theta_num = np.linalg.solve(K, rhs)

    # 5. 稳定计算精确解（expm1防止指数溢出）
    Pe_global = v * L / kappa
    theta_exact = np.expm1(v * x / kappa) / np.expm1(Pe_global)

    return x, theta_num, theta_exact, K

# ===================== 单组Pe算例计算、输出、绘图 =====================
def run_case(target_Pe, nel=20, L=1.0, v=1.0):
    le = L / nel
    # 由Pe反推扩散系数 kappa = v*le/(2Pe)
    kappa = v * le / (2 * target_Pe)
    print(f"\n===== 单元Pe = {target_Pe:.2f}, 单元长度le={le:.6f}, 扩散系数κ={kappa:.8f} =====")

    # 三种格式求解
    x_gal, theta_gal, theta_ex, K_gal = solve_advection_diffusion(nel, L, v, kappa, alpha=0.0)
    x_upw, theta_upw, _, _ = solve_advection_diffusion(nel, L, v, kappa, alpha=1.0)
    a_opt = alpha_supg(target_Pe)
    x_supg, theta_supg, _, _ = solve_advection_diffusion(nel, L, v, kappa, alpha=a_opt)

    # 计算最大节点误差
    err_gal = np.max(np.abs(theta_gal - theta_ex))
    err_upw = np.max(np.abs(theta_upw - theta_ex))
    err_supg = np.max(np.abs(theta_supg - theta_ex))
    print(f"标准Galerkin最大误差: {err_gal:.12e}")
    print(f"迎风格式最大误差:     {err_upw:.12e}")
    print(f"SUPG最大误差:          {err_supg:.12e}")

    # 输出全部节点数据表（作业要求输出节点坐标、数值解、精确解）
    print("\n==================== 节点数据表 ====================")
    print(f"{'x坐标':<8}{'精确解':<12}{'Galerkin':<14}{'迎风格式':<14}{'SUPG':<14}")
    for xi, tex, tg, tu, ts in zip(x_gal, theta_ex, theta_gal, theta_upw, theta_supg):
        print(f"{xi:<8.4f}{tex:<12.6f}{tg:<14.6f}{tu:<14.6f}{ts:<14.6f}")

    # 绘图：精确解、Galerkin、迎风、SUPG四条曲线
    plt.figure(figsize=(10, 6))
    plt.plot(x_gal, theta_ex, 'k-', lw=2, label='精确解')
    plt.plot(x_gal, theta_gal, 'r--', marker='o', ms=3, label='标准Galerkin α=0')
    plt.plot(x_upw, theta_upw, 'g-.', marker='s', ms=3, label='迎风格式 α=1')
    plt.plot(x_supg, theta_supg, 'b:', marker='^', ms=3, label=f'SUPG α={a_opt:.4f}')
    plt.xlabel('x')
    plt.ylabel(r'$\theta(x)$')
    plt.title(f'一维对流扩散数值解对比，单元Pe = {target_Pe}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"Pe_{target_Pe}.png", dpi=300)
    plt.show()

    # 任务4：Pe=3时输出总刚矩阵并分析对称、正定性
    if abs(target_Pe - 3.0) < 1e-6:
        print("\n================ Pe=3 标准Galerkin矩阵分析 ================")
        print("全局总刚度矩阵K：")
        print(K_gal)
        is_sym = np.allclose(K_gal, K_gal.T)
        print(f"矩阵是否对称：{is_sym}")
        eig = np.linalg.eigvals(K_gal)
        min_eig = np.min(eig)
        is_posdef = min_eig > -1e-10
        print(f"矩阵是否正定：{is_posdef}")
        print(f"矩阵最小特征值：{min_eig:.4e}")

    return x_gal, theta_gal, theta_upw, theta_supg, theta_ex, err_gal, err_upw, err_supg

# ===================== 主程序入口 =====================
if __name__ == "__main__":
    # 任务2：两组标准算例 Pe=0.1、Pe=3.0，nel=20，L=1，v=1
    print("============ 算例1：Pe=0.1 扩散占优 ============")
    res01 = run_case(target_Pe=0.1, nel=20, L=1, v=1)

    print("\n\n============ 算例2：Pe=3.0 对流占优 ============")
    res3 = run_case(target_Pe=3.0, nel=20, L=1, v=1)

    # 附加题：网格加密收敛测试 nel=10,20,40,80，绘制误差收敛曲线
    print("\n\n============ 附加题：网格加密收敛测试 Pe=3 ============")
    nel_list = [10, 20, 40, 80]
    err_gal_list = []
    err_supg_list = []
    L = 1.0
    v = 1.0
    Pe_target = 3.0
    for nel in nel_list:
        le = L / nel
        kappa = v * le / (2 * Pe_target)
        _, th_gal, th_ex, _ = solve_advection_diffusion(nel, L, v, kappa, alpha=0)
        e_gal = np.max(np.abs(th_gal - th_ex))
        a_opt = alpha_supg(Pe_target)
        _, th_supg, _, _ = solve_advection_diffusion(nel, L, v, kappa, alpha=a_opt)
        e_supg = np.max(np.abs(th_supg - th_ex))
        err_gal_list.append(e_gal)
        err_supg_list.append(e_supg)
        print(f"单元数nel={nel:2d} | Galerkin误差={e_gal:.12e} | SUPG误差={e_supg:.12e}")

    # 绘制log-log收敛曲线
    plt.figure(figsize=(10,6))
    h_list = [L / n for n in nel_list]
    plt.loglog(h_list, err_gal_list, 'ro-', label='标准Galerkin')
    plt.loglog(h_list, err_supg_list, 'bs-', label='SUPG')
    plt.xlabel('单元长度 le (对数坐标)')
    plt.ylabel('最大节点误差 (对数坐标)')
    plt.title('网格加密误差收敛曲线 Pe=3')
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.savefig("convergence_curve.png", dpi=300)
    plt.show()