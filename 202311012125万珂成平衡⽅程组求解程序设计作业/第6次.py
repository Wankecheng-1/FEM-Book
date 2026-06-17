import numpy as np
import time
import math
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

# 全局配置：取消弹窗，统一保存图片，兼容所有环境
plt.rcParams["font.family"] = ["SimHei", ]
plt.rcParams["axes.unicode_minus"] = False
np.set_printoptions(precision=8, suppress=True)


# ======================== 复用2.3桁架基础模块 ========================
def truss1d_element_stiffness(x1, x2, E, A):
    dx = x2[0] - x1[0]
    L = abs(dx)
    if L < 1e-12:
        raise ValueError("退化单元：两节点坐标重合！")
    coeff = E * A / L
    Ke = np.array([
        [coeff, -coeff],
        [-coeff, coeff]
    ], dtype=np.float64)
    return Ke


def truss2d_element_stiffness(x1, x2, E, A):
    dx = x2[0] - x1[0]
    dy = x2[1] - x1[1]
    L = np.sqrt(dx ** 2 + dy ** 2)
    if L < 1e-12:
        raise ValueError("退化单元：两节点坐标重合！")
    cx = dx / L
    cy = dy / L
    coeff = E * A / L
    c2 = cx * cx
    cy2 = cy * cy
    cxy = cx * cy
    Ke = coeff * np.array([
        [c2, cxy, -c2, -cxy],
        [cxy, cy2, -cxy, -cy2],
        [-c2, -cxy, c2, cxy],
        [-cxy, -cy2, cxy, cy2]
    ], dtype=np.float64)
    return Ke


def truss3d_element_stiffness(x1, x2, E, A):
    x1 = np.array(x1, dtype=float)
    x2 = np.array(x2, dtype=float)
    dx = x2[0] - x1[0]
    dy = x2[1] - x1[1]
    dz = x2[2] - x1[2]
    L = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    if L < 1e-12:
        raise ValueError("退化单元：两节点坐标重合，无法计算！")
    cx = dx / L
    cy = dy / L
    cz = dz / L
    C = np.array([
        [cx * cx, cx * cy, cx * cz],
        [cx * cy, cy * cy, cy * cz],
        [cx * cz, cy * cz, cz * cz]
    ])
    coeff = E * A / L
    Ke = np.zeros((6, 6))
    Ke[0:3, 0:3] = coeff * C
    Ke[0:3, 3:6] = -coeff * C
    Ke[3:6, 0:3] = -coeff * C
    Ke[3:6, 3:6] = coeff * C
    return L, (cx, cy, cz), Ke


def build_LM(IEN, node_dofs=3):
    n_elem = IEN.shape[1]
    LM = np.zeros((2 * node_dofs, n_elem), dtype=int)
    for e in range(n_elem):
        n1 = IEN[0, e]
        n2 = IEN[1, e]
        dof1 = [node_dofs * n1 + i for i in range(node_dofs)]
        dof2 = [node_dofs * n2 + i for i in range(node_dofs)]
        LM[:, e] = dof1 + dof2
    return LM


def assemble_K(n_dofs, LM, Ke_list):
    K = np.zeros((n_dofs, n_dofs), dtype=np.float64)
    n_elem = LM.shape[1]
    for e in range(n_elem):
        Ke = Ke_list[e]
        idx = LM[:, e]
        for a in range(len(idx)):
            for b in range(len(idx)):
                i = idx[a]
                j = idx[b]
                K[i, j] += Ke[a, b]
    return K


def get_reduced_system(K, F, fixed_dofs, fixed_values):
    n = K.shape[0]
    all_dofs = np.arange(n)
    free_dofs = np.setdiff1d(all_dofs, fixed_dofs)
    K_FF = K[np.ix_(free_dofs, free_dofs)]
    K_EF = K[np.ix_(fixed_dofs, free_dofs)]
    F_F = F[free_dofs]
    d_E = np.array(fixed_values, dtype=np.float64).reshape(-1, 1)
    rhs = F_F - (K_EF.T @ d_E).flatten()
    return free_dofs, fixed_dofs, K_FF, rhs, d_E


def reconstruct_disp_reaction(free_dofs, fixed_dofs, d_F, K, F, d_E):
    n = K.shape[0]
    d = np.zeros(n, dtype=np.float64)
    d[free_dofs] = d_F
    d[fixed_dofs] = d_E.flatten()
    reaction = (K @ d.reshape(-1, 1)).flatten()
    return d, reaction


def truss1d_force(x1, x2, E, A, de):
    dx = x2[0] - x1[0]
    L = abs(dx)
    eps = (de[1] - de[0]) / L
    sigma = E * eps
    N = sigma * A
    return eps, sigma, N


def truss2d_force(x1, x2, E, A, de):
    dx = x2[0] - x1[0]
    dy = x2[1] - x1[1]
    L = np.sqrt(dx ** 2 + dy ** 2)
    cx = dx / L
    cy = dy / L
    du = de[2] - de[0]
    dv = de[3] - de[1]
    eps = (cx * du + cy * dv) / L
    sigma = E * eps
    N = sigma * A
    return eps, sigma, N


def check_global_stiffness_property(K, tol=1e-6):
    print("\n" + "-" * 60)
    print("                总体刚度矩阵性质验证")
    print("-" * 60)
    is_symmetric = np.allclose(K, K.T, atol=tol)
    n = K.shape[0]
    det_K = np.linalg.det(K) if n <= 1000 else "超大矩阵跳过计算"
    rank_K = np.linalg.matrix_rank(K, tol=tol) if n <= 1000 else "超大矩阵跳过计算"
    full_rank = n
    is_singular = False
    if isinstance(rank_K, int):
        is_singular = abs(det_K) < tol or rank_K < full_rank
    eig_vals = np.linalg.eigvals(K) if n <= 1000 else "超大矩阵"
    real_eig = np.real(eig_vals) if isinstance(eig_vals, np.ndarray) else []
    all_non_neg = np.all(real_eig >= -1e-9) if len(real_eig) > 0 else False
    total_elem = K.size
    zero_elem = np.sum(np.abs(K) < tol)
    sparse_ratio = zero_elem / total_elem

    print(f"1. 矩阵对称性：{is_symmetric}")
    print(f"2. 行列式 det(K) = {det_K}")
    print(f"   秩/满秩：{rank_K} / {full_rank}，奇异：{is_singular}")
    if len(real_eig) > 0:
        print(f"3. 最小特征值：{np.min(real_eig):.4e}，半正定：{all_non_neg}")
    print(f"4. 零元素占比：{sparse_ratio:.2%}")
    print(f"5. 非零元素个数：{total_elem - zero_elem}")
    print("-" * 60 + "\n")
    return is_symmetric, is_singular, all_non_neg


# ======================== LDLT求解核心 ========================
def vec_norm(v):
    return np.linalg.norm(v, 2)


def residual_norm(K, a, R):
    r = R - K @ a
    r_norm = vec_norm(r)
    R_norm = vec_norm(R)
    rel_res = r_norm / R_norm if R_norm > 1e-12 else 0.0
    return r, r_norm, rel_res


def matrix_cond(K):
    return np.linalg.cond(K, 2) if K.shape[0] <= 1000 else "超大矩阵跳过计算"


def timer_func(func, runs=3):
    t_sum = 0.0
    res = None
    for _ in range(runs):
        t0 = time.perf_counter()
        res = func()
        t1 = time.perf_counter()
        t_sum += (t1 - t0)
    return res, t_sum / runs


def ldlt_factor(K):
    K = np.array(K, dtype=np.float64)
    n = K.shape[0]
    L = np.eye(n, dtype=np.float64)
    D = np.zeros(n, dtype=np.float64)
    min_main = 1e-10
    min_pivot = float('inf')
    for j in range(n):
        s = 0.0
        for k in range(j):
            s += L[j, k] * D[k] * L[j, k]
        D[j] = K[j, j] - s
        if D[j] < min_pivot:
            min_pivot = D[j]
        if D[j] <= min_main:
            raise ValueError(f"LDLT分解失败：第{j + 1}个主元 = {D[j]:.4e}，矩阵非正定/存在零主元")
        for i in range(j + 1, n):
            s2 = 0.0
            for k in range(j):
                s2 += L[i, k] * D[k] * L[j, k]
            L[i, j] = (K[i, j] - s2) / D[j]
    return L, D, min_pivot


def ldlt_solve(L, D, R):
    n = L.shape[0]
    R = np.array(R, dtype=np.float64).reshape(-1, )
    y = np.zeros(n)
    z = np.zeros(n)
    a = np.zeros(n)
    for i in range(n):
        s = 0.0
        for k in range(i):
            s += L[i, k] * y[k]
        y[i] = R[i] - s
    for i in range(n):
        z[i] = y[i] / D[i]
    for i in range(n - 1, -1, -1):
        s = 0.0
        for k in range(i + 1, n):
            s += L[k, i] * a[k]
        a[i] = z[i] - s
    return a


def solve_equilibrium(K_FF, rhs, method="ldlt"):
    solve_time = 0.0
    min_pivot = None
    if method == "ldlt":
        try:
            t0 = time.perf_counter()
            L, D, min_pivot = ldlt_factor(K_FF)
            sol = ldlt_solve(L, D, rhs)
            t1 = time.perf_counter()
            solve_time = t1 - t0
            return sol, True, "LDLT 稠密求解成功", min_pivot, solve_time
        except Exception as e:
            return None, False, str(e), min_pivot, solve_time
    elif method == "mkl_pardiso":
        try:
            t0 = time.perf_counter()
            K_sp = csr_matrix(K_FF)
            sol = spsolve(K_sp)
            t1 = time.perf_counter()
            solve_time = t1 - t0
            return sol, True, "MKL PARDISO 稀疏求解成功", None, solve_time
        except Exception as e:
            return None, False, str(e), None, solve_time
    else:
        raise NotImplementedError("仅支持 ldlt / mkl_pardiso")


# ======================== 绘图函数（保存到本地，不弹窗） ========================
def plot_and_save(x_grid, y_grid, data, title, save_name, cmap="jet"):
    plt.figure(figsize=(8, 6))
    cf = plt.contourf(x_grid, y_grid, data, levels=30, cmap=cmap)
    plt.colorbar(cf)
    plt.title(title, fontsize=12)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()
    plt.savefig(save_name, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"图片已保存为：{save_name}")


# ======================== 全部算例 ========================
def run_truss_1d():
    print("=" * 70)
    print("                算例0：一维两单元杆桁架")
    print("=" * 70)
    node_coords = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    IEN = np.array([[0, 1], [1, 2]]).T
    E = 200e9
    A1 = 1e-4
    A2 = 2e-4
    node_dof = 1
    n_nodes = 3
    n_dofs = n_nodes * node_dof

    Ke1 = truss1d_element_stiffness(node_coords[0], node_coords[1], E, A1)
    Ke2 = truss1d_element_stiffness(node_coords[1], node_coords[2], E, A2)
    Ke_list = [Ke1, Ke2]
    LM = build_LM(IEN, node_dof)
    K = assemble_K(n_dofs, LM, Ke_list)

    print(f"1. 总体刚度矩阵阶数 n = {K.shape[0]}")
    check_global_stiffness_property(K)

    F = np.zeros(n_dofs)
    F[2] = 1000.0
    fixed_dofs = [0]
    fixed_vals = [0.0]

    free_dofs, _, K_FF, rhs, d_E = get_reduced_system(K, F, fixed_dofs, fixed_vals)
    print(f"2. 缩减刚度矩阵阶数 = {K_FF.shape[0]}")
    print(f"   缩减矩阵特征值: {np.real(np.linalg.eigvals(K_FF))}")
    print(f"   缩减矩阵条件数: {matrix_cond(K_FF):.2e}")

    d_F, flag, msg, min_pivot, solve_time = solve_equilibrium(K_FF, rhs)
    print(f"3. LDLT分解状态：{flag}，提示信息：{msg}")
    if not flag:
        print("【错误终止】缩减刚度矩阵非正定！")
        print("=" * 70 + "\n")
        return
    print(f"   主元最小值：{min_pivot:.4e}")
    print(f"   求解耗时：{solve_time:.6f}")

    r, r_norm, rel_res = residual_norm(K_FF, d_F, rhs)
    print(f"4. 残差向量 r = \n{np.round(r, 8)}")
    print(f"   残差范数 ||r|| = {r_norm:.4e}")
    print(f"   相对残差 ||r||/||R|| = {rel_res:.4e}")

    d_full, reaction = reconstruct_disp_reaction(free_dofs, fixed_dofs, d_F, K, F, d_E)
    print(f"5. 解向量（自由自由度）a = \n{np.round(d_F, 8)}")
    print(f"6. 完整节点位移:\n{np.round(d_full, 8)}")
    print(f"   节点1位移: {d_full[0]:.8f} m, 节点2位移: {d_full[1]:.8f} m, 节点3位移: {d_full[2]:.8f} m")
    print(f"7. 支座反力: {reaction[fixed_dofs][0]:.2f} N")

    de1 = d_full[LM[:, 0]]
    de2 = d_full[LM[:, 1]]
    eps1, sig1, N1 = truss1d_force(node_coords[0], node_coords[1], E, A1, de1)
    eps2, sig2, N2 = truss1d_force(node_coords[1], node_coords[2], E, A2, de2)
    print(f"8. 单元1: 应变={eps1:.6e}, 应力={sig1 / 1e6:.2f} MPa, 轴力={N1:.2f} N")
    print(f"   单元2: 应变={eps2:.6e}, 应力={sig2 / 1e6:.2f} MPa, 轴力={N2:.2f} N")
    print("=" * 70 + "\n")


def run_truss_2d():
    print("=" * 70)
    print("                算例0：二维两杆三角桁架")
    print("=" * 70)
    node_coords = np.array([[0, 0, 0], [3, 0, 0], [0, 4, 0]])
    IEN = np.array([[0, 2], [1, 2]]).T
    E = 210e9
    A = 1e-4
    node_dof = 2
    n_nodes = 3
    n_dofs = n_nodes * node_dof

    Ke1 = truss2d_element_stiffness(node_coords[0], node_coords[2], E, A)
    Ke2 = truss2d_element_stiffness(node_coords[1], node_coords[2], E, A)
    Ke_list = [Ke1, Ke2]
    LM = build_LM(IEN, node_dof)
    K = assemble_K(n_dofs, LM, Ke_list)

    print(f"1. 总体刚度矩阵阶数 n = {K.shape[0]}")
    check_global_stiffness_property(K)

    F = np.zeros(n_dofs)
    F[4] = -1000.0
    fixed_dofs = [0, 1, 2, 3]
    fixed_vals = [0.0, 0.0, 0.0, 0.0]

    free_dofs, _, K_FF, rhs, d_E = get_reduced_system(K, F, fixed_dofs, fixed_vals)
    print(f"2. 缩减刚度矩阵阶数 = {K_FF.shape[0]}")
    print(f"   缩减矩阵特征值: {np.real(np.linalg.eigvals(K_FF))}")
    print(f"   缩减矩阵条件数: {matrix_cond(K_FF):.2e}")

    d_F, flag, msg, min_pivot, solve_time = solve_equilibrium(K_FF, rhs)
    print(f"3. LDLT分解状态：{flag}，提示信息：{msg}")
    if not flag:
        print("【错误终止】矩阵非正定！")
        print("=" * 70 + "\n")
        return
    print(f"   主元最小值：{min_pivot:.4e}")
    print(f"   求解耗时：{solve_time:.6f}")

    r, r_norm, rel_res = residual_norm(K_FF, d_F, rhs)
    print(f"4. 残差向量 r = \n{np.round(r, 8)}")
    print(f"   残差范数 ||r|| = {r_norm:.4e}")
    print(f"   相对残差 ||r||/||R|| = {rel_res:.4e}")

    d_full, reaction = reconstruct_disp_reaction(free_dofs, fixed_dofs, d_F, K, F, d_E)
    print(f"5. 解向量（自由自由度）a = \n{np.round(d_F, 8)}")
    print(f"6. 完整节点位移 (x0,y0,x1,y1,x2,y2):\n{np.round(d_full, 8)}")
    print(f"   节点2位移：x={d_full[4]:.8f} m, y={d_full[5]:.8f} m")
    print(f"7. 支座反力:\n{np.round(reaction[fixed_dofs], 2)}")

    de1 = d_full[LM[:, 0]]
    de2 = d_full[LM[:, 1]]
    eps1, sig1, N1 = truss2d_force(node_coords[0], node_coords[2], E, A, de1)
    eps2, sig2, N2 = truss2d_force(node_coords[1], node_coords[2], E, A, de2)
    print(f"8. 单元1 (0-2): 应变={eps1:.6e}, 应力={sig1 / 1e6:.2f} MPa, 轴力={N1:.2f} N")
    print(f"   单元2 (1-2): 应变={eps2:.6e}, 应力={sig2 / 1e6:.2f} MPa, 轴力={N2:.2f} N")
    print("=" * 70 + "\n")


def run_ill_condition():
    print("=" * 70)
    print("                任务2 病态方程组测试")
    print("=" * 70)
    K = np.array([[1.0000, 1.0000], [1.0000, 1.0001]])
    a_exact = np.array([1.0, 1.0])
    R = K @ a_exact
    print(f"1. 矩阵阶数 n = {K.shape[0]}")
    print(f"2. 矩阵K:\n{K}")
    print(f"   精确解: {a_exact}, 右端项: {R}")
    cond = matrix_cond(K)
    print(f"   条件数 cond(K) = {cond:.2e}")

    print("\n【双精度计算】")
    d_F, flag, msg, min_pivot, solve_time = solve_equilibrium(K, R)
    print(f"3. LDLT分解状态：{flag}，提示信息：{msg}")
    if flag:
        print(f"   主元最小值：{min_pivot:.4e}")
        print(f"   求解耗时：{solve_time:.6f}")
        _, rnorm_d, relres_d = residual_norm(K, d_F, R)
        err_d = vec_norm(d_F - a_exact) / vec_norm(a_exact)
        print(f"4. 解向量 a = {np.round(d_F, 8)}")
        print(f"5. 残差范数: {rnorm_d:.4e}, 相对残差: {relres_d:.4e}")
        print(f"6. 相对解误差: {err_d:.4e}")
    else:
        print("双精度矩阵分解失败，跳过残差计算")

    def round_sig(x, sig=4):
        if abs(x) < 1e-12:
            return 0.0
        return np.around(x, sig - int(math.floor(math.log10(abs(x)))) - 1)

    K4 = np.array([[round_sig(v) for v in row] for row in K])
    R4 = np.array([round_sig(v) for v in R])
    print("\n【4位有效数字截断】")
    print(f"1. 截断矩阵阶数 n = {K4.shape[0]}")
    print(f"2. 截断矩阵:\n{K4}")
    a4, flag_4, msg_4, min_pivot_4, solve_time_4 = solve_equilibrium(K4, R4)
    print(f"3. LDLT分解状态：{flag_4}，提示信息：{msg_4}")
    if flag_4:
        print(f"   主元最小值：{min_pivot_4:.4e}" if min_pivot_4 else "   主元最小值：无")
        _, r4, rr4 = residual_norm(K4, a4, R4)
        err4 = vec_norm(a4 - a_exact) / vec_norm(a_exact)
        print(f"4. 解向量 a = {np.round(a4, 8)}")
        print(f"5. 残差范数: {r4:.4e}, 相对残差: {rr4:.4e}")
        print(f"6. 相对解误差: {err4:.4e}")
    else:
        print("截断后矩阵非正定，LDLT分解失败，跳过残差计算")
    print("=" * 70 + "\n")


def run_tridiag_test():
    print("=" * 70)
    print("                算例1 三对角矩阵耗时测试")
    print("=" * 70)
    n_list = [10, 100, 500]
    for n in n_list:
        K = np.zeros((n, n))
        for i in range(n):
            K[i, i] = 2.0
            if i > 0:
                K[i, i - 1] = -1.0
                K[i - 1, i] = -1.0
        a_exact = np.ones(n)
        R = K @ a_exact
        print(f"\n=== 矩阵阶数 n = {n} ===")
        non_zero = np.count_nonzero(K)
        print(f"1. 非零元素个数：{non_zero}")
        print(f"2. 稀疏格式：稠密矩阵（三对角结构）")
        print(f"3. 条件数 cond(K) = {matrix_cond(K):.2e}")

        def solve_func():
            L, D, min_pivot = ldlt_factor(K)
            sol = ldlt_solve(L, D, R)
            return sol, min_pivot

        res_tuple, t_avg = timer_func(solve_func)
        sol = res_tuple[0]
        min_pivot = res_tuple[1]

        print(f"4. LDLT分解状态：成功")
        print(f"   主元最小值：{min_pivot:.4e}")
        print(f"   平均求解耗时：{t_avg:.4f} 秒")
        print(f"5. 解向量前5个元素：{np.round(sol[:5], 8)} (完整解为全1向量)")
        r, r_norm, rel_res = residual_norm(K, sol, R)
        err = vec_norm(sol - a_exact) / vec_norm(a_exact)
        print(f"6. 残差范数 ||r|| = {r_norm:.4e}")
        print(f"   相对残差 ||r||/||R|| = {rel_res:.4e}")
        print(f"7. 相对解误差：{err:.4e}")
        print(f"8. 求解器名称：自定义LDLT稠密求解器")
    print("\n" + "=" * 70 + "\n")


def run_non_pos_def():
    print("=" * 70)
    print("                算例2 非正定矩阵检测")
    print("=" * 70)
    K = np.array([[1, 2], [2, 1]])
    R = np.array([1, 1])
    print(f"1. 矩阵阶数 n = {K.shape[0]}")
    print(f"2. 测试矩阵:\n{K}")
    print(f"   右端项 R = {R}")
    sol, flag, msg, min_pivot, solve_time = solve_equilibrium(K, R)
    print(f"3. LDLT分解状态：{flag}")
    print(f"   提示信息：{msg}")
    print(f"4. 求解耗时：{solve_time:.6f} 秒")
    if not flag:
        print(f"5. 解向量：无")
        print(f"6. 残差向量/范数：无")
    else:
        print(f"5. 解向量 a = {sol}")
        r, r_norm, rel_res = residual_norm(K, sol, R)
        print(f"6. 残差向量 r = {r}")
        print(f"   残差范数 ||r|| = {r_norm:.4e}")
        print(f"   相对残差 ||r||/||R|| = {rel_res:.4e}")
    print("=" * 70 + "\n")


# 修复版泊松方程（边界缩聚 + 正定缩减矩阵 + 图片本地保存）
def run_poisson_2d():
    print("=" * 70)
    print("            拓展算例：二维泊松方程有限元求解")
    print("=" * 70)
    nx, ny = 20, 20
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    x_grid, y_grid = np.meshgrid(x, y)
    n_dof = nx * ny

    # 理论解
    def u_exact(xx, yy):
        return xx * (1 - xx) * yy * (1 - yy)

    u_true = u_exact(x_grid, y_grid)
    u_true_flat = u_true.flatten()

    hx = x[1] - x[0]
    hy = y[1] - y[0]
    K = np.zeros((n_dof, n_dof))
    R = np.full(n_dof, 2.0)

    # 组装整体矩阵
    for j in range(ny):
        for i in range(nx):
            idx = j * nx + i
            coeff_x = 1.0 / (hx ** 2)
            coeff_y = 1.0 / (hy ** 2)
            K[idx, idx] = 2 * coeff_x + 2 * coeff_y
            if i > 0:
                K[idx, j * nx + i - 1] = -coeff_x
            if i < nx - 1:
                K[idx, j * nx + i + 1] = -coeff_x
            if j > 0:
                K[idx, (j - 1) * nx + i] = -coeff_y
            if j < ny - 1:
                K[idx, (j + 1) * nx + i] = -coeff_y

    # 四边全部为约束自由度 u=0
    fixed_dofs = []
    for j in range(ny):
        for i in range(nx):
            if i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
                fixed_dofs.append(j * nx + i)
    fixed_vals = np.zeros_like(fixed_dofs)

    # 边界缩聚（核心修复：对缩减矩阵求解，保证正定）
    free_dofs, _, K_FF, rhs, d_E = get_reduced_system(K, R, fixed_dofs, fixed_vals)
    print(f"1. 整体矩阵阶数 n = {n_dof}")
    print(f"   缩减矩阵阶数 = {K_FF.shape[0]}")

    sol_F, flag, msg, min_p, t_solve = solve_equilibrium(K_FF, rhs)
    print(f"2. LDLT分解状态：{flag}，{msg}")
    if min_p is not None:
        print(f"   主元最小值：{min_p:.4e}")
    print(f"   求解耗时：{t_solve:.6f} s")

    if not flag:
        print("泊松方程矩阵分解失败，跳过误差计算与绘图！")
        print("=" * 70 + "\n")
        return

    # 重构全场解
    u_num_full = np.zeros(n_dof)
    u_num_full[free_dofs] = sol_F
    u_num_full[fixed_dofs] = d_E.flatten()
    u_num = u_num_full.reshape(ny, nx)

    # 误差计算
    err_grid = np.abs(u_true - u_num)
    max_err = np.max(err_grid)
    l2_err = vec_norm(u_num_full - u_true_flat) / vec_norm(u_true_flat)

    print(f"3. 理论解（全场最大值）: {np.max(u_true):.6e}")
    print(f"4. 数值解（全场最大值）: {np.max(u_num):.6e}")
    print(f"5. 节点最大绝对误差: {max_err:.6e}")
    print(f"6. 离散L2相对误差: {l2_err:.6e}")

    # 保存图片到代码同级目录（无需弹窗）
    print("\n>>> 正在保存结果图片...")
    plot_and_save(x_grid, y_grid, u_num, "泊松方程 - 有限元数值解", "数值解云图.png", cmap="viridis")
    plot_and_save(x_grid, y_grid, err_grid, "泊松方程 - 绝对误差云图", "误差云图.png", cmap="hot")

    print("=" * 70 + "\n")


# ======================== 程序入口 ========================
if __name__ == "__main__":
    run_truss_1d()
    run_truss_2d()
    run_ill_condition()
    run_tridiag_test()
    run_non_pos_def()
    run_poisson_2d()
    input("全部算例执行完毕，按回车键关闭控制台窗口...")