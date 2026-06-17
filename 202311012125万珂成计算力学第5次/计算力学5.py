import numpy as np

# ===================== 1. 三维杆单元刚度矩阵计算 =====================
def truss3d_element_stiffness(x1, x2, E, A):
    x1 = np.array(x1, dtype=float)
    x2 = np.array(x2, dtype=float)
    dx = x2[0] - x1[0]
    dy = x2[1] - x1[1]
    dz = x2[2] - x1[2]
    L = np.sqrt(dx**2 + dy**2 + dz**2)
    if L < 1e-12:
        raise ValueError("退化单元：两个节点坐标重合，无法计算！")
    cx = dx / L
    cy = dy / L
    cz = dz / L
    # 方向余弦子矩阵C
    C = np.array([
        [cx*cx, cx*cy, cx*cz],
        [cx*cy, cy*cy, cy*cz],
        [cx*cz, cy*cz, cz*cz]
    ])
    coeff = E * A / L
    Ke = np.zeros((6, 6))
    Ke[0:3, 0:3] = coeff * C
    Ke[0:3, 3:6] = -coeff * C
    Ke[3:6, 0:3] = -coeff * C
    Ke[3:6, 3:6] = coeff * C
    return L, (cx, cy, cz), Ke

# ===================== 2. 生成对号矩阵LM =====================
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

# ===================== 3. 总体刚度矩阵组装 =====================
def assemble_K(n_dofs, LM, Ke_list):
    K = np.zeros((n_dofs, n_dofs))
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

# ===================== 4. 缩减法求解（增加奇异矩阵容错） =====================
def solve_with_reduction(K, F, fixed_dofs, fixed_values):
    n = K.shape[0]
    all_dofs = np.arange(n)
    free_dofs = np.setdiff1d(all_dofs, fixed_dofs)

    K_FF = K[np.ix_(free_dofs, free_dofs)]
    K_EF = K[np.ix_(fixed_dofs, free_dofs)]
    F_F = F[free_dofs]

    d_E = np.array(fixed_values).reshape(-1, 1)
    rhs = F_F - (K_EF.T @ d_E).flatten()

    # 奇异矩阵兜底：solve报错则切换最小二乘
    try:
        d_F = np.linalg.solve(K_FF, rhs)
    except np.linalg.LinAlgError:
        print("自由刚度矩阵奇异，启用最小二乘求解")
        d_F = np.linalg.lstsq(K_FF, rhs, rcond=None)[0]

    # 重构完整位移向量
    d = np.zeros(n)
    d[free_dofs] = d_F
    d[fixed_dofs] = fixed_values
    # 计算全域节点力（包含支座反力）
    reaction = (K @ d.reshape(-1, 1)).flatten()
    return d, reaction

# ===================== 5. 单元应变、应力、轴力后处理 =====================
def truss_element_force(x1, x2, E, A, de):
    dx = x2[0] - x1[0]
    dy = x2[1] - x1[1]
    dz = x2[2] - x1[2]
    L = np.sqrt(dx**2 + dy**2 + dz**2)
    cx = dx / L
    cy = dy / L
    cz = dz / L
    B = np.array([-cx, -cy, -cz, cx, cy, cz])
    eps = B @ de
    sigma = E * eps
    N = sigma * A
    return eps, sigma, N

# ===================== 6. 总体刚度矩阵性质验证函数 =====================
def check_global_stiffness_property(K, tol=1e-6):
    print("\n" + "-"*60)
    print("                总体刚度矩阵性质验证")
    print("-"*60)

    # 1. 对称性验证
    is_symmetric = np.allclose(K, K.T, atol=tol)
    print(f"1. 矩阵对称性：{is_symmetric}")

    # 2. 奇异性验证（行列式 + 矩阵秩）
    det_K = np.linalg.det(K)
    rank_K = np.linalg.matrix_rank(K, tol=tol)
    full_rank = K.shape[0]
    is_singular = abs(det_K) < tol or rank_K < full_rank
    print(f"2. 矩阵行列式 det(K) = {det_K:.4e}")
    print(f"   矩阵秩 / 满秩：{rank_K} / {full_rank}")
    print(f"   矩阵是否奇异：{is_singular}")

    # 3. 半正定性：特征值非负
    eig_vals = np.linalg.eigvals(K)
    real_eig = np.real(eig_vals)
    all_non_neg = np.all(real_eig >= -tol)
    print(f"3. 特征值最小值：{np.min(real_eig):.4e}")
    print(f"   所有特征值非负（半正定）：{all_non_neg}")

    # 4. 稀疏性统计
    total_elem = K.size
    zero_elem = np.sum(np.abs(K) < tol)
    sparse_ratio = zero_elem / total_elem
    print(f"4. 稀疏性：总元素数={total_elem}, 零元素数={zero_elem}")
    print(f"   零元素占比 = {sparse_ratio:.2%}，矩阵为稀疏矩阵")

    print("-"*60 + "\n")
    return is_symmetric, is_singular, all_non_neg

# ==============================================================================
# 算例1：一维两单元杆结构
# ==============================================================================
print("=" * 70)
print("                   算例 1：一维两单元拉杆结构")
print("=" * 70)
# 模型定义
node_coords_1 = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
IEN_1 = np.array([[0, 1], [1, 2]]).T
E1 = 200e9
A1_1 = 1e-4
A1_2 = 2e-4

# 计算单元刚度矩阵
L1, _, Ke1 = truss3d_element_stiffness(node_coords_1[0], node_coords_1[1], E1, A1_1)
L2, _, Ke2 = truss3d_element_stiffness(node_coords_1[1], node_coords_1[2], E1, A1_2)
Ke_list_1 = [Ke1, Ke2]

# 组装总体刚度
n_nodes_1 = 3
node_dof = 3
n_dofs_1 = n_nodes_1 * node_dof
LM1 = build_LM(IEN_1, node_dof)
K1 = assemble_K(n_dofs_1, LM1, Ke_list_1)

# -------- 新增：总体刚度矩阵性质验证 --------
check_global_stiffness_property(K1)

# 载荷：节点3 x方向施加1000N拉力
F1 = np.zeros(n_dofs_1)
F1[6] = 1000.0

# 边界条件：仅固定节点0的x向自由度，释放y/z避免矩阵奇异
fixed_dofs_1 = [0]
fixed_vals_1 = [0.0]
d1, reaction1 = solve_with_reduction(K1, F1, fixed_dofs_1, fixed_vals_1)

# 输出结构整体结果
print("总体刚度矩阵对角线元素：")
print(np.round(np.diag(K1), 2))
print("\n全部节点位移向量：")
print(np.round(d1, 8))
print(f"\n支座约束反力：{np.round(reaction1[fixed_dofs_1][0], 2)} N")
print(f"节点3 X向位移：{d1[6]:.8f} m")

# 单元内力计算输出
de_elem1 = d1[LM1[:, 0]]
eps_1, sig_1, N_1 = truss_element_force(node_coords_1[0], node_coords_1[1], E1, A1_1, de_elem1)
de_elem2 = d1[LM1[:, 1]]
eps_2, sig_2, N_2 = truss_element_force(node_coords_1[1], node_coords_1[2], E1, A1_2, de_elem2)
print(f"\n单元1结果：应变={eps_1:.6e}，应力={sig_1/1e6:.2f} MPa，轴力={N_1:.2f} N")
print(f"单元2结果：应变={eps_2:.6e}，应力={sig_2/1e6:.2f} MPa，轴力={N_2:.2f} N")

# ==============================================================================
# 算例2：二维两杆桁架结构
# ==============================================================================
print("\n" + "=" * 70)
print("                   算例 2：二维两杆三角桁架")
print("=" * 70)
# 模型定义
node_coords_2 = np.array([[0, 0, 0], [3, 0, 0], [0, 4, 0]])
IEN_2 = np.array([[0, 2], [1, 2]]).T
E2 = 210e9
A2 = 1e-4

# 单元刚度
_, _, KeA = truss3d_element_stiffness(node_coords_2[0], node_coords_2[2], E2, A2)
_, _, KeB = truss3d_element_stiffness(node_coords_2[1], node_coords_2[2], E2, A2)
Ke_list_2 = [KeA, KeB]

# 组装总体刚度
n_nodes_2 = 3
n_dofs_2 = n_nodes_2 * 3
LM2 = build_LM(IEN_2, node_dof)
K2 = assemble_K(n_dofs_2, LM2, Ke_list_2)

# -------- 新增：总体刚度矩阵性质验证 --------
check_global_stiffness_property(K2)

# 载荷：顶部节点x向1000N
F2 = np.zeros(n_dofs_2)
F2[6] = 1000.0

# 边界：节点1、2全部自由度固定
fixed_dofs_2 = [0, 1, 2, 3, 4, 5]
fixed_vals_2 = [0.0] * 6
d2, reaction2 = solve_with_reduction(K2, F2, fixed_dofs_2, fixed_vals_2)

# 整体输出
print("全部节点位移向量：")
print(np.round(d2, 8))
print(f"\n支座全部约束反力：\n{np.round(reaction2[fixed_dofs_2], 2)}")

# 单元内力
de_A = d2[LM2[:, 0]]
eps_A, sig_A, N_A = truss_element_force(node_coords_2[0], node_coords_2[2], E2, A2, de_A)
de_B = d2[LM2[:, 1]]
eps_B, sig_B, N_B = truss_element_force(node_coords_2[1], node_coords_2[2], E2, A2, de_B)
print(f"\n斜杆单元A：应变={eps_A:.6e}，应力={sig_A/1e6:.2f} MPa，轴力={N_A:.2f} N")
print(f"水平单元B：应变={eps_B:.6e}，应力={sig_B/1e6:.2f} MPa，轴力={N_B:.2f} N")