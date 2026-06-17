import numpy as np

def truss3d_element_stiffness(x1, x2, E, A):
    """
    计算三维杆单元的长度、方向余弦、6×6全局刚度矩阵
    :param x1: 节点1坐标 [x, y, z] (m)
    :param x2: 节点2坐标 [x, y, z] (m)
    :param E: 弹性模量 (Pa)
    :param A: 杆件横截面积 (m^2)
    :return: L(长度), (cx, cy, cz)(方向余弦), Ke(6×6全局刚度矩阵)
    """
    x1 = np.array(x1, dtype=float)
    x2 = np.array(x2, dtype=float)
    dx = x2[0] - x1[0]
    dy = x2[1] - x1[1]
    dz = x2[2] - x1[2]

    # 计算单元长度，判断退化单元
    L = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    if L < 1e-12:
        raise ValueError("错误：两个节点坐标重合，为退化单元，无法计算！")

    # 方向余弦
    cx = dx / L
    cy = dy / L
    cz = dz / L

    # 构造3阶子矩阵C
    C = np.array([
        [cx ** 2, cx * cy, cx * cz],
        [cx * cy, cy ** 2, cy * cz],
        [cx * cz, cy * cz, cz ** 2]
    ])

    # 组装6×6全局刚度矩阵
    factor = E * A / L
    Ke = np.zeros((6, 6))
    Ke[0:3, 0:3] = factor * C
    Ke[0:3, 3:6] = -factor * C
    Ke[3:6, 0:3] = -factor * C
    Ke[3:6, 3:6] = factor * C

    return L, (cx, cy, cz), Ke

def truss3d_element_stress(x1, x2, E, A, de):
    """
    根据节点位移计算三维杆单元应变、应力、轴力
    :param x1: 节点1坐标 [x, y, z]
    :param x2: 节点2坐标 [x, y, z]
    :param E: 弹性模量 (Pa)
    :param A: 横截面积 (m^2)
    :param de: 节点位移向量 [u1,v1,w1,u2,v2,w2] (m)
    :return: epsilon(轴向应变), sigma(应力, Pa), N(轴力, N)
    """
    # 复用刚度函数计算长度与方向余弦，消除代码冗余
    L, (cx, cy, cz), _ = truss3d_element_stiffness(x1, x2, E, A)
    de = np.array(de, dtype=float).reshape(6, 1)

    # 应变-位移矩阵 B
    B = np.array([-cx, -cy, -cz, cx, cy, cz]).reshape(1, 6)
    epsilon = B @ de
    sigma = E * epsilon
    N = sigma * A

    return epsilon.item(), sigma.item(), N.item()

def print_matrix(mat, name):
    """格式化打印矩阵，提升可读性"""
    print(f"\n{name}：")
    for row in mat:
        print("  ".join(f"{val:.4e}" for val in row))

# ===================== 算例 1：沿x轴一维杆单元 =====================
print("=" * 70)
print("                    算例 1：沿X轴一维杆单元")
print("=" * 70)
x1_1 = [0, 0, 0]
x2_1 = [2, 0, 0]
E1 = 200e9
A1 = 1.0e-4
de1 = [0, 0, 0, 1e-3, 0, 0]

L1, dir1, Ke1 = truss3d_element_stiffness(x1_1, x2_1, E1, A1)
eps1, sig1, N1 = truss3d_element_stress(x1_1, x2_1, E1, A1, de1)

print(f"单元长度 L = {L1:.2f} m")
print(f"方向余弦 (cx, cy, cz) = {dir1[0]:.1f}, {dir1[1]:.1f}, {dir1[2]:.1f}")
# 打印6×6刚度矩阵
print_matrix(Ke1, "全局刚度矩阵 Ke (6×6)")
print(f"轴向应变 ε = {eps1:.6e}")
print(f"轴向应力 σ = {sig1:.2e} Pa = {sig1/1e6:.2f} MPa")
print(f"轴力 N = {N1:.2e} N")

# ===================== 算例 2：空间任意方向杆单元 =====================
print("\n" + "=" * 70)
print("                  算例 2：空间任意方向杆单元")
print("=" * 70)
x1_2 = [0, 0, 0]
x2_2 = [1, 2, 2]
E2 = 210e9
A2 = 2.0e-4
de2 = [0, 0, 0, 1e-3, 2e-3, 2e-3]

L2, dir2, Ke2 = truss3d_element_stiffness(x1_2, x2_2, E2, A2)
eps2, sig2, N2 = truss3d_element_stress(x1_2, x2_2, E2, A2, de2)

print(f"单元长度 L = {L2:.1f} m")
print(f"方向余弦 (cx, cy, cz) = {dir2[0]:.3f}, {dir2[1]:.3f}, {dir2[2]:.3f}")
# 打印6×6刚度矩阵
print_matrix(Ke2, "全局刚度矩阵 Ke (6×6)")
print(f"轴向应变 ε = {eps2:.6e}")
print(f"轴向应力 σ = {sig2:.2e} Pa = {sig2/1e6:.2f} MPa")
print(f"轴力 N = {N2:.2e} N")

# ---------------- 算例2 刚度矩阵性质验证 ----------------
print("\n" + "=" * 70)
print("                刚度矩阵性质验证（算例2）")
print("=" * 70)
# 1. 对称性校验
is_symmetric = np.allclose(Ke2, Ke2.T)
print(f"1. 刚度矩阵是否对称：{is_symmetric}")

# 2. 奇异性校验（行列式趋近于0）
det_Ke = np.linalg.det(Ke2)
print(f"2. 刚度矩阵行列式 det(Ke) = {det_Ke:.4e}")
print(f"   矩阵是否奇异：{abs(det_Ke) < 1e-6}")

# 3. 特征值校验（半正定：特征值≥0）
eig_vals = np.linalg.eigvals(Ke2)
print(f"3. 刚度矩阵全部特征值：\n   {np.round(eig_vals.real, 4)}")
all_pos = np.all(eig_vals.real >= -1e-6)
print(f"   所有特征值非负（半正定）：{all_pos}")

# 4. 刚体平移验证（题目必做：刚体位移无内力）
print("\n4. 刚体平移位移验证：")
# 整体刚体平移：所有节点同位移 [dx,dy,dz]，无相对变形
de_rigid = [0.001, 0.002, 0.003, 0.001, 0.002, 0.003]
eps_rigid, sig_rigid, N_rigid = truss3d_element_stress(x1_2, x2_2, E2, A2, de_rigid)
print(f"   刚体平移下应变 ε = {eps_rigid:.2e}")
print(f"   刚体平移下应力 σ = {sig_rigid:.2e} Pa")
print(f"   刚体平移下轴力 N = {N_rigid:.2e} N")
print("   结论：刚体平移不产生应变、应力、轴力，符合力学规律")

# ---------------- 任务4：刚度矩阵物理意义验证（必做） ----------------
print("\n" + "=" * 70)
print("              任务4：刚度矩阵物理意义验证")
print("=" * 70)
# 选取第3列（j=3，对应节点2的x向自由度，索引从0开始）
col_idx = 3
# 构造位移向量：仅第col_idx个自由度位移=1，其余为0
de_phy = np.zeros(6)
de_phy[col_idx] = 1.0
# 计算节点力 Fe = Ke * de
Fe = Ke2 @ de_phy.reshape(6, 1)
print(f"选取第 {col_idx+1} 列（自由度索引 {col_idx}），令该自由度位移=1，其余为0")
print("节点力向量 Fe = Ke · de：")
for idx, force in enumerate(Fe.flatten()):
    print(f"  自由度{idx+1} 节点力：{force:.4e} N")
print("\n物理说明：")
print("1. 刚度矩阵 Ke 的第 j 列，表示：仅第 j 个自由度产生单位位移、其余位移为0时，")
print("   全部6个自由度上需要施加的节点力；")
print("2. 矩阵元素 k_ij：第 j 个自由度产生单位位移时，在第 i 个自由度上引起的节点力。")