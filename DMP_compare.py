import numpy as np
import matplotlib.pyplot as plt

# ---------- 轨迹生成（你给的） ----------
def conditional_spirals(batch_size=1000):
    t = np.linspace(0, 1, batch_size)

    alpha = 4
    s = np.exp(-alpha * t)

    theta = 6.0 * np.pi * s
    r = t + 0.0015 * theta

    x = np.stack([
        r * np.cos(theta),
        r * np.sin(theta)
    ], axis=1)

    return x, t, s

def conditional_C(batch_size=1000):
    t = np.linspace(0, 1, batch_size)

    # 控制尾巴长度的衰减项
    beta = 2.5
    s = np.exp(-beta * t)

    # 主体：一个圆
    theta = 2 * np.pi * t
    r = 1.0

    x_circle = r * np.cos(theta)
    y_circle = r * np.sin(theta)

    # 尾巴：向下的漂移
    tail_strength = 0.6
    y_tail = -tail_strength * t

    # 组合成 alpha
    x = np.stack([
        x_circle,
        y_circle + y_tail
    ], axis=1)

    return x, t, s

def conditional_alpha(batch_size=1000):
    t = np.linspace(0, 1, batch_size)

    # 相位分两段：loop + tail
    t1 = t[t <= 0.7] / 0.7        # loop 部分
    t2 = (t[t > 0.7] - 0.7) / 0.3  # tail 部分

    # loop：不完整圆（大约 1.6π）
    theta1 = 1.6 * np.pi * t1 + 0.2 * np.pi
    r1 = 1.0

    x1 = r1 * np.cos(theta1)
    y1 = r1 * np.sin(theta1)

    # tail：从内部向下拉一笔
    x_tail_start = x1[-1]
    y_tail_start = y1[-1]

    x2 = x_tail_start + 0.1 * np.sin(np.pi * t2)
    y2 = y_tail_start - 1.2 * t2

    x = np.concatenate([x1, x2])
    y = np.concatenate([y1, y2])

    s = np.exp(-3 * t)

    return np.stack([x, y], axis=1), t, s


y_demo, t, s = conditional_alpha(1000)
dt = t[1] - t[0]

# 速度 & 加速度
ydot = np.gradient(y_demo, axis=0) / dt
yddot = np.gradient(ydot, axis=0) / dt

y0 = y_demo[0]
g = y_demo[-1]

# ---------- 高斯基函数 ----------
N = 15
C = np.linspace(0, 1, N)
H = np.ones(N) * 30

def basis(s):
    return np.exp(-H * (s - C) ** 2)

# ---------- 计算目标 forcing term ----------
alpha = 25
beta = alpha / 4

f_target = yddot - alpha * (beta * (g - y_demo) - ydot)

# ---------- 回归求权重 ----------
Psi = np.array([basis(si) for si in s])  # (T,N)
Psi_norm = Psi / (np.sum(Psi, axis=1, keepdims=True) + 1e-6)

W = np.linalg.lstsq(Psi_norm, f_target, rcond=None)[0]  # (N,2)

# ---------- DMP rollout ----------
def dmp_rollout(W, y0, g, s, dt, scale):
    y = y0.copy()
    ydot = np.zeros(2)

    Y = []

    for si in s:
        psi = basis(si)
        psi = psi / (np.sum(psi) + 1e-6)

        f = psi @ W  # (2,)

        yddot = alpha * (beta * (g - y) - ydot) + f * scale

        ydot += yddot * dt
        y += ydot * dt

        Y.append(y.copy())

    return np.array(Y)

g_new = np.array([g[0]+1.5, g[1]-1.5])

# L2 norm
distance_demo = np.linalg.norm(g - y0)
distance_real = np.linalg.norm(g_new - y0)
scale_l2 = distance_real/distance_demo
print("scale_l2:", scale_l2)

# Projection
v_demo = g - y0
v_real = g_new - y0
scale_projection = np.dot(v_real, v_demo) / (np.dot(v_demo, v_demo) + 1e-8)
print("scale_projection:", scale_projection)

Y_hat_l2 = dmp_rollout(W, y0, g_new, s, dt, scale_l2)
Y_hat_projection = dmp_rollout(W, y0, g_new, s, dt, scale_projection)
Y_hat_origin = dmp_rollout(W, y0, g_new, s, dt, 1)

# ---------- 画图 ----------
plt.figure(figsize=(6, 6))
plt.plot(y_demo[:, 0], y_demo[:, 1], 'k', label="demo")
plt.plot(Y_hat_l2[:, 0], Y_hat_l2[:, 1], 'r--', label="DMP_2013_l2")
plt.plot(Y_hat_projection[:, 0], Y_hat_projection[:, 1], 'g--', label="DMP_2013_projection")
plt.plot(Y_hat_origin[:, 0], Y_hat_origin[:, 1], 'b--', label="DMP_2006")
plt.axis("equal")
plt.legend()
plt.title("DMP Trajectory Reconstruction")
plt.show()
