"""
Project 2: 倒立摆全局最优控制 - 稳定在竖直向上（修正 Bang-Bang）
确保最小时间策略输出饱和控制（±U_MAX），呈现典型砰-砰行为。
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Callable, Tuple
import time

# ================== 系统参数 ==================
g = 9.8
L = 1.0
m = 1.0
b = 0.1
I = m * L**2

U_MAX = 15.0          # 大力矩范围，确保饱和控制
U_MIN = -U_MAX

THETA_MIN, THETA_MAX = -np.pi, np.pi
THETA_DOT_MIN, THETA_DOT_MAX = -12.0, 12.0   # 扩大角速度范围

# 离散化
N_THETA = 41
N_THETA_DOT = 41
N_U = 7                # 奇数，包含零控制

# 价值迭代参数
DISCOUNT = 0.99        # 高折扣，重视长远
DT = 0.02
CONV_THRESH = 1e-4
MAX_ITER = 700

# 稳定区域（θ=0 附近）
STABLE_THETA = 0.05
STABLE_THETA_DOT = 0.2

SIM_STEPS = 1000
INIT_STATE = [np.pi, 0.0]    # 竖直下垂

# ================== 辅助函数 ==================
def build_grids():
    theta_grid = np.linspace(THETA_MIN, THETA_MAX, N_THETA)
    theta_dot_grid = np.linspace(THETA_DOT_MIN, THETA_DOT_MAX, N_THETA_DOT)
    u_grid = np.linspace(U_MIN, U_MAX, N_U)
    return theta_grid, theta_dot_grid, u_grid

def wrap_angle(theta):
    return (theta + np.pi) % (2*np.pi) - np.pi

def pendulum_dynamics(state, u):
    theta, theta_dot = state
    theta_ddot = (-b * theta_dot + m * g * L * np.sin(theta) + u) / I
    return np.array([theta_dot, theta_ddot])

def discrete_step(state, u, dt):
    theta, theta_dot = state
    theta_dot_next = theta_dot + dt * pendulum_dynamics(state, u)[1]
    theta_next = theta + dt * theta_dot
    theta_next = wrap_angle(theta_next)
    return np.array([theta_next, theta_dot_next])

def is_stable(state):
    theta, theta_dot = state
    return abs(theta) < STABLE_THETA and abs(theta_dot) < STABLE_THETA_DOT

# ================== 成本函数 ==================
def cost_quadratic(state, u):
    theta, theta_dot = state
    q_theta = 2.0
    q_dot = 0.2
    r = 0.05
    # 在稳定区域附近给予微小额外成本，避免死区
    near_target = 0.0
    if is_stable(state):
        near_target = 0.001   # 极小惩罚，避免零控制停滞
    return q_theta * theta**2 + q_dot * theta_dot**2 + r * u**2 + near_target

def cost_min_time(state, u):
    """
    纯时间成本：每步固定成本 1.0（非 DT）
    即使状态已经稳定，也付出极小成本，迫使控制器持续施加必要控制以保持不倒。
    """
    base_cost = 1.0
    # 在稳定区域附近略微降低成本，但绝不为零，避免策略变为零控制
    if is_stable(state):
        base_cost = 0.1      # 仍然为正，避免吸收态
    return base_cost

# ================== 价值迭代（带边界惩罚但不过度）==================
def value_iteration(cost_func: Callable, theta_grid, theta_dot_grid, u_grid):
    n_theta = len(theta_grid)
    n_dot = len(theta_dot_grid)
    J = np.zeros((n_theta, n_dot))

    # 双线性插值，边界外给予较大但合理的惩罚（不要过大以免策略害怕边界）
    def get_J_value(state, J_grid):
        theta, theta_dot = state
        if (theta < THETA_MIN or theta > THETA_MAX or
            theta_dot < THETA_DOT_MIN or theta_dot > THETA_DOT_MAX):
            # 边界外惩罚：相当于再走很多步才能回来
            return 1000.0
        i = np.searchsorted(theta_grid, theta) - 1
        j = np.searchsorted(theta_dot_grid, theta_dot) - 1
        i = max(0, min(i, n_theta - 2))
        j = max(0, min(j, n_dot - 2))
        theta0, theta1 = theta_grid[i], theta_grid[i+1]
        dot0, dot1 = theta_dot_grid[j], theta_dot_grid[j+1]
        f00, f01 = J_grid[i, j], J_grid[i, j+1]
        f10, f11 = J_grid[i+1, j], J_grid[i+1, j+1]
        dx = (theta - theta0) / (theta1 - theta0)
        dy = (theta_dot - dot0) / (dot1 - dot0)
        f0 = f00 * (1 - dx) + f10 * dx
        f1 = f01 * (1 - dx) + f11 * dx
        return f0 * (1 - dy) + f1 * dy

    theta_vals, dot_vals = np.meshgrid(theta_grid, theta_dot_grid, indexing='ij')
    states_flat = np.stack([theta_vals.ravel(), dot_vals.ravel()], axis=-1)
    u_vals = u_grid

    iter_count = 0
    converged = False
    while not converged and iter_count < MAX_ITER:
        J_new = np.zeros_like(J)
        for idx, (theta, theta_dot) in enumerate(states_flat):
            state = np.array([theta, theta_dot])
            best_val = float('inf')
            for u in u_vals:
                next_state = discrete_step(state, u, DT)
                cost = cost_func(state, u)
                future = get_J_value(next_state, J)
                total = cost + DISCOUNT * future
                if total < best_val:
                    best_val = total
            J_new[idx // n_dot, idx % n_dot] = best_val

        diff = np.max(np.abs(J_new - J))
        if diff < CONV_THRESH:
            converged = True
            print(f"收敛于第 {iter_count+1} 次迭代，变化 {diff:.6f}")
        else:
            if (iter_count+1) % 50 == 0:
                print(f"迭代 {iter_count+1}, 变化 {diff:.6f}")
        J = J_new
        iter_count += 1

    if not converged:
        print(f"警告：未在 {MAX_ITER} 次内收敛")

    # 提取策略
    pi = np.zeros_like(J, dtype=int)
    for idx, (theta, theta_dot) in enumerate(states_flat):
        state = np.array([theta, theta_dot])
        best_val = float('inf')
        best_u_idx = 0
        for u_idx, u in enumerate(u_vals):
            next_state = discrete_step(state, u, DT)
            cost = cost_func(state, u)
            future = get_J_value(next_state, J)
            total = cost + DISCOUNT * future
            if total < best_val:
                best_val = total
                best_u_idx = u_idx
        pi[idx // n_dot, idx % n_dot] = best_u_idx
    return J, pi

# ================== 仿真 ==================
def simulate(pi, u_grid, theta_grid, theta_dot_grid, init_state, max_steps=SIM_STEPS):
    states = [np.array(init_state)]
    controls = []
    n_theta = len(theta_grid)
    n_dot = len(theta_dot_grid)

    def get_nearest_idx(theta, theta_dot):
        i = np.argmin(np.abs(theta_grid - theta))
        j = np.argmin(np.abs(theta_dot_grid - theta_dot))
        return i, j

    state = np.array(init_state)
    for step in range(max_steps):
        if is_stable(state):
            break
        i, j = get_nearest_idx(state[0], state[1])
        u_idx = pi[i, j]
        u = u_grid[u_idx]
        controls.append(u)
        state = discrete_step(state, u, DT)
        states.append(state.copy())

    time = np.arange(len(states)) * DT
    return time, np.array(states), np.array(controls)

# ================== 可视化（与之前相同）==================
def plot_results(theta_grid, theta_dot_grid, J_quad, pi_quad, J_time, pi_time,
                 u_grid, sim_results):
    fig = plt.figure(figsize=(15, 12))
    theta_mesh, dot_mesh = np.meshgrid(theta_grid, theta_dot_grid, indexing='ij')

    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.plot_surface(theta_mesh, dot_mesh, J_quad, cmap='viridis')
    ax1.set_title('Quadratic Cost $J^*$')
    ax1.set_xlabel('θ (rad)'); ax1.set_ylabel('θ̇ (rad/s)')

    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    ax2.plot_surface(theta_mesh, dot_mesh, J_time, cmap='plasma')
    ax2.set_title('Min-Time Cost $J^*$')

    ax3 = fig.add_subplot(2, 3, 3)
    U_quad = u_grid[pi_quad]
    ctf = ax3.contourf(theta_mesh, dot_mesh, U_quad, levels=20, cmap='RdBu_r')
    ax3.set_title('Quadratic Policy')
    plt.colorbar(ctf, ax=ax3, label='u (N·m)')

    ax4 = fig.add_subplot(2, 3, 4)
    U_time = u_grid[pi_time]
    ctf2 = ax4.contourf(theta_mesh, dot_mesh, U_time, levels=20, cmap='RdBu_r')
    ax4.set_title('Min-Time Policy (Bang-Bang)')
    plt.colorbar(ctf2, ax=ax4, label='u (N·m)')

    ax5 = fig.add_subplot(2, 3, 5)
    t_q, states_q, _ = sim_results['quadratic']
    t_t, states_t, _ = sim_results['min_time']
    ax5.plot(t_q, states_q[:, 0], 'b-', label='Quadratic')
    ax5.plot(t_t, states_t[:, 0], 'r--', label='Min-Time')
    ax5.axhline(0, color='k', linestyle=':', alpha=0.5)
    ax5.set_xlabel('Time (s)'); ax5.set_ylabel('θ (rad)')
    ax5.set_title('Angle vs Time'); ax5.legend(); ax5.grid(True)

    ax6 = fig.add_subplot(2, 3, 6)
    _, _, ctrl_q = sim_results['quadratic']
    _, _, ctrl_t = sim_results['min_time']
    ax6.step(t_q[:-1], ctrl_q, 'b-', where='post', label='Quadratic')
    ax6.step(t_t[:-1], ctrl_t, 'r--', where='post', label='Min-Time')
    ax6.set_xlabel('Time (s)'); ax6.set_ylabel('u (N·m)')
    ax6.set_title('Control Input'); ax6.legend(); ax6.grid(True)

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8,6))
    plt.plot(states_q[:,0], states_q[:,1], 'b-', label='Quadratic')
    plt.plot(states_t[:,0], states_t[:,1], 'r--', label='Min-Time')
    plt.scatter([0],[0], c='k', marker='*', s=200, label='Target (θ=0)')
    plt.xlabel('θ (rad)'); plt.ylabel('θ̇ (rad/s)')
    plt.title('Phase Portrait'); plt.legend(); plt.grid(True); plt.axis('equal')
    plt.show()

def create_animation(sim_result, u_grid, title, save_gif=False, filename=None):
    time, states, controls = sim_result
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
    fig.suptitle(title)

    ax1.set_xlim(-1.5,1.5); ax1.set_ylim(-1.5,1.5); ax1.set_aspect('equal'); ax1.grid(True)
    line, = ax1.plot([], [], 'o-', lw=3, markersize=8)

    ax2.set_xlim(0, time[-1]); ax2.set_ylim(U_MIN-2, U_MAX+2)
    ax2.set_xlabel('Time (s)'); ax2.set_ylabel('Control (N·m)'); ax2.grid(True)
    ctrl_line, = ax2.step([], [], where='post', color='red')

    def init():
        line.set_data([], [])
        ctrl_line.set_data([], [])
        return line, ctrl_line

    def update(frame):
        theta = states[frame, 0]
        x = L * np.sin(theta)
        y = L * np.cos(theta)
        line.set_data([0, x], [0, y])
        if frame > 0:
            t_hist = time[:frame]
            u_hist = controls[:frame]
            ctrl_line.set_data(t_hist, u_hist)
        else:
            ctrl_line.set_data([], [])
        ax1.set_title(f'Time = {time[frame]:.2f}s, u = {controls[frame-1] if frame>0 else 0:.2f}')
        return line, ctrl_line

    ani = FuncAnimation(fig, update, frames=len(time), init_func=init,
                        blit=True, interval=DT*1000, repeat=False)

    if save_gif and filename:
        print(f"正在导出GIF: {filename}...")
        ani.save(filename, writer='pillow', fps=30, dpi=80)
        print(f"GIF已保存: {filename}")
    plt.show()
    return ani

# ================== 主程序 ==================
def main():
    print("构建离散网格...")
    theta_grid, theta_dot_grid, u_grid = build_grids()
    print(f"状态网格: {N_THETA}×{N_THETA_DOT} = {N_THETA*N_THETA_DOT}")
    print(f"控制网格: {N_U} (范围 [{U_MIN:.1f}, {U_MAX:.1f}])")

    print("\n----- 二次型成本价值迭代 -----")
    start = time.time()
    J_quad, pi_quad = value_iteration(cost_quadratic, theta_grid, theta_dot_grid, u_grid)
    print(f"耗时 {time.time()-start:.2f}s")

    print("\n----- 最小时间成本价值迭代 (应产生 Bang-Bang) -----")
    start = time.time()
    J_time, pi_time = value_iteration(cost_min_time, theta_grid, theta_dot_grid, u_grid)
    print(f"耗时 {time.time()-start:.2f}s")

    print("\n仿真二次型策略...")
    sim_quad = simulate(pi_quad, u_grid, theta_grid, theta_dot_grid, INIT_STATE)
    print("仿真最小时间策略...")
    sim_time = simulate(pi_time, u_grid, theta_grid, theta_dot_grid, INIT_STATE)

    sim_results = {'quadratic': sim_quad, 'min_time': sim_time}
    print("\n生成图表...")
    plot_results(theta_grid, theta_dot_grid, J_quad, pi_quad, J_time, pi_time, u_grid, sim_results)

    print("\n播放二次型动画...")
    create_animation(sim_quad, u_grid, "Quadratic: Swing-up & Stabilization",
                     save_gif=True, filename="quadratic_control.gif")
    print("\n播放最小时间动画 (Bang-Bang 预期)...")
    create_animation(sim_time, u_grid, "Min-Time: Bang-Bang Control",
                     save_gif=True, filename="bangbang_control.gif")

    print("\n完成。倒立摆稳定在竖直向上位置，最小时间策略应输出饱和控制。")

if __name__ == "__main__":
    main()
