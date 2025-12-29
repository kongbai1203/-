import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#========================================只需修改此处========================================
fps_value=200 #设置帧率FPS
dt = 9.99e-6  #设置频差，如100FPS与100.1HZ的频差为9.99微秒
# 加载数据
data = pd.read_csv(r"D:\study\work\test_photo\MBL_100.1HZ_100FPS_2\MBL_100.1HZ_100FPS_2_frame_mean_values.csv") #所处理文件路径
#============================================================================================

# 设置字体为支持中文的字体
plt.rcParams['font.family'] = 'Times New Roman, SimSun'
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['font.size'] = 26  # 设置全局默认字号为14
#plt.rcParams['figure.dpi'] = 300  # 提高图像分辨率

def RLD(intensity, delaytime, max_tau=2000, threshold=5):
    fit_lifetime = []
    for i in range(1, len(intensity)):
        delay = delaytime[i] - delaytime[i - 1]
        if np.isclose(intensity[i - 1], intensity[i], rtol=1e-5, atol=1e-8):  # 增加浮点数容差处理
            continue
        lifetime = delay / np.log(intensity[i - 1] / intensity[i])  # 计算瞬时寿命
        if (not np.isinf(lifetime)) and (not np.isnan(lifetime)) and (0 < lifetime < max_tau):
            fit_lifetime.append(lifetime)
    if len(fit_lifetime) > 0:
        mean_tau = np.mean(fit_lifetime)
        std_tau = np.std(fit_lifetime)
        fit_lifetime = [tau for tau in fit_lifetime if abs(tau - mean_tau) < threshold * std_tau]
    if len(fit_lifetime) > 0:
        average = np.mean(fit_lifetime)
    else:
        average = np.nan
    return fit_lifetime, average


# CMM 算法
def fit_cmm(t, y):
    centroid = np.sum(t * y) / np.sum(y)  # 计算质心
    tau = centroid - t[0]  # 估计寿命
    return tau


# 最小二乘法（LSM）拟合
def exponential_decay(t, A, tau, B):
    return A * np.exp(-t / tau) + B


def fit_lsm(t, y):
    # 改进初始参数估计
    A0 = np.max(y) - np.min(y)  # A 的初始值为信号的最大值减去最小值
    B0 = np.min(y)  # B 的初始值为信号的最小值
    tau0 = (t[-1] - t[0]) / 4  # tau 的初始值为时间范围的 1/4
    p0 = [A0, tau0, B0]  # 初始参数值
    # 增加 maxfev 值
    popt, _ = curve_fit(exponential_decay, t, y, p0=p0, maxfev=2000)  # 拟合
    return popt[0], popt[1], popt[2]  # 返回 A, tau, B


# 修改时间偏移优化逻辑
def optimize_time_shift(base_t, base_signal, candidate_t, candidate_signal, A, tau, B, base_r2, shift_range=200,
                        shift_step=0.05, min_r2=0.85):
    best_r2 = -np.inf
    optimal_shift = 0
    improve_flag = False

    # 计算初始对齐值
    initial_shift = t_start - candidate_t[0]
    shift_steps = np.arange(initial_shift - shift_range, initial_shift + shift_range + shift_step, shift_step)

    for shift in shift_steps:
        # 应用时间偏移
        shifted_t = candidate_t + shift

        # 合并时间序列
        merged_t = np.concatenate([base_t, shifted_t])
        merged_signal = np.concatenate([base_signal, candidate_signal])

        # 按时间排序
        sort_idx = np.argsort(merged_t)
        merged_t = merged_t[sort_idx]
        merged_signal = merged_signal[sort_idx]

        # 执行拟合
        try:
            popt, _ = curve_fit(exponential_decay, merged_t, merged_signal,
                                p0=[A, tau, B], maxfev=2000)
            y_fit = exponential_decay(merged_t, *popt)

            # 计算R²
            residuals = merged_signal - y_fit
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((merged_signal - np.mean(merged_signal)) ** 2)
            current_r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

            # 判断是否最优
            if current_r2 > best_r2:
                best_r2 = current_r2
                optimal_shift = shift
        except:
            continue
    # 判断是否提升效果
    if best_r2 > 0.95:  # 需要设定合理的提升阈值
        improve_flag = True

    return optimal_shift, improve_flag


def extract_decay_segments(time_data, signal, threshold=0., min_length=4, max_upward_points=1):
    """
    自动提取衰减序列，允许一定数量的信号升高点。
    :param time_data: 时间序列
    :param signal: 荧光信号序列
    :param threshold: 检测下降的阈值（相对于峰值）
    :param min_length: 最小衰减序列长度
    :param max_upward_points: 允许的最大信号升高点数
    :return: 提取的衰减序列和时间序列，以及每个衰减序列的起始索引
    """
    decay_segments = []
    decay_times = []
    start_indices = []  # 记录每个衰减序列的起始索引
    i = 0
    while i < len(signal) - 1:
        # 检测信号峰值
        if signal[i] > signal[i + 1] + 1:
            peak_value = signal[i]
            # 找到下降阶段
            j = i + 1
            upward_count = 0  # 记录信号升高的点数
            while j < len(signal):
                if signal[j] < signal[j - 1]:
                    upward_count = 0  # 下降时重置计数
                else:
                    upward_count += 1  # 升高时计数加 1
                    if upward_count > max_upward_points:
                        break  # 超过允许的最大升高点数，停止提取
                j += 1
            # 提取衰减序列
            if j - i >= min_length and signal[i] - signal[j - 1] >= threshold * peak_value:
                segment = signal[i:j]
                t_segment = time_data[i:j]
                decay_segments.append(segment)
                decay_times.append(t_segment)
                start_indices.append(i)  # 记录起始索引
            i = j
        else:
            i += 1
    return decay_segments, decay_times, start_indices


# 根据 Index 对数据进行排序sshuju
data = data.sort_values(by='Index').reset_index(drop=True)

# 提取时间序列和信号
Index = data['Index'].values
time_data = Index * dt
signal = data['MeanGrayValue'].values

# 自动提取衰减序列
decay_segments, decay_times, start_indices = extract_decay_segments(time_data, signal, threshold=0.1, min_length=3)

# 在提取衰减序列后添加
decay_lengths = []  # 存储每个周期的衰减序列长度
max_first = []

for i, (segment, t_segment, start_idx) in enumerate(zip(decay_segments, decay_times, start_indices)):
    # 找到下降率最高的点
    max_drop = 0
    max_drop_idx = 0
    for j in range(1, len(segment)):
        drop = segment[j - 1] - segment[j]
        if drop > max_drop:
            max_drop = drop
            max_drop_idx = j

    # 从max_drop_idx开始提取序列
    decay_sequence = segment[max_drop_idx:]

    # 记录长度
    decay_lengths.append(len(decay_sequence))
    max_first.append(segment[max_drop_idx])

# 计算统计信息
if decay_lengths:
    average_length = np.mean(decay_lengths)
    print(f"\n平均衰减序列长度: {average_length:.2f} 个数据点")
    print(f"对应的平均时间长度: {average_length / fps_value * 1000:.2f} ms")  # FPS转换为毫秒
    print(f"\n平均信号: {np.mean(max_first):.2f} ")

# 对每个周期进行拟合
results = []
best_fit = None
best_score = -np.inf

# 设定参数的合理范围
tau_min = 10  # 最小衰减时间
tau_max = 1500  # 最大衰减时间

# 找到信号值的最高点作为信号顶点
max_signal_value = np.max(signal)
print(f"Max signal value: {max_signal_value}")

for i, (segment, t_segment, start_idx) in enumerate(zip(decay_segments, decay_times, start_indices)):
    # 检查数据点数量
    if len(segment) < 3:  # 最小数据点数为 3
        print(f"跳过周期 {i + 1}, 数据点太少: {len(segment)}")
        continue

    try:
        # 找到下降率最高的点
        max_drop = 0  # 最大下降率
        max_drop_idx = 0  # 最大下降点的位置
        for j in range(1, len(segment)):
            drop = segment[j - 1] - segment[j]  # 计算下降率
            if drop > max_drop:
                max_drop = drop
                max_drop_idx = j
        # 从下降率最高的点开始提取新的衰减序列
        segment = segment[max_drop_idx:]
        t_segment = t_segment[max_drop_idx:]

        # 检查新序列的数据点数量
        if len(segment) < 3:  # 最小数据点数为 3
            print(f"周期 {i + 1} 下降率最高的点后数据点太少: {len(segment)}")
            continue

        # 将时间序列调整为从 0 开始，并转换为微秒
        t_segment_adjusted = (t_segment - t_segment[0]) * 1e6

        # RLD 算法
        rld_lifetimes, rld_average = RLD(segment, t_segment_adjusted)

        # CMM 算法
        tau_cmm = fit_cmm(t_segment_adjusted, segment)

        # LSM 算法
        A_lsm, tau_lsm, B_lsm = fit_lsm(t_segment_adjusted, segment)

        # 计算拟合值
        y_fit = exponential_decay(t_segment_adjusted, A_lsm, tau_lsm, B_lsm)

        # 计算 R²（决定系数）
        residuals = segment - y_fit
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((segment - np.mean(segment)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        # 计算残差的均值和标准差
        residual_mean = np.mean(residuals)
        residual_std = np.std(residuals)

        # 检查参数是否在合理范围内
        is_tau_valid = (tau_min <= tau_lsm <= tau_max)

        # 计算综合评分（R² 减去残差标准差的比例）
        score = r2 - (residual_std / np.mean(segment))

        # 保存结果
        results.append({
            'cycle': i + 1,
            'RLD_tau': rld_average,
            'CMM_tau': tau_cmm,
            'LSM_tau': tau_lsm,
            'R2': r2,
            'residual_mean': residual_mean,
            'residual_std': residual_std,
            'max_drop_index': max_drop_idx
        })
        print(f"Cycle {i + 1} fitted successfully: R2 = {r2}, LSM_tau = {tau_lsm}, residual_std = {residual_std}")

        # 检查是否为最佳拟合
        if score > best_score and is_tau_valid:
            best_score = score
            best_fit = {
                'cycle': i + 1,
                'RLD_tau': rld_average,
                'CMM_tau': tau_cmm,
                'LSM_tau': tau_lsm,
                'R2': r2,
                'residual_mean': residual_mean,
                'residual_std': residual_std,
                't_segment': t_segment_adjusted,
                'segment': segment,
                'y_fit': y_fit,
                'max_drop_index': max_drop_idx,
                'start_index': start_idx,
                'A': A_lsm,  # 添加 A 的值
                'B': B_lsm  # 添加 B 的值
            }
            print(f"Updated best fit: Cycle {i + 1}, R2 = {r2}, LSM_tau = {tau_lsm}, residual_std = {residual_std}")
    except Exception as e:
        print(f"Error fitting segment {i + 1}: {str(e)}")

# ================ 包含倒推起始时间的完整合并逻辑 ================
if best_fit is not None:
    # 通过最佳拟合曲线倒推理论起始时间
    A_best = best_fit['A']
    tau_best = best_fit['LSM_tau']
    B_best = best_fit['B']
    max_signal = best_fit['segment'].max()

    # 理论模型：I(t) = A*exp(-t/tau) + B → t_start = -tau*ln((I(0)-B)/A)
    # 假设实际触发前信号为B，触发时上升到A+B
    t_start = -tau_best * np.log((max_signal - B_best) / A_best)
    print(f"理论信号起始点计算：t_start = {t_start:.2f}μs")

    # 初始化优化数据集（基准周期已对齐理论起点）
    optimized_data = {
        'times': best_fit['t_segment'] - t_start,  # 调整基准周期时间轴
        'signals': best_fit['segment'],
        'A': A_best,
        'tau': tau_best,
        'B': B_best,
        'R2': best_fit['R2']
    }


    # 包含时间排序去重的预处理函数
    def preprocess_data(timestamps, signals, precision=0.1):
        """预处理合并数据（排序+去重）"""
        # 排序处理
        sorted_idx = np.argsort(timestamps)
        sorted_t = timestamps[sorted_idx]
        sorted_s = signals[sorted_idx]

        # 按精度去重（防止过采样）
        unique_t = np.round(sorted_t, decimals=int(np.log10(1 / precision)))
        _, unique_idx = np.unique(unique_t, return_index=True)
        return sorted_t[unique_idx], sorted_s[unique_idx]


    # 迭代处理各周期
    for cycle_idx, (segment, t_segment, start_idx) in enumerate(zip(decay_segments, decay_times, start_indices)):
        if cycle_idx + 1 == best_fit['cycle']:
            continue  # 跳过基准周期

        # 从最大下降点开始提取数据
        max_drop = 0  # 最大下降率
        max_drop_idx = 0  # 最大下降点的位置
        for j in range(1, len(segment)):
            drop = segment[j - 1] - segment[j]  # 计算下降率
            if drop > max_drop:
                max_drop = drop
                max_drop_idx = j

        # 从 max_drop_idx 开始提取序列
        segment = segment[max_drop_idx:]
        t_segment = t_segment[max_drop_idx:]

        # +++ 新增代码：仅保留前20个数据点 +++
        segment = segment[:20]
        t_segment = t_segment[:20]

        # 检查新序列的数据点数量
        if len(segment) < 3:  # 最小数据点数为 3
            print(f"周期 {cycle_idx + 1} 下降率最高的点后数据点太少: {len(segment)}")
            continue

        # 转换时间到微秒级，并基于理论起点调整
        raw_time = (t_segment - t_segment[0]) * 1e6  # 原始周期相对时间
        aligned_time = raw_time - t_start  # 对齐理论起点

        # 根据第一个合并数据点大小在理论曲线上的位置调整时间
        first_point_value = segment[0]
        # 求解理论曲线上等于 first_point_value 对应的时间
        try:
            adjusted_time_shift = -tau_best * np.log((first_point_value - B_best) / A_best)
            aligned_time = aligned_time + adjusted_time_shift
        except ValueError:
            print(f"周期 {cycle_idx + 1} 无法根据第一个数据点调整时间，可能数据点超出理论曲线范围")
            continue

        # 优化该周期的时间偏移量
        optimal_shift, shift_valid = optimize_time_shift(
            base_t=optimized_data['times'],
            base_signal=optimized_data['signals'],
            candidate_t=aligned_time,
            candidate_signal=segment,
            A=optimized_data['A'],
            tau=optimized_data['tau'],
            B=optimized_data['B'],
            base_r2=optimized_data['R2'],
            shift_range=50,  # ±50μs搜索范围
            shift_step=0.5,  # 0.5μs分辨率
            min_r2=0.005  # R²提升最小阈值
        )

        if not shift_valid:
            print(f"⏩ 跳过周期 {cycle_idx + 1}，R²未达提升要求")
            continue

        # 应用最优偏移
        shifted_time = aligned_time + optimal_shift

        # 合并数据集
        merged_t = np.concatenate([optimized_data['times'], shifted_time])
        merged_s = np.concatenate([optimized_data['signals'], segment])

        # 数据预处理
        processed_t, processed_s = preprocess_data(merged_t, merged_s)

        # 更新拟合模型
        try:
            new_params, _ = curve_fit(exponential_decay,
                                      processed_t,
                                      processed_s,
                                      p0=[optimized_data['A'],
                                          optimized_data['tau'],
                                          optimized_data['B']],
                                      maxfev=3000)
            # 计算新R²
            y_pred = exponential_decay(processed_t, *new_params)
            ss_res = np.sum((processed_s - y_pred) ** 2)
            ss_tot = np.sum((processed_s - np.mean(processed_s)) ** 2)
            new_r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

            # 更新优化数据
            optimized_data.update({
                'times': processed_t,
                'signals': processed_s,
                'A': new_params[0],
                'tau': new_params[1],
                'B': new_params[2],
                'R2': new_r2
            })
            print(
                f"✅ 周期 {cycle_idx + 1} 成功合并 | 偏移量:{optimal_shift:.2f}μs | τ:{new_params[1]:.1f}μs | R方:{new_r2:.4f}")

            # 使用拟合出的方程计算100微秒时的数值
            time_point = 100  # 100微秒
            A = optimized_data['A']
            tau = optimized_data['tau']
            B = optimized_data['B']
            value_100us = exponential_decay(time_point, A, tau, B)
            print(f"使用拟合方程计算，在100微秒时的数值为: {value_100us}")

            # 可视化当前状态
            plt.figure(figsize=(12, 6))
            plt.scatter(optimized_data['times'], optimized_data['signals'],
                        alpha=0.4, c='blue', label="已合并数据")
            plt.plot(processed_t, y_pred, 'r-',
                     label=f"拟合曲线 (τ={new_params[1]:.1f}μs)")
            plt.xlabel("时间 (μs)")
            plt.ylabel("信号强度")
            plt.title(f"合并进度：已整合 {cycle_idx + 1} 个衰减周期 (R方={new_r2:.4f})")
            plt.legend()
            plt.show()

        except RuntimeError as e:
            print(f"❌ 周期 {cycle_idx + 1} 拟合失败：{str(e)}")
            continue

    # 使用最后的合并数据再次拟合
    final_t = optimized_data['times']
    final_s = optimized_data['signals']
    try:
        final_params, _ = curve_fit(exponential_decay, final_t, final_s,
                                    p0=[optimized_data['A'], optimized_data['tau'], optimized_data['B']],
                                    maxfev=3000)
        final_y_pred = exponential_decay(final_t, *final_params)

        # 计算最终的 R²
        final_residuals = final_s - final_y_pred
        final_ss_res = np.sum(final_residuals ** 2)
        final_ss_tot = np.sum((final_s - np.mean(final_s)) ** 2)
        final_r2 = 1 - (final_ss_res / final_ss_tot) if final_ss_tot != 0 else 0

        print(f"最终拟合参数: A = {final_params[0]:.2f}, tau = {final_params[1]:.2f}, B = {final_params[2]:.2f}")
        print(f"最终 R²: {final_r2:.4f}")

        # 绘制最终的拟合曲线与合并数据对比图
        plt.figure(figsize=(12, 7))
        plt.scatter(final_t, final_s, alpha=0.3, c='dodgerblue', edgecolors='none',
                    label=f"合并数据 (N={len(final_t)})")
        plt.plot(final_t, final_y_pred, 'r-', lw=2.5,
                 label=rf'最终拟合曲线 ($\tau$ = {final_params[1]:.1f}μs)')
        plt.xlabel("时间 (微秒)", fontsize=20)
        plt.ylabel("信号强度 (AU)", fontsize=20)
        plt.title(r"最终拟合曲线 VS 合并实验数据", fontsize=16, pad=20)
        plt.xlim(-50, final_params[1] * 5)  # 显示前5倍寿命区间
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right', framealpha=0.95)
        plt.tight_layout()

        # 添加统计注释
        text_str = '\n'.join((
            r'$\tau_{fit}$ = %.1f μs' % final_params[1],
            r'$B_{baseline}$ = %.1f AU' % final_params[2],
            r'$R^2$ = %.3f' % final_r2,
            r'$N_{points}$ = %d' % len(final_t)
        ))
        plt.gcf().text(0.72, 0.35, text_str, fontsize=12,
                       bbox=dict(facecolor='white', alpha=0.85))

        plt.show()
    except RuntimeError as e:

        print(f"最终拟合失败：{str(e)}")
