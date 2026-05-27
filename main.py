import matplotlib
matplotlib.use("Agg")  # 必须保留，服务器无显示器，使用无界面绘图后端
import os
import re
import time
import warnings
import datetime
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as sig
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import savgol_filter
# 引入多进程库，实现多核并行处理
import multiprocessing as mp
import threading
# 全局文件锁，防止多进程日志错乱
LOG_LOCK = threading.Lock()

# 全局绘图设置：Linux 环境无中文字体，使用通用字体
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
warnings.filterwarnings("ignore")  # 忽略警告，保证输出干净

# ------------------------------------------------------------------------------
# 主程序入口（Docker 环境专用，路径固定）
# 功能：遍历 /data 下的 wav 文件，按批次并行处理，生成时域图、LOFAR、DEMON、CMS 图谱
# ------------------------------------------------------------------------------
def main():
    # ===================== 【Docker 固定路径，不可修改】=====================
    folder_path = "/data"           # 输入音频文件所在目录
    save_folder = "/data/Results"   # 输出图片保存目录
    # ====================================================================

    os.makedirs(save_folder, exist_ok=True)  # 自动创建输出目录

    # 读取目录下所有 wav 文件（不区分大小写）
    file_list = [f for f in os.listdir(folder_path) if f.lower().endswith(".wav")]
    file_list = sort_files_by_number(file_list)  # 按文件名数字排序

    if not file_list:
        print("❌ 未找到任何 .wav 文件")
        return

    # 批次处理设置：每次只处理 1 个文件
    batchSize = 1
    startBatch = 1             # 从第1个文件开始
    endBatch = len(file_list)
    batchStarts = list(range(startBatch, endBatch + 1, batchSize))
    batchEnds = [min(s + batchSize - 1, len(file_list)) for s in batchStarts]
    totalBatches = len(batchStarts)

    # 修正打印文案：当前为多核并行模式
    print(f'\n📊 并行处理任务开始，总批次数: {totalBatches}')

    # 初始化错误日志
    logFile = os.path.join(save_folder, 'ErrorLog.txt')
    with open(logFile, 'w', encoding='utf-8') as f:
        f.write(f'==== 错误日志开始 {datetime.datetime.now():%Y-%m-%d %H:%M:%S} ====\n')

    # 构造批次列表，用于多进程分发任务
    batch_task_list = list(zip(batchStarts, batchEnds))

    # 定义单批次任务包装函数
    def task_wrapper(batch_info):
        s, e = batch_info
        try:
            runBatchProcess(folder_path, file_list, save_folder, s, e)
            print(f'🟩 批次 [{s} ~ {e}] 处理完成')
        except Exception as e:
            print(f'⚠️ 批次 {s}-{e} 执行失败：{str(e)}')
            logError(save_folder, s, e, 0, e)

    # 根据服务器CPU核心数设置进程数，上限60，适配64核服务器
    process_num = min(mp.cpu_count(), 60)
    print(f"🚀 启用进程数：{process_num}")

    # 创建进程池并执行并行任务
    pool = mp.Pool(processes=process_num)
    pool.map(task_wrapper, batch_task_list)
    pool.close()
    pool.join()

    print('\n✅ 全部批次处理完成！')

# ------------------------------------------------------------------------------
# 核心处理函数
# 功能：读取一批音频文件 → 归一化 → 绘图 → FIR滤波 → LOFAR/DEMON/CMS分析
# ------------------------------------------------------------------------------
def runBatchProcess(folderPath, fileNames, saveFolder, batchStart, batchEnd):
    maxRetries = 1  # 每批次最多重试1次
    attempt = 0
    success = False
    tStart = time.time()

    # 失败重试机制
    while not success and attempt <= maxRetries:
        attempt += 1
        try:
            print(f'\n正在处理文件 {batchStart} 到 {batchEnd} (尝试 {attempt}/{maxRetries + 1})')

            allDataCell = []
            fileStartTime = None
            Fs = None

            # 读取当前批次内的所有文件
            for idx in range(batchStart - 1, batchEnd):
                fname = fileNames[idx]
                fpath = os.path.join(folderPath, fname)
                print(f'读取文件: {fname}')

                Fs, data = wav.read(fpath)
                if data.ndim > 1:
                    data = data.mean(axis=1)  # 多声道转单声道
                data = data.astype(np.float64)
                allDataCell.append(data)

                # 尝试从文件名前10位解析时间戳（格式：yymmddHHMM）
                if fileStartTime is None:
                    try:
                        fileStartTime = datetime.datetime.strptime(fname[:10], "%y%m%d%H%M")
                    except:
                        fileStartTime = datetime.datetime(1970, 1, 1)

            # 拼接批次内所有数据
            allData = np.concatenate(allDataCell)
            maxAbs = np.max(np.abs(allData))
            if maxAbs != 0 and not np.isnan(maxAbs):
                allData /= maxAbs  # 幅值归一化

            timeSeconds = np.arange(len(allData)) / Fs
            nameOnly = Path(fileNames[batchStart - 1]).stem  # 文件名（无后缀）

            # ==================== 绘制时域波形图 ====================
            h1 = plotTimeDomainSignalSafe(timeSeconds, allData, fileStartTime, batchStart, batchEnd)
            outJpg = os.path.join(saveFolder, f'{nameOnly}_TimeDomain_{batchStart}_{batchEnd}.jpg')
            h1.savefig(outJpg, dpi=300, bbox_inches='tight')
            plt.close(h1)
            print(f'已保存: {os.path.basename(outJpg)}')

            # ==================== FIR 带通滤波（10~10000Hz）====================
            freq_LB = [10, 10000]
            b = sig.firwin(129, [f / (Fs / 2) for f in freq_LB], pass_zero=False)
            x2 = sig.lfilter(b, [1], allData)
            x2 /= np.max(np.abs(x2))

            # ==================== LOFAR 谱分析（1~1000Hz）====================
            freq_band = [1, 1000]
            T_int = 2
            overlap_ratio = 0.85
            PdB, freqs, times = myLOFAR_STFT(x2, Fs, T_int, overlap_ratio, 'hann', 2)

            # 计算真实时间轴
            numFrames = PdB.shape[1]
            hopLen = T_int * Fs * (1 - overlap_ratio)
            frameTimesSec = (np.arange(numFrames) * hopLen + T_int * Fs / 2) / Fs
            lofarTimeActual = [fileStartTime + datetime.timedelta(seconds=t) for t in frameTimesSec]

            # 频率分段：1-500Hz / 501-1000Hz
            freqMask = (freqs >= freq_band[0]) & (freqs <= freq_band[1])
            freqsSel = freqs[freqMask]
            PdB_sel = PdB[freqMask, :]
            PdB_sel -= np.max(PdB_sel)

            idx1 = freqsSel <= 500
            idx2 = freqsSel > 500

            # 绘图并保存 LOFAR
            fig = plt.figure(figsize=(8, 6))
            plt.subplot(2, 1, 1)
            plt.pcolormesh(lofarTimeActual, freqsSel[idx1], PdB_sel[idx1], cmap='jet', shading='gouraud')
            plt.xlabel('时间')
            plt.ylabel('频率/Hz')
            plt.ylim([1, 500])
            plt.clim([-70, 0])
            plt.colorbar()
            plt.title('1-500 Hz')

            plt.subplot(2, 1, 2)
            plt.pcolormesh(lofarTimeActual, freqsSel[idx2], PdB_sel[idx2], cmap='jet', shading='gouraud')
            plt.xlabel('时间')
            plt.ylabel('频率/Hz')
            plt.ylim([501, 1000])
            plt.clim([-70, 0])
            plt.colorbar()
            plt.title('501-1000 Hz')
            plt.tight_layout()

            figPath = os.path.join(saveFolder, f'{nameOnly}_Origin_LOFAR_{batchStart:04d}.png')
            fig.savefig(figPath, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print("✅ 时频谱 已保存")

            # ==================== DEMON 调制分析（1:1对齐MATLAB）====================
            f_bp1 = 10
            f_bp2 = 1000
            b_bp = sig.firwin(129, [f_bp1 / (Fs / 2), f_bp2 / (Fs / 2)], pass_zero=False)

            x_bp = sig.lfilter(b_bp, [1], x2)
            env = np.abs(sig.hilbert(x_bp))  # 希尔伯特变换求包络
            env -= np.mean(env)
            smooth_win = int(0.05 * Fs)
            env = sig.convolve(env, np.ones(smooth_win) / smooth_win, mode='same')
            env /= np.max(np.abs(env))
            b_hp = sig.firwin(65, 1 / (Fs / 2), pass_zero="highpass")
            env = sig.lfilter(b_hp, [1], env)

            # DEMON 时频分析
            T_demon = 4.0
            overlap_demon = 0.8
            P_demon, f_demon, t_demon = mySTFT_Demon(env, Fs, T_demon, overlap_demon)

            # 归一化
            P_demon = np.abs(P_demon)
            P_demon = P_demon / np.max(P_demon)
            P_demon = 10 * np.log10(P_demon + 1e-6)

            # 只保留 0~200Hz
            fMax = 200
            idxF = f_demon <= fMax
            f_demon = f_demon[idxF]
            P_demon = P_demon[idxF, :]

            # 时间轴对齐
            numFramesD = len(t_demon)
            hopLenD = round(T_demon * Fs * (1 - overlap_demon))
            frameTimesD = (np.arange(numFramesD) * hopLenD + round(T_demon * Fs) / 2) / Fs
            demonTimeActual = [fileStartTime + datetime.timedelta(seconds=t) for t in frameTimesD]

            # 绘图 DEMON 时频谱
            figD = plt.figure(figsize=(8, 4))
            plt.pcolormesh(demonTimeActual, f_demon, P_demon, cmap='jet', shading='gouraud')
            plt.xlabel('时间')
            plt.ylabel('调制频率/Hz')
            plt.ylim([1, 200])
            plt.clim([-70, 0])
            plt.colorbar()
            plt.title('DEMON调制时频谱')
            plt.tight_layout()

            dPath = os.path.join(saveFolder, f'{nameOnly}_DEMON_TimeFreq_CORRECT.png')
            figD.savefig(dPath, dpi=300, bbox_inches='tight')
            plt.close(figD)
            print('✅ DEMON 已保存')

            # ==================== 一维 DEMON 功率谱 ====================
            x_demon_raw = x2 - np.mean(x2)
            sos = sig.butter(4, 1.0, fs=Fs, btype="high", output="sos")
            x_demon_1d = sig.sosfiltfilt(sos, x_demon_raw)
            x_demon_1d /= np.max(np.abs(x_demon_1d))

            ff_1d, X_1d = compute_demon_1d_simple(x_demon_1d, Fs)
            df = ff_1d[1] - ff_1d[0]
            cut_num = int(200 / df)
            ff_cut = ff_1d[:cut_num]
            X_cut = X_1d[:cut_num]
            threshold = savgol_filter(X_cut, 45, 2)  # 平滑门限

            # 绘图一维 DEMON
            fig1d = plt.figure(figsize=(8, 4))
            plt.plot(ff_cut, X_cut, 'k', linewidth=1.2)
            plt.plot(ff_cut, threshold, 'r', linewidth=1.2)
            plt.grid(True)
            plt.xlim(1, 200)
            plt.ylim([-30, 1])
            plt.xlabel("频率/Hz")
            plt.ylabel("幅值/dB")
            plt.title("DEMON一维功率谱")
            plt.tight_layout()
            save_path_1d = os.path.join(saveFolder, f"{nameOnly}_DEMON_1D.jpg")
            fig1d.savefig(save_path_1d, dpi=300, bbox_inches='tight')
            plt.close(fig1d)
            print("✅ 一维 DEMON 已保存")

            # ==================== CMS 循环调制谱分析 ====================
            f1, f2 = 100, 800
            b_cms = sig.firwin(129, [f1 / (Fs / 2), f2 / (Fs / 2)], pass_zero=False)
            x_cms = sig.lfilter(b_cms, [1], allData)

            PdB1, freqs1, times1 = myLOFAR_STFT(x_cms, Fs, 1, 0.5, 'hann', 0)

            Plinear = 10 ** (PdB1 / 10)
            Fnum, Tnum = Plinear.shape
            CMS = np.zeros((Tnum, Fnum))

            # 对每一列做FFT提取循环频率
            for i in range(Fnum):
                env = Plinear[i, :] - np.mean(Plinear[i, :])
                CMS[:, i] = np.abs(np.fft.fft(env))

            CMSdB = 10 * np.log10(CMS + 1e-6)
            CMSdB -= np.max(CMSdB)

            # 绘图 CMS
            figC = plt.figure(figsize=(8, 4))
            plt.pcolormesh(freqs1, np.arange(Tnum), CMSdB, cmap='jet', shading='gouraud')
            plt.xlabel('载频/Hz')
            plt.ylabel('双循环频率/Hz')
            plt.xlim([0, 1000])
            plt.ylim([0, 50])
            plt.clim([-15, 0])
            plt.colorbar()
            plt.title('CMS 循环调制谱')

            cPath = os.path.join(saveFolder, f'{nameOnly}_CMS_{batchStart:04d}.png')
            figC.savefig(cPath, dpi=300, bbox_inches='tight')
            plt.close(figC)

            success = True
            print(f'✅ 批次 {batchStart}-{batchEnd} 完成')

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f'⚠️ 失败：{str(e)}')
            if attempt > maxRetries:
                print('❌ 跳过')

# ==================== 一维 DEMON 谱计算（使用 Welch 法）====================
def compute_demon_1d_simple(x, fs):
    f, pxx = sig.welch(
        x ** 2, fs,
        nperseg=fs,
        noverlap=fs // 2,
        window="boxcar"
    )
    pxx = 10 * np.log10(pxx / np.max(pxx) + 1e-12)
    return f, pxx

# ==================== LOFAR 谱计算（STFT）====================
def myLOFAR_STFT(x, Fs, T_int, overlap_ratio, window_type, smooth_len):
    x = x.ravel()
    N = int(T_int * Fs)
    hop = max(1, int(N * (1 - overlap_ratio)))
    nfft = N

    if window_type == 'hann':
        win = sig.windows.hann(N)
    else:
        win = sig.windows.hamming(N)

    num_frames = (len(x) - N) // hop + 1
    P = np.zeros((nfft // 2 + 1, num_frames))

    for i in range(num_frames):
        idx = slice(i * hop, i * hop + N)
        frame = x[idx] - np.mean(x[idx])
        frame *= win
        X = np.fft.fft(frame, nfft)
        spec = np.abs(X[:nfft // 2 + 1]) ** 2
        spec[1:-1] *= 2
        P[:, i] = 10 * np.log10(spec + 1e-12)

    freqs = np.linspace(0, Fs / 2, nfft // 2 + 1)
    times = (np.arange(num_frames) * hop + N / 2) / Fs
    P -= np.max(P)
    return P, freqs, times

# ==================== DEMON 专用 STFT ====================
def mySTFT_Demon(x, Fs, T_int, overlap_ratio):
    x = x.ravel()
    N = int(T_int * Fs)
    N += N % 2
    hop = max(1, int(N * (1 - overlap_ratio)))
    nfft = N
    win = sig.windows.hann(N)

    num_frames = (len(x) - N) // hop + 1
    P = np.zeros((nfft // 2 + 1, num_frames))

    for i in range(num_frames):
        idx = slice(i * hop, i * hop + N)
        frame = x[idx] - np.mean(x[idx])
        frame *= win
        X = np.fft.fft(frame, nfft)
        spec = np.abs(X[:nfft // 2 + 1]) ** 2
        P[:, i] = spec

    f = np.linspace(0, Fs / 2, nfft // 2 + 1)
    t = (np.arange(num_frames) * hop + N / 2) / Fs
    return P, f, t

# ==================== 按文件名中的数字排序 ====================
def sort_files_by_number(fileNames):
    def get_num(s):
        nums = re.findall(r'\d+', s)
        return int(nums[0]) if nums else 9999

    return sorted(fileNames, key=get_num)

# ==================== 错误日志记录 ====================
def logError(saveFolder, s, e, a, exc):
    LOG_LOCK.acquire()
    try:
        with open(os.path.join(saveFolder, 'ErrorLog.txt'), 'a', encoding='utf-8') as f:
            f.write(f'{datetime.datetime.now()} 批次{s}-{e} 失败：{str(exc)}\n')
    finally:
        LOG_LOCK.release()

# ==================== 绘制时域图（超长信号自动降采样）====================
def plotTimeDomainSignalSafe(timeSec, dataVec, startTime, batchStart, batchEnd):
    L = len(timeSec)
    maxP = 50000  # 最多绘制5万个点，避免绘图卡顿
    if L > maxP:
        idx = np.linspace(0, L - 1, maxP, dtype=int)
        t = timeSec[idx]
        y = dataVec[idx]
    else:
        t = timeSec
        y = dataVec

    y /= np.max(np.abs(y))
    fig = plt.figure(figsize=(8, 4))
    plt.plot(t, y)
    plt.grid(True)
    plt.xlabel('时间')
    plt.ylabel('幅度/v')
    return fig

if __name__ == "__main__":
    main()