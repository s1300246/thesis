import socket
import struct
import numpy as np
import pandas as pd
from collections import deque
import matplotlib.pyplot as plt
import time
import datetime

# ==========================================
#  RTF Radar System (1m Wall Logic)
#  - ノイズ除去: 5cm未満は無視(青)
#  - 障害物(赤): 5cm 〜 1m
#  - 壁(黄): 1m以上
# ==========================================

UDP_PORT = 56301
OFFSET_HEADER = 36
POINT_STEP = 14

# バッファ設定
MAX_BUFFER_SIZE = 50000 

# レーダー設定
MAX_RADIUS = 5.0
MIN_RADIUS = 0.5
RADIUS_STEP = 0.5
ANGLE_SECTORS = 8

# --- ★判定パラメータ ---
VOXEL_SIZE = 0.2 

# 1. ノイズカット (これ以下は無視)
NOISE_FILTER_HEIGHT = 0.05  # 5cm

# 2. PCA検知の閾値
PCA_ROUGHNESS_THRESHOLD = 0.001 
PCA_VERTICALITY_THRESHOLD = 0.4

# 3. ★壁の高さ閾値 (ここを1.0mに変更)
# 1.0m以上なら「壁(黄)」、それ以下なら「障害物(赤)」
WALL_HEIGHT_THRESHOLD = 1.0

# 高さフィルタ
HEIGHT_LIMIT_MIN = -1.5 
HEIGHT_LIMIT_MAX = 2.0

# --- 矢印パラメータ ---
SCORE_SMOOTHING = 0.85

# --- 色定義 ---
COLOR_SAFE = '#a7d9ed'      # 水色 (安全)
COLOR_WALL = '#ffff00'      # 黄色 (>1m)
COLOR_DANGER = '#ff0000'    # 赤色 (5cm~1m)
COLOR_EMPTY = '#e0e0e0'     # グレー
COLOR_BLIND = '#808080'     # 死角

class AppState:
    def __init__(self):
        self.smoothed_scores = np.zeros(ANGLE_SECTORS)
        self.arrow_visible = False
state = AppState()

def parse_mid360_packet(data):
    if len(data) <= OFFSET_HEADER: return []
    points_list = []
    num_points = (len(data) - OFFSET_HEADER) // POINT_STEP
    for i in range(num_points):
        start = OFFSET_HEADER + (i * POINT_STEP)
        try:
            x, y, z = struct.unpack_from('<iii', data, start)
            if not (x == 0 and y == 0 and z == 0):
                points_list.append([x, y, z])
        except struct.error: break
    return points_list

def analyze_radar_filtered(points_np):
    df = pd.DataFrame(points_np, columns=['x', 'y', 'z'])
    df = df[(df['z'] >= HEIGHT_LIMIT_MIN) & (df['z'] <= HEIGHT_LIMIT_MAX)]
    
    r = np.sqrt(df['x']**2 + df['y']**2)
    df = df[r >= MIN_RADIUS]
    if len(df) == 0: return {}
    
    df['r_idx'] = np.floor((r - MIN_RADIUS) / RADIUS_STEP).astype(int)
    theta = np.arctan2(df['y'], df['x'])
    angle_width = 2 * np.pi / ANGLE_SECTORS
    shifted_theta = theta + (angle_width / 2)
    normalized_shifted = (shifted_theta + 2 * np.pi) % (2 * np.pi)
    df['theta_idx'] = np.floor(normalized_shifted / angle_width).astype(int) % ANGLE_SECTORS

    grid_status = {}

    for (rid, tid), group in df.groupby(['r_idx', 'theta_idx']):
        if MIN_RADIUS + (rid + 1) * RADIUS_STEP > MAX_RADIUS: continue
        if len(group) < 3: continue 
        
        # --- Step 1: ノイズカット ---
        z_max = group['z'].max()
        z_min = group['z'].min()
        delta_z = z_max - z_min
        
        # 5cm未満なら無視して青
        if delta_z < NOISE_FILTER_HEIGHT:
            grid_status[(rid, tid)] = 0
            continue

        # --- Step 2: PCA計算 ---
        local_pts = group[['x', 'y', 'z']].values
        centered = local_pts - np.mean(local_pts, axis=0)
        cov = np.cov(centered, rowvar=False)
        eig_val, eig_vec = np.linalg.eigh(cov)
        
        min_eigen = eig_val[0]      # 厚み
        nz = abs(eig_vec[:, 0][2])  # 垂直成分
        
        is_obstacle = False
        
        if min_eigen >= PCA_ROUGHNESS_THRESHOLD:
            is_obstacle = True
        elif nz <= PCA_VERTICALITY_THRESHOLD:
            is_obstacle = True
            
        # 安全策: 物理的に10cm以上あればPCA関係なく障害物
        if delta_z >= 0.10: 
            is_obstacle = True

        status = 0
        if is_obstacle:
            # --- Step 3: 高さ1m判定 ---
            if delta_z >= WALL_HEIGHT_THRESHOLD:
                status = 1 # 黄色 (1m以上の壁)
            else:
                status = 2 # 赤 (5cm~1mの障害物)
        
        if (rid, tid) in grid_status:
            if status > grid_status[(rid, tid)]: grid_status[(rid, tid)] = status
        else:
            grid_status[(rid, tid)] = status
            
    return grid_status

def on_key_press(event):
    if event.key == 's':
        filename = f"radar_1m_wall_{datetime.datetime.now().strftime('%H%M%S')}.png"
        plt.savefig(filename)
        print(f"Saved: {filename}")

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try: sock.bind(("", UDP_PORT))
    except OSError: 
        print("ポートエラー: 別のPythonが動いていませんか？")
        return

    point_buffer = deque(maxlen=MAX_BUFFER_SIZE)

    plt.ion()
    fig = plt.figure(figsize=(9, 10), facecolor='white')
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection='polar')
    fig.canvas.mpl_connect('key_press_event', on_key_press)
    
    ax.set_ylim(0, MAX_RADIUS)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    
    grid_radii = np.arange(MIN_RADIUS + RADIUS_STEP, MAX_RADIUS + 0.01, RADIUS_STEP)
    ax.set_rgrids(grid_radii, angle=0, fmt='%.1f m')
    ax.set_thetagrids([])
    ax.set_facecolor('#f0f0f0')
    ax.add_artist(plt.Circle((0, 0), MIN_RADIUS, transform=ax.transData._b, color=COLOR_BLIND, alpha=0.3, zorder=10))
    
    angle_width = 2 * np.pi / ANGLE_SECTORS
    plot_thetas = np.array([i * angle_width for i in range(ANGLE_SECTORS)])
    
    bars_layers = []
    num_rings = int((MAX_RADIUS - MIN_RADIUS) // RADIUS_STEP)
    for rid in range(num_rings):
        bottom_radius = MIN_RADIUS + rid * RADIUS_STEP
        bars = ax.bar(plot_thetas, height=RADIUS_STEP, width=angle_width, bottom=bottom_radius, color=COLOR_EMPTY, edgecolor='gray', linewidth=0.5)
        bars_layers.append(bars)
    
    arrow_patch = ax.annotate('', xy=(0, 0), xytext=(0, 0), arrowprops=dict(facecolor='green', shrink=0.05, width=5, headwidth=15), zorder=20)
    
    title_text = fig.text(0.5, 0.96, "RTF Radar (1m Threshold)", ha='center', fontsize=14, fontweight='bold')
    info_text = fig.text(0.5, 0.93, f"Ignore <5cm | Red: 5cm~1m | Yellow: >1m", ha='center', fontsize=10, color='gray')
    
    print(f"--- 1m判定モード ---")
    print(f"・高さ 5cm未満 -> 青 (無視)")
    print(f"・高さ 5cm〜1m -> 赤")
    print(f"・高さ 1m以上 -> 黄")
    last_draw = time.perf_counter()

    try:
        while True:
            if not plt.fignum_exists(fig.number): break
            data, _ = sock.recvfrom(2048)
            if len(data) > 0:
                new_points = parse_mid360_packet(data)
                if new_points: point_buffer.extend(new_points)
                
                if time.perf_counter() - last_draw >= 1.0:
                    last_draw = time.perf_counter()
                    if len(point_buffer) > 1000:
                        points_np = np.array(point_buffer, dtype=np.float64) / 1000.0
                        
                        grid_status = analyze_radar_filtered(points_np)
                        
                        instant_scores = np.zeros(ANGLE_SECTORS)
                        passable_sectors = []

                        for rid in range(len(bars_layers)):
                            bars = bars_layers[rid]
                            for tid, bar in enumerate(bars):
                                status = grid_status.get((rid, tid), -1)
                                
                                if status == 2:   # 赤
                                    bar.set_facecolor(COLOR_DANGER)
                                    instant_scores[tid] -= 5
                                elif status == 1: # 黄
                                    bar.set_facecolor(COLOR_WALL)
                                    instant_scores[tid] -= 2
                                elif status == 0: # 水色
                                    bar.set_facecolor(COLOR_SAFE)
                                    instant_scores[tid] += 1
                                else:
                                    bar.set_facecolor(COLOR_EMPTY)
                                    instant_scores[tid] -= 0.5

                        for tid in range(ANGLE_SECTORS):
                            status_near = grid_status.get((0, tid), -1)
                            if status_near == 0:
                                passable_sectors.append(tid)

                        state.smoothed_scores = 0.85 * state.smoothed_scores + 0.15 * instant_scores
                        
                        if len(passable_sectors) > 0:
                            state.arrow_visible = True
                            candidate_scores = state.smoothed_scores[passable_sectors]
                            best_candidate_idx = np.argmax(candidate_scores)
                            best_tid = passable_sectors[best_candidate_idx]
                            best_angle = plot_thetas[best_tid]

                            stop_distance = MAX_RADIUS
                            for rid in range(num_rings):
                                status = grid_status.get((rid, best_tid), -1)
                                if status >= 1: 
                                    stop_distance = MIN_RADIUS + rid * RADIUS_STEP
                                    break
                            
                            arrow_patch.xy = (best_angle, stop_distance)
                            arrow_patch.set_visible(True)
                        else:
                            state.arrow_visible = False
                            arrow_patch.set_visible(False)
                        
                        fig.canvas.draw()
                        fig.canvas.flush_events()
                        point_buffer.clear()
    except KeyboardInterrupt: pass
    finally: sock.close(); plt.close()

if __name__ == "__main__": main()
