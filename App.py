import os
import sys
import copy
import random
from collections import deque

import numpy as np
import requests
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import imageio
import cv2  # for dilation

# Gym and RL imports:
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback

# -------------------------------
# Helper & Simulation Functions
# -------------------------------

MODEL_FILE = "fireline_dqn.zip"

def download_dem_opentopography(output_file="san_francisco_bay_dem.tif"):
    if not os.path.exists(output_file):
        url = "https://portal.opentopography.org/API/globaldem"
        params = {
            "demtype": "SRTMGL1",
            "south": 36,
            "north": 38.3,
            "west": -123.0,
            "east": -117.0,
            "outputFormat": "GTiff",
            "API_Key": "074cf58d7228c5449b3f9669cdca5b31"
        }
        print("Downloading DEM data from OpenTopography...")
        response = requests.get(url, params=params)
        if response.status_code == 200:
            with open(output_file, "wb") as f:
                f.write(response.content)
            print("Download complete. DEM saved as:", output_file)
        else:
            print("Error downloading DEM. Status code:", response.status_code)
            raise Exception("Failed to download DEM data.")
    else:
        print("DEM file already exists. Using local file.")
    return output_file

def convex_hull(points):
    points = sorted(set(points))
    if len(points) <= 1:
        return points
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]

def bresenham_line(x0, y0, x1, y1):
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    if dx > dy:
        err = dx/2.0
        while x != x1:
            points.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy/2.0
        while y != y1:
            points.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    points.append((x1, y1))
    return points

def point_in_polygon(pt, polygon):
    x, y = pt
    inside = False
    n = len(polygon)
    if n == 0:
        return False
    p1x, p1y = polygon[0]
    for i in range(n+1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y) and y <= max(p1y, p2y) and x <= max(p1x, p2x):
            if p1y != p2y:
                xinters = (y-p1y) * (p2x-p1x) / (p2y-p1y) + p1x
            if p1x == p2x or x <= xinters:
                inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def is_adjacent_to_fireline(pt, state):
    col, row = pt
    rows, cols = state.shape
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            r = row + dr
            c = col + dc
            if 0 <= r < rows and 0 <= c < cols:
                if state[r, c] == 3:
                    return True
    return False

def find_nearest_connected(pt, state):
    rows, cols = state.shape
    visited = set()
    queue = deque([pt])
    visited.add(pt)
    while queue:
        current = queue.popleft()
        if is_adjacent_to_fireline(current, state) and state[current[1], current[0]] == 0:
            return current
        col, row = current
        for dc in [-1, 0, 1]:
            for dr in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                neighbor = (col+dc, row+dr)
                r, c = neighbor[1], neighbor[0]
                if 0 <= r < rows and 0 <= c < cols and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
    return None

def compute_wind_factor(dr, dc, wind_vector, wind_speed, beta):
    neighbor_vec = np.array([dc, -dr])
    norm = np.linalg.norm(neighbor_vec)
    if norm == 0:
        return 1
    neighbor_unit = neighbor_vec / norm
    alignment = np.dot(neighbor_unit, wind_vector)
    return np.exp(beta * wind_speed * alignment)

def compute_slope_factor(current_elev, neighbor_elev, k):
    dz = neighbor_elev - current_elev
    slope_angle = np.arctan(dz)
    return np.exp(k * slope_angle)

def update_state(state, topo, wind_vector, wind_speed, p_base, beta, k, kernel_radius, sigma):
    new_state = state.copy()
    rows, cols = state.shape
    burning_cells = np.argwhere(state == 1)
    for r, c in burning_cells:
        new_state[r, c] = 2
        current_elev = topo[r, c]
        for dr in range(-kernel_radius, kernel_radius+1):
            for dc in range(-kernel_radius, kernel_radius+1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r+dr, c+dc
                if 0 <= nr < rows and 0 <= nc < cols and state[nr, nc] == 0:
                    if topo[nr, nc] <= 0:
                        continue
                    distance = np.sqrt(dr**2+dc**2)
                    weight = np.exp(- (distance**2)/(2*sigma**2))
                    wind_factor = compute_wind_factor(dr, dc, wind_vector, wind_speed, beta)
                    neighbor_elev = topo[nr, nc]
                    slope_factor = compute_slope_factor(current_elev, neighbor_elev, k)
                    p = p_base * weight * wind_factor * slope_factor
                    p = min(1.0, p)
                    if np.random.random() < p:
                        new_state[nr, nc] = 1
    return new_state

def polygon_to_mask(polygon, grid_shape):
    mask = np.zeros(grid_shape, dtype=np.uint8)
    pts = np.array(polygon, np.int32)
    pts = pts.reshape((-1,1,2))
    cv2.fillPoly(mask, [pts], 1)
    return mask

def dilate_mask(mask, dilation_size=1):
    kernel = np.ones((dilation_size*2+1, dilation_size*2+1), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=1)
    return dilated

def preprotect_cities(state, city_blobs, dilation_size=1):
    rows, cols = state.shape
    for blob in city_blobs:
        mask = polygon_to_mask(blob, (rows, cols))
        dilated = dilate_mask(mask, dilation_size=dilation_size)
        state[dilated == 1] = 3
    return state

def generate_city_blob(city, blob_radius=3, num_points=8):
    points = []
    for _ in range(num_points):
        angle = random.uniform(0, 2*np.pi)
        r = random.uniform(0.8*blob_radius, blob_radius)
        offset = (int(r * np.cos(angle)), int(r * np.sin(angle)))
        points.append((city[0] + offset[0], city[1] + offset[1]))
    blob = convex_hull(points)
    if blob[0] != blob[-1]:
        blob.append(blob[0])
    return blob

# Modified plot_simulation: no colorbar, no axis labels/ticks, accepts figsize parameter.
def plot_simulation(topo, extent, state, city_blobs, step, out_path, figsize=(10,8)):
    fig, ax = plt.subplots(figsize=figsize)
    water = np.ma.masked_where(topo > 0, topo)
    land = np.ma.masked_where(topo <= 0, topo)
    terrain = plt.cm.terrain
    shifted = terrain(np.linspace(0.2,1,256))
    terrain_shifted = LinearSegmentedColormap.from_list("terrain_shifted", shifted)
    ax.imshow(land, extent=extent, cmap=terrain_shifted)
    dark_blue = ListedColormap(['#00008B'])
    ax.imshow(water, extent=extent, cmap=dark_blue)
    land_elev = topo.astype(float)
    land_elev[land_elev <= 0] = np.nan
    land_elev = np.flipud(land_elev)
    levels = np.linspace(np.nanmin(land_elev), np.nanmax(land_elev), 7)
    contours = ax.contour(land_elev, levels=levels, extent=extent, colors='black', linewidths=0.5)
    ax.clabel(contours, inline=True, fontsize=8, fmt='%1.0f')
    fire_colors = [(0,0,0,0), (1,0,0,0.6), (0,0,0,0.8), (1,1,0,0.9)]
    fire_cmap = ListedColormap(fire_colors)
    ax.imshow(state, extent=extent, cmap=fire_cmap, interpolation='none')
    if city_blobs:
        rows_, cols_ = topo.shape
        left, right, bottom, top = extent
        for blob in city_blobs:
            poly_coords = []
            for (col, row) in blob:
                x = left + (col/cols_)*(right-left)
                y = top - (row/rows_)*(top-bottom)
                poly_coords.append((x,y))
            xs, ys = zip(*poly_coords)
            ax.fill(xs, ys, facecolor='lightcoral', alpha=0.5, hatch='//', edgecolor='red', linewidth=2)
    ax.set_title(f"Forest Fire Simulation - Step {step}")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(out_path, dpi=150)
    plt.close()

# Augmented topomap drawing: no legend, no axis labels/ticks.
def draw_augmented_topomap(ax, topo, extent):
    ax.clear()
    water = np.ma.masked_where(topo > 0, topo)
    land = np.ma.masked_where(topo <= 0, topo)
    terrain = plt.cm.terrain
    shifted = terrain(np.linspace(0.2,1,256))
    terrain_shifted = LinearSegmentedColormap.from_list("terrain_shifted", shifted)
    ax.imshow(land, extent=extent, cmap=terrain_shifted)
    dark_blue = ListedColormap(['#00008B'])
    ax.imshow(water, extent=extent, cmap=dark_blue)
    land_elev = topo.astype(float)
    land_elev[land_elev <= 0] = np.nan
    land_elev = np.flipud(land_elev)
    levels = np.linspace(np.nanmin(land_elev), np.nanmax(land_elev), 7)
    contours = ax.contour(land_elev, levels=levels, extent=extent, colors='black', linewidths=0.5)
    ax.clabel(contours, inline=True, fontsize=8, fmt='%1.0f')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Augmented Terrain Map")

def update_fireline_advanced(state, fireline_data, fireline_speed,
                             topo, wind_vector, wind_speed, p_base, beta, k, kernel_radius, sigma,
                             prediction_steps=3, line_thickness=1):
    rows, cols = state.shape
    pred_state = copy.deepcopy(state)
    for _ in range(prediction_steps):
        pred_state = update_state(pred_state, topo, wind_vector, wind_speed, p_base, beta, k, kernel_radius, sigma)
    pred_coords = [(c, r) for r in range(rows) for c in range(cols) if pred_state[r, c] in (1,2)]
    if len(pred_coords) < 3:
        return state, fireline_data
    hull = convex_hull(pred_coords)
    if len(hull) < 3:
        return state, fireline_data
    centroid = (np.mean([pt[0] for pt in pred_coords]), np.mean([pt[1] for pt in pred_coords]))
    dilated_hull = []
    for (x, y) in hull:
        vec = np.array([x-centroid[0], y-centroid[1]])
        norm = np.linalg.norm(vec)
        if norm == 0:
            dilated_hull.append((x,y))
        else:
            shift = (vec/norm).round().astype(int)
            dilated_point = (x + int(shift[0]), y + int(shift[1]))
            dilated_point = (max(0, min(cols-1, dilated_point[0])), max(0, min(rows-1, dilated_point[1])))
            dilated_hull.append(dilated_point)
    chain = []
    for i in range(len(dilated_hull)):
        x0, y0 = dilated_hull[i]
        x1, y1 = dilated_hull[(i+1)%len(dilated_hull)]
        chain.extend(bresenham_line(x0, y0, x1, y1))
    chain = list(dict.fromkeys(chain))
    fireline_data['chain'] = chain
    if 'index' not in fireline_data:
        fireline_data['index'] = 0
    idx = fireline_data['index']
    new_points = []
    for i in range(fireline_speed):
        if idx < len(chain):
            new_points.append(chain[idx])
            idx += 1
        else:
            break
    fireline_data['index'] = idx
    for pt in new_points:
        if not is_adjacent_to_fireline(pt, state):
            candidate = find_nearest_connected(pt, state)
            if candidate is None:
                candidate = pt
        else:
            candidate = pt
        col, row = candidate
        for dr in range(-line_thickness, line_thickness+1):
            for dc in range(-line_thickness, line_thickness+1):
                if dr**2+dc**2 <= line_thickness**2:
                    r_new = row+dr
                    c_new = col+dc
                    if 0 <= r_new < rows and 0 <= c_new < cols:
                        if state[r_new, c_new] == 0:
                            state[r_new, c_new] = 3
    return state, fireline_data

# -------------------------------
# RL Environment (unchanged)
# -------------------------------
class FirelineRLTargetEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, init_state, candidate_chain, city_blobs, safety_margin, weight_city, weight_veg):
        super(FirelineRLTargetEnv, self).__init__()
        self.grid_shape = init_state.shape
        self.state = init_state.copy()
        self.candidate_chain = candidate_chain
        self.city_blobs = city_blobs
        self.safety_margin = safety_margin
        self.weight_city = weight_city
        self.weight_veg = weight_veg
        self.action_space = spaces.Discrete(len(candidate_chain))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        self.placed = []
        self.p_base = 0.15
        self.beta = 0.5
        self.k = 2.0
        self.kernel_radius = 3
        self.sigma = 1.5
        self.fireline_data = {'chain': None, 'index': 0}
        self.wind_vector = np.array([1, 0])
        self.wind_speed = 5.0

    def step(self, action):
        candidate = self.candidate_chain[action]
        if not is_adjacent_to_fireline(candidate, self.state):
            alt = find_nearest_connected(candidate, self.state)
            if alt is not None:
                candidate = alt
        col, row = candidate
        self.state[row, col] = 3
        self.placed.append(candidate)
        
        self.state = update_state(self.state, self.state, self.wind_vector, self.wind_speed,
                                    self.p_base, self.beta, self.k, self.kernel_radius, self.sigma)
        self.state, self.fireline_data = update_fireline_advanced(
            self.state, self.fireline_data, fireline_speed=5,
            topo=self.state, wind_vector=self.wind_vector, wind_speed=self.wind_speed,
            p_base=self.p_base, beta=self.beta, k=self.k, kernel_radius=self.kernel_radius,
            sigma=self.sigma, prediction_steps=3, line_thickness=2)
        
        reward = self._compute_reward()
        terminated = not self._all_cities_safe()
        truncated = len(self.placed) >= len(self.candidate_chain)
        obs = self._get_obs()
        return obs, reward, terminated, truncated, {}
    
    def _compute_reward(self):
        rows, cols = self.state.shape
        city_cost = 0
        breach = False
        for blob in self.city_blobs:
            total = 0
            safe = 0
            for r in range(rows):
                for c in range(cols):
                    if point_in_polygon((c, r), blob):
                        total += 1
                        if self.state[r, c] not in (1, 2):
                            safe += 1
                        else:
                            breach = True
            if total > 0:
                ratio = safe / total
                if ratio < 1.0:
                    city_cost += (1.0 - ratio) * 10000
                else:
                    city_cost -= 500
        veg_cost = np.sum(self.state == 0)
        total_cost = self.weight_city * city_cost + self.weight_veg * veg_cost
        if breach:
            total_cost += 50000
        return -total_cost
    
    def _all_cities_safe(self):
        rows, cols = self.state.shape
        for blob in self.city_blobs:
            for r in range(rows):
                for c in range(cols):
                    if point_in_polygon((c, r), blob) and self.state[r, c] in (1, 2):
                        return False
        return True
    
    def _get_obs(self):
        return np.zeros(10, dtype=np.float32)
    
    def reset(self, seed=None, options=None):
        grid_shape = self.grid_shape
        self.state = np.zeros(grid_shape, dtype=int)
        num_city_polys = random.choice([1, 2, 3])
        self.city_blobs = []
        for _ in range(num_city_polys):
            city_center = (random.randint(0, grid_shape[1]-1), random.randint(0, grid_shape[0]-1))
            blob = generate_city_blob(city_center, blob_radius=3, num_points=8)
            self.city_blobs.append(blob)
        self.state = preprotect_cities(self.state, self.city_blobs, dilation_size=1)
        while True:
            start_row = random.randint(0, grid_shape[0]-1)
            start_col = random.randint(0, grid_shape[1]-1)
            if all(not point_in_polygon((start_col, start_row), blob) for blob in self.city_blobs):
                break
        self.state[start_row, start_col] = 1
        wind_direction = random.uniform(0, 360)
        wind_rad = np.deg2rad(wind_direction)
        self.wind_vector = np.array([np.cos(wind_rad), np.sin(wind_rad)])
        self.wind_speed = random.uniform(1, 10)
        self.placed = []
        self.fireline_data = {'chain': None, 'index': 0}
        return self._get_obs(), {}
    
    def render(self, mode='human'):
        print("Fireline cells placed:", len(self.placed))

def get_rl_target_env(randomize=True):
    grid_shape = (20,20)
    extent = [0, grid_shape[1], 0, grid_shape[0]]
    if randomize:
        init_state = np.zeros(grid_shape, dtype=int)
        num_city_polys = random.choice([1, 2, 3])
        city_blobs = []
        for i in range(num_city_polys):
            city_center = (random.randint(0, grid_shape[1]-1), random.randint(0, grid_shape[0]-1))
            blob = generate_city_blob(city_center, blob_radius=3, num_points=8)
            city_blobs.append(blob)
        init_state = preprotect_cities(init_state, city_blobs, dilation_size=1)
        while True:
            start_row = random.randint(0, grid_shape[0]-1)
            start_col = random.randint(0, grid_shape[1]-1)
            if all(not point_in_polygon((start_col, start_row), blob) for blob in city_blobs):
                break
        init_state[start_row, start_col] = 1
        wind_direction = random.uniform(0, 360)
        wind_rad = np.deg2rad(wind_direction)
        wind_vector = np.array([np.cos(wind_rad), np.sin(wind_rad)])
        wind_speed = random.uniform(1, 10)
    else:
        raise NotImplementedError("Interactive mode not implemented in UI version.")
    candidate_chain = ([(i,0) for i in range(grid_shape[1])] +
                       [(grid_shape[1]-1, i) for i in range(grid_shape[0])] +
                       [(i, grid_shape[0]-1) for i in range(grid_shape[1]-1, -1, -1)] +
                       [(0, i) for i in range(grid_shape[0]-1, -1, -1)])
    
    safety_margin = 3
    weight_city = 10
    weight_veg = 0.1

    print("Using randomized environment parameters for RL training:")
    print(f"City Polygons: {city_blobs}")
    print("Wildfire Start chosen randomly")
    print(f"Wind Vector: {wind_vector}, Wind Speed: {wind_speed:.2f}")
    
    env = FirelineRLTargetEnv(init_state, candidate_chain, city_blobs, safety_margin, weight_city, weight_veg)
    env.wind_vector = wind_vector
    env.wind_speed = wind_speed
    return env

# -------------------------------
# Worker Threads for Simulation, RL Training & Testing
# -------------------------------
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QPixmap

class SimulationWorker(QThread):
    frame_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(str)
    
    def __init__(self, params, topo, extent, city_blobs, fire_start, wind_points, figsize):
        super().__init__()
        self.params = params
        self.topo = topo
        self.extent = extent
        self.city_blobs = city_blobs if city_blobs else []
        self.fire_start = fire_start
        self.wind_points = wind_points
        self.figsize = figsize  # store the fixed canvas size
    def run(self):
        grid_shape = self.topo.shape
        state = np.zeros(grid_shape, dtype=int)
        if self.fire_start is None:
            init_row, init_col = grid_shape[0]//2, grid_shape[1]//2
        else:
            init_row, init_col = self.fire_start
        state[init_row, init_col] = 1
        city_blobs = self.city_blobs
        if len(self.wind_points) == 2:
            origin, tip = self.wind_points
            arrow = np.array(tip) - np.array(origin)
            norm = np.linalg.norm(arrow)
            if norm == 0:
                wind_vector = np.array([1, 0])
                wind_speed = 5.0
            else:
                wind_vector = arrow / norm
                wind_speed = norm
        else:
            wind_direction = random.uniform(0, 360)
            wind_rad = np.deg2rad(wind_direction)
            wind_vector = np.array([np.cos(wind_rad), np.sin(wind_rad)])
            wind_speed = random.uniform(1, 10)
        p_base = self.params['p_base']
        beta = self.params['beta']
        k = self.params['k']
        kernel_radius = self.params['kernel_radius']
        sigma = self.params['sigma']
        fireline_start = self.params['fireline_start']
        fireline_speed = self.params['fireline_speed']
        pred_steps = self.params['pred_steps']
        line_thickness = self.params['line_thickness']
        n_steps = int(self.params['n_steps'])
        
        fireline_data = {'chain': None, 'index': 0}
        frames = []
        frame_folder = "frames_sim"
        os.makedirs(frame_folder, exist_ok=True)
        breach_occurred = False
        for step in range(n_steps):
            state = update_state(state, self.topo, wind_vector, wind_speed, p_base, beta, k, kernel_radius, sigma)
            if step >= fireline_start:
                state, fireline_data = update_fireline_advanced(
                    state, fireline_data, fireline_speed,
                    self.topo, wind_vector, wind_speed,
                    p_base, beta, k, kernel_radius, sigma,
                    prediction_steps=pred_steps,
                    line_thickness=line_thickness
                )
            for blob in city_blobs:
                for r in range(state.shape[0]):
                    for c in range(state.shape[1]):
                        if point_in_polygon((c, r), blob) and state[r, c] in (1,2):
                            breach_occurred = True
                            break
                    if breach_occurred:
                        break
                if breach_occurred:
                    break
            frame_path = os.path.join(frame_folder, f"frame_{step:03d}.png")
            # Use fixed canvas size for frame generation:
            plot_simulation(self.topo, self.extent, state, city_blobs, step, frame_path, figsize=self.figsize)
            frames.append(frame_path)
            self.frame_signal.emit(frame_path)
            if breach_occurred or np.all(state != 1):
                break
        gif_filename = "fire_simulation.gif"
        imageio.mimsave(gif_filename, [imageio.imread(f) for f in frames], duration=0.5)
        self.finished_signal.emit(gif_filename)

class RLTrainingWorker(QThread):
    progress_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()
    def __init__(self, total_timesteps):
        super().__init__()
        self.total_timesteps = total_timesteps
    def run(self):
        try:
            self.progress_signal.emit("Starting RL training...")
            train_rl_agent(self.total_timesteps, self.progress_signal.emit)
            self.progress_signal.emit("RL training complete!")
            self.finished_signal.emit()
        except Exception as e:
            self.progress_signal.emit(f"Error: {str(e)}")

class RLTestingWorker(QThread):
    progress_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()
    def __init__(self, model_file):
        super().__init__()
        self.model_file = model_file
    def run(self):
        try:
            self.progress_signal.emit("Starting RL test episode...")
            test_rl_agent(self.model_file, self.progress_signal.emit)
            self.progress_signal.emit("RL test episode complete!")
            self.finished_signal.emit()
        except Exception as e:
            self.progress_signal.emit(f"Error: {str(e)}")

# -------------------------------
# RL Training & Testing Functions with Logging
# -------------------------------
class QtPrintTrainingCallback(BaseCallback):
    def __init__(self, log_func, verbose=0):
        super(QtPrintTrainingCallback, self).__init__(verbose)
        self.episode = 0
        self.log_func = log_func
    def _on_step(self) -> bool:
        if self.n_calls % 1000 == 0:
            self.log_func(f"Step: {self.n_calls}, Episode: {self.episode}")
        return True
    def _on_rollout_end(self) -> None:
        self.episode += 1

def train_rl_agent(total_timesteps, log_func):
    env = get_rl_target_env(randomize=True)
    check_env(env, warn=True)
    if os.path.exists(MODEL_FILE):
        log_func("Loading existing model...")
        model = DQN.load(MODEL_FILE, env=env)
    else:
        model = DQN("MlpPolicy", env, verbose=1)
    callback = QtPrintTrainingCallback(log_func)
    model.learn(total_timesteps=total_timesteps, callback=callback)
    model.save(MODEL_FILE)
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        env.render()
    log_func("Test episode finished. Check that no city was breached.")

def test_rl_agent(model_file, log_func):
    env = get_rl_target_env(randomize=True)
    model = DQN.load(model_file, env=env)
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        env.render()
        log_func("Action taken, state updated.")
    log_func("Test episode finished.")

# -------------------------------
# UI Widgets
# -------------------------------
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTextEdit, QFormLayout,
    QDoubleSpinBox, QSpinBox, QGroupBox, QSplitter, QSizePolicy
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap

# --- Simulation Widget ---
class SimulationWidget(QWidget):
    reset_program_signal = pyqtSignal()
    # New signal to notify simulation finished
    simulation_finished_signal = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        main_layout = QHBoxLayout()
        self.setLayout(main_layout)
        
        # Left panel: controls
        control_group = QGroupBox("Simulation Controls")
        form = QFormLayout()
        self.p_base_spin = QDoubleSpinBox()
        self.p_base_spin.setDecimals(3)
        self.p_base_spin.setRange(0.0, 1.0)
        self.p_base_spin.setValue(0.15)
        form.addRow("Base ignition probability:", self.p_base_spin)
        
        self.beta_spin = QDoubleSpinBox()
        self.beta_spin.setDecimals(2)
        self.beta_spin.setRange(0, 10)
        self.beta_spin.setValue(0.5)
        form.addRow("Wind sensitivity factor β:", self.beta_spin)
        
        self.k_spin = QDoubleSpinBox()
        self.k_spin.setDecimals(2)
        self.k_spin.setRange(0, 10)
        self.k_spin.setValue(2.0)
        form.addRow("Slope sensitivity factor k:", self.k_spin)
        
        self.kernel_radius_spin = QSpinBox()
        self.kernel_radius_spin.setRange(1, 10)
        self.kernel_radius_spin.setValue(3)
        form.addRow("Kernel radius:", self.kernel_radius_spin)
        
        self.sigma_spin = QDoubleSpinBox()
        self.sigma_spin.setDecimals(2)
        self.sigma_spin.setRange(0.1, 10)
        self.sigma_spin.setValue(1.5)
        form.addRow("Sigma for Gaussian kernel:", self.sigma_spin)
        
        self.fireline_start_spin = QSpinBox()
        self.fireline_start_spin.setRange(0, 200)
        self.fireline_start_spin.setValue(10)
        form.addRow("Fireline start step:", self.fireline_start_spin)
        
        self.fireline_speed_spin = QSpinBox()
        self.fireline_speed_spin.setRange(1, 20)
        self.fireline_speed_spin.setValue(5)
        form.addRow("Fireline speed (cells/step):", self.fireline_speed_spin)
        
        self.pred_steps_spin = QSpinBox()
        self.pred_steps_spin.setRange(1, 10)
        self.pred_steps_spin.setValue(3)
        form.addRow("Prediction steps:", self.pred_steps_spin)
        
        self.line_thickness_spin = QSpinBox()
        self.line_thickness_spin.setRange(1, 10)
        self.line_thickness_spin.setValue(2)
        form.addRow("Fireline thickness:", self.line_thickness_spin)
        
        self.n_steps_spin = QSpinBox()
        self.n_steps_spin.setRange(10, 500)
        self.n_steps_spin.setValue(150)
        form.addRow("Total simulation steps:", self.n_steps_spin)
        
        # Selection buttons
        self.city_btn = QPushButton("Select Protected Areas")
        self.fire_start_btn = QPushButton("Select Wildfire Start")
        self.wind_btn = QPushButton("Select Wind")
        self.finish_sel_btn = QPushButton("Finish Selection")
        self.reset_sel_btn = QPushButton("Reset Program")
        sel_layout = QVBoxLayout()
        sel_layout.addWidget(self.city_btn)
        sel_layout.addWidget(self.fire_start_btn)
        sel_layout.addWidget(self.wind_btn)
        sel_layout.addWidget(self.finish_sel_btn)
        sel_layout.addWidget(self.reset_sel_btn)
        form.addRow(sel_layout)
        
        self.run_sim_btn = QPushButton("Run Simulation")
        form.addRow(self.run_sim_btn)
        
        control_group.setLayout(form)
        main_layout.addWidget(control_group, 1)
        
        # Right panel: embedded matplotlib canvas
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure
        self.canvas = FigureCanvas(Figure())
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.ax = self.canvas.figure.add_subplot(111)
        main_layout.addWidget(self.canvas, 2)
        
        # Store the original figure size for consistency
        self.original_figsize = self.canvas.figure.get_size_inches().copy()
        
        # Variables for interactive selection
        self.selection_mode = None   # "city", "fire_start", "wind"
        self.current_selection = []  # temporary list of (x,y) in map coordinates
        self.city_blobs = []         # saved protected area polygons (grid coordinates)
        self.fire_start = None       # saved fire start (grid coordinates)
        self.wind_points = []        # saved wind points (map coordinates)
        
        # Load elevation map from DEM
        dem_file = download_dem_opentopography()
        with rasterio.open(dem_file) as ds:
            elevation = ds.read(1)
            bounds = ds.bounds
            self.extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
        scale = 10
        self.topo = elevation[::scale, ::scale]
        draw_augmented_topomap(self.ax, self.topo, self.extent)
        self.canvas.draw()
        
        # Connect canvas and buttons
        self.canvas.mpl_connect('button_press_event', self.on_canvas_click)
        self.city_btn.clicked.connect(lambda: self.set_selection_mode("city"))
        self.fire_start_btn.clicked.connect(lambda: self.set_selection_mode("fire_start"))
        self.wind_btn.clicked.connect(lambda: self.set_selection_mode("wind"))
        self.finish_sel_btn.clicked.connect(self.finish_selection)
        self.reset_sel_btn.clicked.connect(self.reset_program)
        self.run_sim_btn.clicked.connect(self.start_simulation)
        
        self.sim_worker = None

    def resizeEvent(self, event):
        # Update the figure size to match the widget size.
        size = self.size()
        dpi = self.canvas.figure.get_dpi()
        self.canvas.figure.set_size_inches(size.width()/dpi, size.height()/dpi)
        self.canvas.draw()
        super().resizeEvent(event)

    def set_selection_mode(self, mode):
        self.selection_mode = mode
        self.current_selection = []
        print(f"Selection mode set to: {mode}")

    def on_canvas_click(self, event):
        if event.inaxes != self.ax:
            return
        if self.selection_mode is None:
            return
        self.current_selection.append((event.xdata, event.ydata))
        self.ax.plot(event.xdata, event.ydata, 'ro')
        self.canvas.draw()

    def finish_selection(self):
        rows, cols = self.topo.shape
        left, right, bottom, top = self.extent
        if self.selection_mode == "city":
            polygon = []
            for x, y in self.current_selection:
                col = int((x - left) / (right - left) * cols)
                row = int((top - y) / (top - bottom) * rows)
                polygon.append((col, row))
            if polygon and polygon[0] != polygon[-1]:
                polygon.append(polygon[0])
            self.city_blobs.append(polygon)
            print("Added protected area:", polygon)
        elif self.selection_mode == "fire_start":
            if self.current_selection:
                x, y = self.current_selection[0]
                col = int((x - left) / (right - left) * cols)
                row = int((top - y) / (top - bottom) * rows)
                self.fire_start = (row, col)
                print("Wildfire start selected at:", self.fire_start)
        elif self.selection_mode == "wind":
            if len(self.current_selection) >= 2:
                self.wind_points = self.current_selection[:2]
                print("Wind points selected:", self.wind_points)
        self.current_selection = []
        self.selection_mode = None
        draw_augmented_topomap(self.ax, self.topo, self.extent)
        # Overlay saved selections:
        for poly in self.city_blobs:
            xs = []
            ys = []
            for (c, r) in poly:
                x = left + (c/cols) * (right - left)
                y = top - (r/rows) * (top - bottom)
                xs.append(x)
                ys.append(y)
            self.ax.plot(xs, ys, 'r--')
        if self.fire_start is not None:
            r, c = self.fire_start
            x = left + (c/cols) * (right - left)
            y = top - (r/rows) * (top - bottom)
            self.ax.plot(x, y, marker='o', markersize=8, color='red')
        if self.wind_points and len(self.wind_points) >= 2:
            origin = self.wind_points[0]
            tip = self.wind_points[1]
            self.ax.arrow(origin[0], origin[1], tip[0]-origin[0], tip[1]-origin[1],
                          head_width=0.2, head_length=0.3, fc='cyan', ec='cyan')
        self.canvas.draw()

    def reset_program(self):
        # If a simulation is running, terminate it
        if self.sim_worker is not None and self.sim_worker.isRunning():
            self.sim_worker.terminate()
            self.sim_worker = None

        # Reset selections:
        self.city_blobs = []
        self.fire_start = None
        self.wind_points = []
        self.current_selection = []
        self.selection_mode = None
        # Reset spin boxes to their default values:
        self.p_base_spin.setValue(0.15)
        self.beta_spin.setValue(0.5)
        self.k_spin.setValue(2.0)
        self.kernel_radius_spin.setValue(3)
        self.sigma_spin.setValue(1.5)
        self.fireline_start_spin.setValue(10)
        self.fireline_speed_spin.setValue(5)
        self.pred_steps_spin.setValue(3)
        self.line_thickness_spin.setValue(2)
        self.n_steps_spin.setValue(150)
        # Restore the original canvas figure size and clear any simulation frames
        self.canvas.figure.set_size_inches(self.original_figsize)
        self.canvas.figure.clf()
        self.ax = self.canvas.figure.add_subplot(111)
        draw_augmented_topomap(self.ax, self.topo, self.extent)
        # Reset zoom to full extent:
        left, right, bottom, top = self.extent
        self.ax.set_xlim(left, right)
        self.ax.set_ylim(bottom, top)
        self.canvas.draw()
        print("Program has been reset to the beginning.")
        self.reset_program_signal.emit()

    def start_simulation(self):
        self.run_sim_btn.setEnabled(False)
        
        # Zoom into the map before simulation starts.
        # Here we use a fixed zoom factor (e.g., 2x zoom into the center of the extent).
        left, right, bottom, top = self.extent
        center_x = (left + right) / 2
        center_y = (top + bottom) / 2
        zoom_factor = 2  # Change this factor to control how much you zoom in.
        half_width = (right - left) / (2 * zoom_factor)
        half_height = (top - bottom) / (2 * zoom_factor)
        self.ax.set_xlim(center_x - half_width, center_x + half_width)
        self.ax.set_ylim(center_y - half_height, center_y + half_height)
        self.canvas.draw()
        
        params = {
            'p_base': self.p_base_spin.value(),
            'beta': self.beta_spin.value(),
            'k': self.k_spin.value(),
            'kernel_radius': self.kernel_radius_spin.value(),
            'sigma': self.sigma_spin.value(),
            'fireline_start': self.fireline_start_spin.value(),
            'fireline_speed': self.fireline_speed_spin.value(),
            'pred_steps': self.pred_steps_spin.value(),
            'line_thickness': self.line_thickness_spin.value(),
            'n_steps': self.n_steps_spin.value()
        }
        # Use the stored original canvas size for consistent simulation frame size
        canvas_size = self.original_figsize
        self.sim_worker = SimulationWorker(params, self.topo, self.extent, self.city_blobs,
                                           self.fire_start, self.wind_points, figsize=canvas_size)
        self.sim_worker.frame_signal.connect(self.update_sim_frame)
        self.sim_worker.finished_signal.connect(self.simulation_finished)
        self.sim_worker.start()

    def update_sim_frame(self, frame_file):
        self.canvas.figure.clf()
        self.ax = self.canvas.figure.add_subplot(111)
        img = plt.imread(frame_file)
        self.ax.imshow(img)
        self.ax.axis('off')
        self.canvas.draw()

    def simulation_finished(self, gif_file):
        print("Simulation complete; GIF saved as:", gif_file)
        self.run_sim_btn.setEnabled(True)
        # Emit a signal to notify the Model Selection widget
        self.simulation_finished_signal.emit(gif_file)

# --- RL Training Widget ---
class RLTrainingWidget(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        self.setLayout(layout)
        form = QFormLayout()
        self.timesteps_spin = QSpinBox()
        self.timesteps_spin.setRange(1000, 1000000)
        self.timesteps_spin.setValue(50000)
        form.addRow("Training Timesteps:", self.timesteps_spin)
        layout.addLayout(form)
        self.train_btn = QPushButton("Start RL Training")
        layout.addWidget(self.train_btn)
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        layout.addWidget(self.log_area)
        self.train_btn.clicked.connect(self.start_training)
        self.train_worker = None
    def start_training(self):
        self.train_btn.setEnabled(False)
        total = self.timesteps_spin.value()
        self.train_worker = RLTrainingWorker(total)
        self.train_worker.progress_signal.connect(self.append_log)
        self.train_worker.finished_signal.connect(self.training_finished)
        self.train_worker.start()
    def append_log(self, msg):
        self.log_area.append(msg)
    def training_finished(self):
        self.append_log("RL training complete!")
        self.train_btn.setEnabled(True)

# --- Model Selection Widget ---
class ModelSelectionWidget(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        self.setLayout(layout)
        btn_layout = QHBoxLayout()
        self.load_btn = QPushButton("Load Pretrained Model")
        self.export_btn = QPushButton("Export Current Model")
        self.test_btn = QPushButton("Test RL Agent")
        self.test_btn.setEnabled(False)
        btn_layout.addWidget(self.load_btn)
        btn_layout.addWidget(self.export_btn)
        btn_layout.addWidget(self.test_btn)
        layout.addLayout(btn_layout)
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        layout.addWidget(self.log_area)
        self.load_btn.clicked.connect(self.load_model)
        self.export_btn.clicked.connect(self.export_model)
        self.test_btn.clicked.connect(self.test_model)
        self.model_file = None
        self.test_worker = None
    def load_model(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Select RL Model", "", "Zip Files (*.zip)")
        if fname:
            self.model_file = fname
            self.log_area.append(f"Loaded model: {fname}")
            self.test_btn.setEnabled(True)
    def export_model(self):
        if not os.path.exists(MODEL_FILE):
            self.log_area.append("No current model to export.")
            return
        dest, _ = QFileDialog.getSaveFileName(self, "Export RL Model", "exported_model.zip", "Zip Files (*.zip)")
        if dest:
            import shutil
            shutil.copy(MODEL_FILE, dest)
            self.log_area.append(f"Model exported to {dest}")
    def test_model(self):
        if not self.model_file:
            self.log_area.append("Please load a model first.")
            return
        self.test_btn.setEnabled(False)
        self.test_worker = RLTestingWorker(self.model_file)
        self.test_worker.progress_signal.connect(self.append_log)
        self.test_worker.finished_signal.connect(self.test_finished)
        self.test_worker.start()
    def append_log(self, msg):
        self.log_area.append(msg)
    def test_finished(self):
        self.log_area.append("RL test episode complete!")
        self.test_btn.setEnabled(True)
    # New method to receive simulation finished notification.
    def notify_simulation_complete(self, gif_file):
        self.log_area.append(f"Simulation complete; GIF saved as: {gif_file}.\nPlease select a download location using the Export button.")

# -------------------------------
# Main Window (Combined Full-Screen UI)
# -------------------------------
from PyQt5.QtWidgets import QSplitter

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fireline Simulation & RL")
        self.showMaximized()
        
        # Top: Simulation Panel
        self.simulation_widget = SimulationWidget()
        self.simulation_widget.reset_program_signal.connect(self.reset_program)
        
        # Bottom: RL Training and Model Selection side-by-side
        self.rl_train_widget = RLTrainingWidget()
        self.model_sel_widget = ModelSelectionWidget()
        # Connect the simulation finished signal to the Model Selection widget's notifier
        self.simulation_widget.simulation_finished_signal.connect(self.model_sel_widget.notify_simulation_complete)
        
        bottom_split = QSplitter(Qt.Horizontal)
        bottom_split.addWidget(self.rl_train_widget)
        bottom_split.addWidget(self.model_sel_widget)
        bottom_split.setSizes([400, 400])
        
        main_split = QSplitter(Qt.Vertical)
        main_split.addWidget(self.simulation_widget)
        main_split.addWidget(bottom_split)
        main_split.setSizes([600, 300])
        self.setCentralWidget(main_split)
    
    def reset_program(self):
        # Example: clear log areas of RL widgets or any other resets needed.
        self.rl_train_widget.log_area.clear()
        self.model_sel_widget.log_area.clear()
        print("Other UI elements have also been reset.")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec_())
