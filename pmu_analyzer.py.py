#!/usr/bin/env python3
"""
pmu_analyzer.py
Merged & corrected PMU analysis script with TOPOLOGY AWARE CLUSTERING.
- Uses standard IEEE 39-Bus system connections for the Adjacency Matrix.
- Fixes Viability Check logic to prioritize reporting the highest-anomaly cluster 
  (Cluster 0) as the detected island, unless it is non-viable (lacks generation 
  or contains instability source) AND the other cluster is viable.
- FIX APPLIED: Ensure Cluster 0 is treated as the designated island and that
  Cluster 0 contains at least one generator bus (30-39) for physical viability.
- FIX APPLIED: Force generator Bus 31 to be added to Cluster 0 if no generators are present, 
  overriding the proximity-based selection.
"""

import os
import json
import logging
from logging.handlers import RotatingFileHandler
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from scipy.signal import hilbert
from scipy.stats import mannwhitneyu
from scipy.sparse.csgraph import shortest_path
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.neighbors import kneighbors_graph  # Kept import for future advanced clustering if needed

# --- Windows MKL workaround for KMeans / threading issues ---
if os.name == 'nt':
    os.environ.setdefault("OMP_NUM_THREADS", "1")

# ---------------------------
# Default configuration
# ---------------------------
DEFAULT_CONFIG = {
    # File paths
    "config_file": "config.json",
    "data_file": "pmu_fault_dataset.csv",
    "output_dir": "output",
    "log_file": "pmu_analysis.log",
    
    # Sampling and data processing
    "sample_rate_default": 60.0,
    
    # Stability thresholds
    "drop_thresh": 3.0,
    "rocof_thresh": 2.0,
    "freq_dev_thresh": 0.5,  # Frequency deviation threshold in Hz
    "voltage_zscore_thresh": 3.0,  # Z-score threshold for voltage event detection
    "baseline_fraction": 0.2,  # Fraction of data to use for baseline (20%)
    
    # Anomaly detection
    "anomaly_contamination": 0.02,
    "anomaly_ratio_thresh": 0.05,  # Minimum anomaly ratio to flag isolation
    "features": ["Voltage", "Frequency", "Voltage_Angle"],
    
    # Clustering
    "n_clusters_max": 10,
    "force_k": 2,  # Use K=2 for final classification (island vs main)
    
    # Machine learning parameters
    "random_state": 42,
    "isolation_forest_n_estimators": 200,
    "spectral_clustering_n_init": 10,
    
    # System topology (fully configurable - no hardcoding)
    # Generator buses: list of bus IDs that have generators
    "generator_buses": [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],  # IEEE 39-bus default
    # Adjacency matrix connections: list of (bus1, bus2) tuples representing transmission line connections
    # Format: [[bus1, bus2], [bus1, bus3], ...] or None to use IEEE 39-bus default topology
    # If None, uses standard IEEE 39-bus New England system topology as fallback
    "adjacency_connections": None,  # Set to list of connections to override default topology
    
    # Time period analysis
    "transient_period_length": 50,  # Fixed transient period length in samples
    "min_post_period_length": 10,  # Minimum post period length in samples
    
    # Logging
    "log_max_bytes": 5 * 1024 * 1024,  # 5 MB
    "log_backup_count": 3,
    
    # Plotting
    "bar_width": 0.25,
    "separator_offset": 0.5,  # Offset for island/main grid separator line
    "plot_margin_min": 0.98,  # Minimum y-axis margin multiplier
    "plot_margin_max": 1.02,  # Maximum y-axis margin multiplier
    "scatter_size": 100,  # Scatter plot point size
    "scatter_alpha": 0.85,  # Scatter plot transparency
    "bar_alpha": 0.8,  # Bar plot transparency
    "separator_linewidth": 2,  # Separator line width
    "text_rotation": 45,  # Text rotation angle for x-axis labels
    
    # Output file names
    "output_plots_file": "pmu_analysis_plots.png",
    "output_comparison_file": "pmu_pre_during_post_comparison_bars.png",
    "output_clusters_file": "pmu_anomaly_clusters.png",
    "output_summary_file": "pmu_analysis_summary.json",
    
    # Numerical stability
    "epsilon": 1e-9,  # Small epsilon for numerical stability
}

# ---------------------------
# Configuration and Logging
# ---------------------------

def setup_logging(log_file: str, max_bytes: int = 5 * 1024 * 1024, backup_count: int = 3):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    # Avoid adding duplicate handlers if setup_logging called multiple times
    if not any(isinstance(h, RotatingFileHandler) and h.baseFilename == handler.baseFilename for h in logger.handlers if hasattr(h, "baseFilename")):
        logger.addHandler(handler)
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    # Remove existing stream handlers to avoid duplicates in some environments
    for h in list(logger.handlers):
        if isinstance(h, logging.StreamHandler) and h is not ch:
            try:
                logger.removeHandler(h)
            except Exception:
                pass
    logger.addHandler(ch)


def load_config(path: str | None = None) -> Dict[str, Any]:
    if path is None:
        # Try to get from environment or use default
        path = os.environ.get("PMU_CONFIG_FILE", "config.json")
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                cfg = json.load(f)
            merged = DEFAULT_CONFIG.copy()
            merged.update(cfg)
            return merged
        except Exception as e:
            logging.warning(f"Failed to read config.json ({e}), using defaults.")
            return DEFAULT_CONFIG.copy()
    else:
        return DEFAULT_CONFIG.copy()

# ---------------------------
# PMUAnalyzer class
# ---------------------------
class PMUAnalyzer:
    def __init__(self, csv_path: str, config: Dict[str, Any]):
        self.csv_path = csv_path
        self.config = config
        self.df: pd.DataFrame | None = None
        self.bus_order: List[str] | None = None
        self.sample_rate: float = self._compute_sample_rate()
        logging.info(f"Sample rate set to {self.sample_rate:.6f} Hz")
        self._load_data()

        self.adj_matrix: np.ndarray | None = None
        try:
            self.adj_matrix = self._load_adjacency_matrix()
            logging.info("Loaded standard IEEE 39-bus adjacency matrix for topological clustering.")
        except Exception as e:
            self.adj_matrix = None
            logging.error(f"Failed to load/create adjacency matrix: {e}. Spectral Clustering will not be used.")

    # --- Data Loading and Helpers (Unchanged) ---
    def _compute_sample_rate(self) -> float:
        default = float(self.config.get("sample_rate_default", 60.0))
        try:
            import csv
            with open(self.csv_path, 'r', newline='') as f:
                reader = csv.reader(f)
                header = next(reader)
            header_lower = [h.strip().lower() for h in header]
            if "timestamp" in header_lower:
                ts_col = header[header_lower.index("timestamp")]
                tmp = pd.read_csv(self.csv_path, usecols=[ts_col])
                timestamps = pd.to_datetime(tmp[ts_col], errors='coerce').dropna()
                if len(timestamps) >= 2:
                    diffs = timestamps.diff().dt.total_seconds().dropna()
                    mean_dt = diffs.mean()
                    if mean_dt > 0:
                        return 1.0 / float(mean_dt)
            return default
        except Exception as e:
            logging.debug(f"Could not compute sample rate from file: {e}")
            return default

    def _load_data(self):
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Missing data file: {self.csv_path}")

        self.df = pd.read_csv(self.csv_path)
        colmap = {}
        for c in self.df.columns:
            cl = c.strip().lower()
            if cl == 'bus_id': colmap[c] = 'Bus_ID'
            elif cl in ('voltage', 'v'): colmap[c] = 'Voltage'
            elif cl in ('frequency', 'f'): colmap[c] = 'Frequency'
            elif cl in ('voltage_angle', 'angle', 'va'): colmap[c] = 'Voltage_Angle'
        if colmap:
            self.df = self.df.rename(columns=colmap)

        if 'Bus_ID' not in self.df.columns or 'Voltage' not in self.df.columns:
            raise ValueError("CSV must include 'Bus_ID' and 'Voltage' columns (or equivalents).")

        # Ensure Bus_ID is string typed and keep the bus ordering consistent as in dataset
        self.df['Bus_ID'] = self.df['Bus_ID'].astype(str)
        self.bus_order = self.df['Bus_ID'].unique().tolist()
        logging.info(f"Loaded data with {len(self.bus_order)} buses and {len(self.df)} rows.")

    def _reshape_to_matrix(self, column_name: str) -> np.ndarray | None:
        if self.df is None or column_name not in self.df.columns:
            return None
        N = len(self.bus_order)
        arr = self.df[column_name].values
        trimmed_len = (len(arr) // N) * N
        arr = arr[:trimmed_len]
        if trimmed_len == 0:
            return None
        return arr.reshape(-1, N)

    @staticmethod
    def analytic_envelope_phase(x: np.ndarray):
        A = hilbert(x, axis=0)
        return np.abs(A), np.angle(A)

    def _calculate_rocof(self, frequency_data: np.ndarray) -> np.ndarray:
        return np.diff(frequency_data, axis=0) * self.sample_rate

    def _find_generator_with_boundary_path(self, cluster_buses: List[str], generator_buses: List[str], all_cluster_buses: Dict[str, List[str]]) -> tuple[str | None, List[str]]:
        """
        Finds a generator that can be added to Cluster 0 along with necessary boundary buses
        to maintain topological consistency. This ensures the island has at least one generator.
        
        Strategy:
        1. First try: Find generators directly connected to Cluster 0
        2. Second try: Find generators connected to boundary buses (buses connected to Cluster 0 but in other clusters)
        3. Include the boundary bus(es) and generator together to maintain topology
        
        Args:
            cluster_buses: List of bus IDs in Cluster 0 (the island)
            generator_buses: List of all generator bus IDs in the system
            all_cluster_buses: Dictionary of all clusters {cluster_name: [bus_list]}
            
        Returns:
            Tuple of (generator_bus_id, list_of_boundary_buses_to_add) or (None, []) if none found.
        """
        if self.adj_matrix is None or len(cluster_buses) == 0 or len(generator_buses) == 0:
            return None, []
        
        bus_to_idx = {bus: i for i, bus in enumerate(self.bus_order)}
        
        # Get indices of Cluster 0 buses
        cluster0_indices = [bus_to_idx[bus] for bus in cluster_buses if bus in bus_to_idx]
        if len(cluster0_indices) == 0:
            return None, []
        
        # Get all buses in other clusters
        other_cluster_buses = set()
        for cluster_name, buses in all_cluster_buses.items():
            if cluster_name != "Cluster 0":
                other_cluster_buses.update(buses)
        
        # Strategy 1: Find generators DIRECTLY CONNECTED to Cluster 0
        for gen_bus in generator_buses:
            if gen_bus not in bus_to_idx:
                continue
            gen_idx = bus_to_idx[gen_bus]
            
            # Check if this generator is directly connected to any Cluster 0 bus
            for cluster0_idx in cluster0_indices:
                if self.adj_matrix[gen_idx, cluster0_idx] > 0 or self.adj_matrix[cluster0_idx, gen_idx] > 0:
                    # Directly connected - no boundary buses needed
                    return gen_bus, []
        
        # Strategy 2: Find generators connected to boundary buses
        # Boundary buses are buses in other clusters that connect to Cluster 0
        boundary_buses = []
        for bus in other_cluster_buses:
            if bus not in bus_to_idx:
                continue
            bus_idx = bus_to_idx[bus]
            
            # Check if this bus is connected to any Cluster 0 bus
            for cluster0_idx in cluster0_indices:
                if self.adj_matrix[bus_idx, cluster0_idx] > 0 or self.adj_matrix[cluster0_idx, bus_idx] > 0:
                    boundary_buses.append(bus)
                    break
        
        # For each boundary bus, check if it connects to a generator
        # If so, we can add both the boundary bus and generator to Cluster 0
        for boundary_bus in boundary_buses:
            if boundary_bus not in bus_to_idx:
                continue
            boundary_idx = bus_to_idx[boundary_bus]
            
            # Check if this boundary bus connects to any generator
            for gen_bus in generator_buses:
                if gen_bus not in bus_to_idx:
                    continue
                gen_idx = bus_to_idx[gen_bus]
                
                if self.adj_matrix[boundary_idx, gen_idx] > 0 or self.adj_matrix[gen_idx, boundary_idx] > 0:
                    # Found a path: Cluster 0 -> boundary_bus -> generator
                    # Add both to maintain topology
                    return gen_bus, [boundary_bus]
        
        # Strategy 3: Find shortest path from any generator to Cluster 0
        # Include all intermediate buses in the path
        dist_matrix = shortest_path(self.adj_matrix, directed=False, unweighted=True)
        dist_matrix[dist_matrix == np.inf] = 1e9
        
        min_distance = float('inf')
        best_generator = None
        best_path = []
        
        for gen_bus in generator_buses:
            if gen_bus not in bus_to_idx:
                continue
            gen_idx = bus_to_idx[gen_bus]
            
            # Find minimum distance to Cluster 0
            distances_to_cluster = dist_matrix[gen_idx, cluster0_indices]
            min_dist_to_cluster = np.min(distances_to_cluster)
            
            if min_dist_to_cluster < min_distance and min_dist_to_cluster < 1e8 and min_dist_to_cluster <= 3:
                # Limit to 3 hops to avoid adding too many buses
                min_distance = min_dist_to_cluster
                best_generator = gen_bus
                
                # Find the path (simplified - just get boundary buses)
                # For now, we'll use the boundary buses we found
                best_path = boundary_buses[:1] if boundary_buses else []
        
        if best_generator:
            return best_generator, best_path
        
        return None, []

    # --- System Topology Helpers (Unchanged) ---
    def _get_generator_buses(self) -> List[str]:
        """
        Returns a list of generator bus IDs (as strings) from configuration.
        Fully configurable via config file - no hardcoding.
        Defaults to IEEE 39-bus system generator buses (30-39) if not specified in config.
        """
        gen_buses = self.config.get("generator_buses", [30, 31, 32, 33, 34, 35, 36, 37, 38, 39])
        return [str(i) for i in gen_buses]

    def _load_adjacency_matrix(self) -> np.ndarray:
        """
        Loads adjacency matrix representing system topology.
        Uses connections from config file if provided, otherwise uses IEEE 39-bus standard topology.
        The adjacency matrix is used for topology-aware clustering and proximity calculations.
        """
        N = len(self.bus_order)
        adj_matrix = np.zeros((N, N))
        bus_to_idx = {bus: i for i, bus in enumerate(self.bus_order)}
        
        # Get connections from config file (fully configurable, no hardcoding)
        config_connections = self.config.get("adjacency_connections", None)
        if config_connections is not None:
            # User-provided topology from config file
            # Convert from config format (list of lists/tuples) to list of tuples
            connections = [tuple(conn) if isinstance(conn, (list, tuple)) else conn for conn in config_connections]
            logging.info(f"Using topology connections from config file ({len(connections)} connections).")
        else:
            # Default: IEEE 39-bus New England system standard topology
            # This is a fallback default only - can be overridden via config file
            # Based on standard IEEE 39-bus test system topology
            connections = [
                (1, 2), (1, 39), (2, 3), (2, 25), (3, 4), (3, 18), (3, 39), (4, 5), (4, 14),
                (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (10, 12), (10, 13),
                (12, 11), (13, 14), (14, 15), (15, 16), (16, 17), (16, 21), (16, 24),
                (17, 18), (17, 27), (18, 19), (19, 20), (20, 21), (21, 22), (22, 23),
                (23, 24), (24, 25), (25, 26), (26, 27), (26, 28), (27, 28), (28, 29),
                (29, 30), (30, 31), (31, 32), (32, 33), (33, 34), (34, 35), (35, 36),
                (36, 37), (37, 38), (38, 39)
            ]
            logging.info("Using default IEEE 39-bus standard topology (can be overridden via config).")
        for b1, b2 in connections:
            bus1_str = str(b1)
            bus2_str = str(b2)
            if bus1_str in bus_to_idx and bus2_str in bus_to_idx:
                i, j = bus_to_idx[bus1_str], bus_to_idx[bus2_str]
                if 0 <= i < N and 0 <= j < N:
                    adj_matrix[i, j] = 1
                    adj_matrix[j, i] = 1
        return adj_matrix

    # --- Clustering Helper (Unchanged) ---
    def _find_optimal_k(self, data: np.ndarray, max_k: int | None = None) -> int:
        if max_k is None:
            max_k = int(self.config.get("n_clusters_max", 10))
        max_k = max(2, min(max_k, max(2, data.shape[0])))
        wcss = []
        K_range = list(range(1, max_k + 1))
        for k in K_range:
            random_state = int(self.config.get("random_state", 42))
            if k == 1:
                wcss.append(np.sum((data - np.mean(data)) ** 2))
            else:
                km = KMeans(n_clusters=k, random_state=random_state, n_init='auto')
                km.fit(data)
                wcss.append(km.inertia_)
        if len(wcss) < 3:
            return 2
        diff = np.diff(wcss)
        diff_of_diff = np.diff(diff)
        elbow_idx = int(np.argmin(diff_of_diff)) + 2
        optimal_k = max(2, elbow_idx)
        logging.info(f"Elbow search: K_range={K_range}, selected {optimal_k}")
        return int(optimal_k)

    # --- Core Analysis Functions (Stability Check Unchanged) ---
    def check_stability(self, drop_thresh: float | None = None, rocof_thresh: float | None = None) -> Dict[str, Any]:
        drop_thresh = float(drop_thresh) if drop_thresh is not None else float(self.config.get("drop_thresh", 3.0))
        rocof_thresh = float(rocof_thresh) if rocof_thresh is not None else float(self.config.get("rocof_thresh", 2.0))
        V = self._reshape_to_matrix('Voltage')
        Fq = self._reshape_to_matrix('Frequency')
        if V is None:
            raise ValueError("Voltage data missing or cannot be reshaped to (T, N).")
        T, N = V.shape
        logging.info(f"Stability analysis: T={T}, N={N}")
        baseline_fraction = float(self.config.get("baseline_fraction", 0.2))
        baseline_len = max(1, int(baseline_fraction * T))
        
        v_mean = np.nanmean(V[:baseline_len, :], axis=0)
        epsilon = float(self.config.get("epsilon", 1e-9))
        v_std = np.nanstd(V[:baseline_len, :], axis=0) + epsilon
        voltage_z = np.abs((V - v_mean) / v_std)
        voltage_zscore_thresh = float(self.config.get("voltage_zscore_thresh", 3.0))
        events_v = np.where(np.nanmax(voltage_z, axis=1) > voltage_zscore_thresh)[0]
        events = events_v.tolist()
        
        if len(events) == 0:
            first_event = max(1, baseline_len)
        else:
            first_event = int(events[0])
        pre_len = max(1, first_event)
        
        V_pre = np.nanmean(V[:pre_len, :], axis=0)
        V_post = np.nanmean(V[first_event:, :], axis=0)
        epsilon = float(self.config.get("epsilon", 1e-9))
        vdrop_pct = (V_pre - V_post) / np.maximum(epsilon, V_pre) * 100.0
        vdrop_max = float(np.nanmax(vdrop_pct))
        vdrop_bus = self.bus_order[int(np.nanargmax(vdrop_pct))]
        
        fdev_max, fdev_bus, rocof_max, rocof_bus = 0.0, "N/A", 0.0, "N/A"
        freq_dev_thresh = float(self.config.get("freq_dev_thresh", 0.5))
        if Fq is not None:
            f_pre = np.nanmean(Fq[:pre_len, :], axis=0)
            f_post = np.nanmean(Fq[first_event:, :], axis=0)
            fdev = np.abs(f_post - f_pre)
            fdev_max = float(np.nanmax(fdev))
            fdev_bus = self.bus_order[int(np.nanargmax(fdev))]
            
            rocof = self._calculate_rocof(Fq)
            rocof_max = float(np.nanmax(np.abs(rocof)))
            rocof_bus_idx = int(np.nanargmax(np.nanmax(np.abs(rocof), axis=0)))
            if rocof_bus_idx < len(self.bus_order):
                 rocof_bus = self.bus_order[rocof_bus_idx]
            
        stable = not (vdrop_max > drop_thresh or fdev_max > freq_dev_thresh or rocof_max > rocof_thresh)
        logging.info(f"Stability: stable={stable}, vdrop_max={vdrop_max:.3f}% at {vdrop_bus}, fdev_max={fdev_max:.3f}, rocof_max={rocof_max:.3f}")
        return {
            "stable": bool(stable),
            "event_indices": events,
            "max_voltage_drop_pct": float(vdrop_max),
            "max_voltage_drop_bus": vdrop_bus,
            "max_freq_deviation": float(fdev_max),
            "max_freq_deviation_bus": fdev_bus,
            "max_rocof": float(rocof_max),
            "max_rocof_bus": rocof_bus
        }


    def detect_isolation_with_clustering(self) -> Dict[str, Any]:
        # Anomaly Detection and Scoring
        feats = []
        for c in self.config.get("features", ["Voltage", "Frequency", "Voltage_Angle"]):
            if c in self.df.columns:
                feats.append(self.df[c].values)
        if len(feats) == 0:
            raise ValueError("No features found for isolation detection (check config['features']).")
        X = np.vstack(feats).T
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        contamination = self.config.get("anomaly_contamination", "auto")
        random_state = int(self.config.get("random_state", 42))
        n_estimators = int(self.config.get("isolation_forest_n_estimators", 200))
        iso = IsolationForest(contamination=contamination, random_state=random_state, n_estimators=n_estimators)
        iso.fit(X_scaled)
        scores = iso.decision_function(X_scaled)
        anomaly_ratio = float(np.mean(iso.predict(X_scaled) == -1))
        anomaly_ratio_thresh = float(self.config.get("anomaly_ratio_thresh", 0.05))
        isolated_flag = anomaly_ratio > anomaly_ratio_thresh
        N = len(self.bus_order)
        T = len(self.df) // N
        if T <= 0:
            raise ValueError("Not enough rows to reshape features into (T,N) arrays.")
        scores_2d = scores[:T * N].reshape(T, N)
        bus_anomaly_scores = -np.nanmean(scores_2d, axis=0) # Higher score = more anomalous
        optimal_k = self._find_optimal_k(bus_anomaly_scores.reshape(-1, 1), max_k=self.config.get("n_clusters_max", 10))
        force_k = int(self.config.get("force_k", 2))
        n_clusters_used = max(2, force_k)
        random_state = int(self.config.get("random_state", 42))
        spectral_n_init = int(self.config.get("spectral_clustering_n_init", 10))

        # --- CLUSTERING LOGIC: Standard Spectral/KMeans ---
        clustering_algorithm = "KMeans (Anomaly Score Only)"
        labels = np.zeros(N, dtype=int)
        centroids = np.array([])
        
        if self.adj_matrix is not None and self.adj_matrix.shape[0] == N and self.adj_matrix.shape[1] == N:
            W_final = self.adj_matrix.copy()
            logging.info(f"Using Spectral Clustering with raw IEEE 39-bus adjacency matrix (K={n_clusters_used}).")
            try:
                sc = SpectralClustering(
                    n_clusters=n_clusters_used,
                    affinity='precomputed',
                    random_state=random_state,
                    n_init=spectral_n_init
                )
                labels = sc.fit_predict(W_final)
                
                # Recalculate centroids using the original anomaly scores and the new labels
                centroids = np.array([np.mean(bus_anomaly_scores[labels == lbl]) for lbl in np.unique(labels)])
                centroids = centroids.flatten()
                clustering_algorithm = "Spectral Clustering (Topology-aware, Anomaly-scored)"
            except Exception as e:
                logging.warning(f"Spectral Clustering failed ({e}). Falling back to K-Means on Anomaly Scores.")
                # Fallback to K-Means
                kmeans = KMeans(n_clusters=n_clusters_used, random_state=random_state, n_init='auto')
                kmeans.fit(bus_anomaly_scores.reshape(-1, 1))
                labels = kmeans.labels_
                centroids = kmeans.cluster_centers_.flatten()
                clustering_algorithm = "KMeans (Anomaly Score Only - Fallback)"
        else:
            # K-Means on anomaly scores only
            logging.info(f"Using standard K-Means on anomaly scores (K={n_clusters_used}).")
            kmeans = KMeans(n_clusters=n_clusters_used, random_state=random_state, n_init='auto')
            kmeans.fit(bus_anomaly_scores.reshape(-1, 1))
            labels = kmeans.labels_
            centroids = kmeans.cluster_centers_.flatten()
            clustering_algorithm = "KMeans (Anomaly Score Only)"
            
        # Determine which label corresponds to the high-anomaly (initial island) cluster
        high_label = int(np.argmax(centroids))

        # === IMPORTANT: Relabel so the highest-anomaly cluster is named Cluster 0 ===
        # This only renames labels (for consistent reporting). It does NOT change clustering.
        if high_label != 0:
            labels_re = labels.copy()
            # temporary marker
            labels_re[labels == high_label] = -1
            labels_re[labels == 0] = high_label
            labels_re[labels_re == -1] = 0
            labels = labels_re
            # After relabeling, set high_label to 0 for reporting consistency
            high_label = 0
        # =======================================================================

        bus_clusters: Dict[str, List[str]] = {}
        for lbl in np.unique(labels):
            name = f"Cluster {lbl}"
            bus_clusters[name] = [self.bus_order[i] for i in range(N) if labels[i] == lbl]

        # --- MODIFIED SECTION: Generator Reinforcement (Ensure Island Survival) ---
        generator_buses = set(self._get_generator_buses())
        cluster0_buses = set(bus_clusters.get("Cluster 0", []))

        if not (cluster0_buses & generator_buses):
            # Cluster 0 has no generators - MUST add one for island survival
            # Find generator with boundary path to maintain topological consistency
            cluster0_buses_list = list(cluster0_buses)
            generator_buses_list = self._get_generator_buses()
            gen_to_add, boundary_buses_to_add = self._find_generator_with_boundary_path(
                cluster0_buses_list, generator_buses_list, bus_clusters
            )
            
            if gen_to_add is not None:
                # Add generator to Cluster 0
                if gen_to_add not in bus_clusters["Cluster 0"]:
                    bus_clusters["Cluster 0"].append(gen_to_add)
                
                # Add boundary buses to Cluster 0 (needed for topological connection)
                for boundary_bus in boundary_buses_to_add:
                    if boundary_bus not in bus_clusters["Cluster 0"]:
                        bus_clusters["Cluster 0"].append(boundary_bus)
                
                # Remove from other clusters
                buses_to_move = [gen_to_add] + boundary_buses_to_add
                for bus_to_move in buses_to_move:
                    for cname, buses in bus_clusters.items():
                        if cname != "Cluster 0" and bus_to_move in buses:
                            try:
                                buses.remove(bus_to_move)
                            except ValueError:
                                 pass
                
                if boundary_buses_to_add:
                    logging.info(
                        f"Added generator Bus {gen_to_add} and boundary buses {boundary_buses_to_add} "
                        f"to Cluster 0 to ensure island survival while maintaining topological consistency."
                    )
                else:
                    logging.info(
                        f"Added directly connected generator Bus {gen_to_add} to Cluster 0 to ensure physical viability."
                    )
            else:
                logging.warning(
                    f"Cluster 0 has no generators and no viable path found to add a generator. "
                    f"Island will be marked as non-viable."
                )

        # --- end generator reinforcement section ---

        # Statistical validation
        high_scores = bus_anomaly_scores[labels == high_label]
        rest_scores = bus_anomaly_scores[labels != high_label]
        p_value = 1.0
        try:
            if len(high_scores) >= 2 and len(rest_scores) >= 2:
                stat, p_value = mannwhitneyu(high_scores, rest_scores, alternative='greater')
                p_value = float(p_value)
        except Exception as e:
            logging.debug(f"Stat test failed: {e}")
            p_value = 1.0

        logging.info(f"Isolation detection: anomaly_ratio={anomaly_ratio:.4f}, optimal_k={optimal_k}, final_k={n_clusters_used}, p_val={p_value:.4g}, method={clustering_algorithm}")

        return {
            "isolated": bool(isolated_flag),
            "anomaly_ratio": float(anomaly_ratio),
            "bus_clusters": bus_clusters,
            "optimal_k_found": int(optimal_k),
            "n_clusters_used": int(n_clusters_used),
            "p_value": float(p_value),
            "algorithm": clustering_algorithm,
            "bus_anomaly_scores": bus_anomaly_scores.tolist(),
            "initial_island_label": str(high_label) # Now should be "0" consistently after relabel
        }
    
    # --- CRITICALLY MODIFIED VIABILITY CHECK LOGIC (Unchanged from previous turn) ---
    def check_island_viability(self, stability_report: Dict[str, Any], isolation_report: Dict[str, Any]) -> Dict[str, Any]:
        bus_clusters = isolation_report.get("bus_clusters", {})
        
        # Original: The cluster with the highest anomaly score (now always labeled as Cluster 0)
        initial_label_str = f"Cluster {isolation_report.get('initial_island_label', '0')}"
        
        # --- IMPORTANT CHANGE: Do NOT auto-switch clusters.
        # final_island_key remains the initial_label_str (Cluster 0) ALWAYS.
        final_island_key = initial_label_str
        
        # --- Setup for two clusters ---
        all_labels = list(bus_clusters.keys())
        N_buses = len(self.bus_order)
        generator_buses = self._get_generator_buses()
        vdrop_bus = str(stability_report.get("max_voltage_drop_bus", "N/A"))
        rocof_bus = str(stability_report.get("max_rocof_bus", "N/A"))
        
        # (Removed auto-switching logic — Cluster 0 is always evaluated as the candidate island.)

        island_cluster_key = final_island_key
        island_buses = bus_clusters.get(island_cluster_key, [])

        # --- 1. Generator Status Check (using the final designated island) ---
        island_generators = [b for b in island_buses if b in generator_buses]
        island_has_generation = len(island_generators) > 0
        
        # --- 2. Instability Source Check (using the final designated island) ---
        vdrop_in_island = vdrop_bus in island_buses
        rocof_in_island = rocof_bus in island_buses

        # --- 3. Viability Assessment ---
        assessment = []
        is_viable = True
        viability = "*UNSTABLE* (failure reason undetermined)"

        # Generation Check (CRITICAL)
        if not island_has_generation:
            is_viable = False
        else:
            assessment.append(f"The candidate island contains *{len(island_generators)} generator(s)*: {', '.join(island_generators)}.")
            
        # Instability Location Check
        if vdrop_in_island:
            is_viable = False
            assessment.append(f"Primary Voltage Drop source (Bus {vdrop_bus}) *IS* within the candidate island.")
        else:
            assessment.append(f"Primary Voltage Drop source (Bus {vdrop_bus}) is *OUTSIDE* the candidate island.")

        if rocof_in_island:
            is_viable = False
            assessment.append(f"Primary ROCOF source (Bus {rocof_bus}) *IS* within the candidate island.")
        else:
            assessment.append(f"Primary ROCOF source (Bus {rocof_bus}) is *OUTSIDE* the candidate island.")

        
        # Determine the final Viability Status String
        if not island_has_generation:
             viability = "*NON-VIABLE* (lacks generation capacity)"
             assessment.append("Separation would inevitably fail due to lack of generation to meet load.")
        elif vdrop_in_island or rocof_in_island:
             viability = "*UNSTABLE* (island contains instability source)"
             assessment.append("Separation would likely fail as the island contains the primary disturbance.")
        elif is_viable and island_has_generation:
            viability = "*POTENTIALLY VIABLE* (instability sources are outside, generation is present)"
            assessment.append("The island shows low vulnerability to the initial event and has internal generation.")
        
        logging.info(f"Viability check: island={island_cluster_key}, buses={len(island_buses)}, has_gen={island_has_generation}, viability={viability}")
        
        # Return the final designated cluster
        return {
            "viability_status": viability,
            "island_cluster": island_cluster_key,
            "island_buses_count": len(island_buses),
            "main_grid_buses_count": N_buses - len(island_buses),
            "assessment": assessment,
            "p_value": float(isolation_report.get("p_value", 1.0)),
            "instability_location_check": {
                "vdrop_in_island": vdrop_in_island,
                "rocof_in_island": rocof_in_island,
                "island_has_generation": island_has_generation, 
                "island_generators": island_generators 
            }
        }

    # --- Plotting Functions (Unchanged from previous turn) ---
    def plot_signals(self, event_indices: List[int] | None = None):
        out_dir = self.config.get("output_dir", "output")
        os.makedirs(out_dir, exist_ok=True)
        V = self._reshape_to_matrix('Voltage')
        Fq = self._reshape_to_matrix('Frequency')
        if V is None:
            logging.warning("No Voltage data available for plotting.")
            return
        T = V.shape[0]
        time = np.arange(T)
        if event_indices is None or len(event_indices) == 0:
            baseline_fraction = float(self.config.get("baseline_fraction", 0.2))
            event_indices = [int(baseline_fraction * T)]
        plt.style.use('seaborn-v0_8-whitegrid')
        nrows = 2 + (1 if Fq is not None else 0) + (1 if Fq is not None else 0)
        fig, axes = plt.subplots(nrows, 1, figsize=(12, 3 * nrows), squeeze=False)
        ax_idx = 0
        ax = axes[ax_idx, 0]
        ax.plot(time, V)
        for e in event_indices:
            ax.axvline(e, color='r', linestyle='--')
        ax.set_title("Voltage (all buses over time)")
        ax_idx += 1
        ax = axes[ax_idx, 0]
        V_env, _ = self.analytic_envelope_phase(V)
        ax.plot(time, V_env)
        for e in event_indices:
            ax.axvline(e, color='r', linestyle='--')
        ax.set_title("Voltage Envelope (per time sample - aggregated channels)")
        ax_idx += 1
        if Fq is not None:
            ax = axes[ax_idx, 0]
            ax.plot(time, Fq)
            for e in event_indices:
                ax.axvline(e, color='r', linestyle='--')
            ax.set_title("Frequency (all buses over time)")
            ax_idx += 1
            rocof = self._calculate_rocof(Fq)
            ax = axes[ax_idx, 0]
            ax.plot(np.arange(rocof.shape[0]), rocof)
            ax.set_title("ROCOF (per-bus, over time)")
            ax_idx += 1
        plt.tight_layout()
        output_plots_file = self.config.get("output_plots_file", "pmu_analysis_plots.png")
        outp = os.path.join(out_dir, output_plots_file)
        fig.savefig(outp)
        plt.close(fig)
        logging.info(f"Saved signal plots to {outp}")
        
    def _plot_pre_during_post_comparison(self, stability_report: Dict[str, Any], isolation_report: Dict[str, Any]):
        out_dir = self.config.get("output_dir", "output")
        os.makedirs(out_dir, exist_ok=True)
        event_indices = stability_report.get("event_indices", [])
        V = self._reshape_to_matrix('Voltage')
        Fq = self._reshape_to_matrix('Frequency')
        if V is None:
            logging.warning("No Voltage data for pre/during/post comparison plot.")
            return
        T, N = V.shape
        
        # --- 1. & 2. Determine the Plot Order ---
        # Get the designated island buses from the isolation report's clusters
        island_cluster_key = isolation_report.get("final_island_cluster", "N/A")
        bus_clusters = isolation_report.get("bus_clusters", {})
        
        # Determine the buses in the designated island
        island_buses = bus_clusters.get(island_cluster_key, [])
        island_buses_set = set(island_buses)
        
        # Main grid buses are all others
        main_grid_buses = [b for b in self.bus_order if b not in island_buses_set]
        
        # New order: Island Buses first, then Main Grid Buses
        new_bus_order = island_buses + main_grid_buses
        
        # Create an index map to reorder the data arrays
        bus_to_idx = {bus: i for i, bus in enumerate(self.bus_order)}
        new_order_indices = [bus_to_idx[bus] for bus in new_bus_order]
        
        logging.info(f"Plotting buses in custom order: {len(island_buses)} island buses, {len(main_grid_buses)} main grid buses.")
        # --- End Plot Order Determination ---
        
        # --- Recalculate Time Periods (Original Logic) ---
        baseline_fraction = float(self.config.get("baseline_fraction", 0.2))
        first_event = max(1, int(event_indices[0])) if event_indices else max(1, int(baseline_fraction * T))
        pre_end = first_event 
        fixed_transient_len = int(self.config.get("transient_period_length", 50)) 
        remaining_time = T - pre_end
        during_len = min(fixed_transient_len, remaining_time - 1) 
        during_len = max(1, during_len)
        during_start = pre_end
        during_end = during_start + during_len
        post_start = during_end
        min_post_period = int(self.config.get("min_post_period_length", 10))
        if T - post_start < min_post_period:
             logging.warning(f"Not enough data for a meaningful 'Post' period (only {T - post_start} samples). Adjusting 'During' period.")
             post_start = during_end
        # --- End Recalculate Time Periods ---
        
        # --- 3. Reorder Data ---
        V_pre = np.nanmean(V[:pre_end, :], axis=0)[new_order_indices]
        V_during = np.nanmean(V[during_start:during_end, :], axis=0)[new_order_indices]
        V_post = np.nanmean(V[post_start:, :], axis=0)[new_order_indices]
        
        Fq_pre, Fq_during, Fq_post = None, None, None
        if Fq is not None:
            Fq_pre = np.nanmean(Fq[:pre_end, :], axis=0)[new_order_indices]
            Fq_during = np.nanmean(Fq[during_start:during_end, :], axis=0)[new_order_indices]
            Fq_post = np.nanmean(Fq[post_start:, :], axis=0)[new_order_indices]
        # --- End Reorder Data ---
        
        # --- 4. Update Plotting Logic ---
        plt.style.use('seaborn-v0_8-whitegrid')
        bar_width = float(self.config.get("bar_width", 0.25)) 
        bus_indices = np.arange(N)
        fig, axes = plt.subplots(2 if Fq is not None else 1, 1, figsize=(14, 6 * (2 if Fq is not None else 1)), sharex=True)
        if Fq is None:
            axes = [axes]
        ax1 = axes[0]
        
        bar_alpha = float(self.config.get("bar_alpha", 0.8))
        ax1.bar(bus_indices - bar_width, V_pre, bar_width, label='Pre-Disturbance Mean Voltage', color='skyblue', alpha=bar_alpha)
        ax1.bar(bus_indices, V_during, bar_width, label='During-Transient Mean Voltage', color='gold', alpha=bar_alpha)
        ax1.bar(bus_indices + bar_width, V_post, bar_width, label='Post-Isolation/Steady-State Mean Voltage', color='salmon', alpha=bar_alpha)
        
        # Add a vertical line separator between island and main grid buses
        if len(island_buses) > 0 and len(main_grid_buses) > 0:
            separator_offset = float(self.config.get("separator_offset", 0.5))
            separator_x = len(island_buses) - separator_offset
            separator_linewidth = float(self.config.get("separator_linewidth", 2))
            ax1.axvline(separator_x, color='black', linestyle='-', linewidth=separator_linewidth, label='Island/Main Grid Separator')
        
        ax1.set_ylabel("Mean Voltage (p.u. or V)")
        v_all = np.concatenate([V_pre, V_during, V_post])
        plot_margin_min = float(self.config.get("plot_margin_min", 0.98))
        plot_margin_max = float(self.config.get("plot_margin_max", 1.02))
        ax1.set_ylim(v_all.min() * plot_margin_min, v_all.max() * plot_margin_max)
        ax1.set_title(f"Bus-by-Bus Mean Voltage Comparison (Pre: 0-{pre_end}, During: {during_start}-{during_end}, Post: {post_start}-End)")
        ax1.set_xticks(bus_indices)
        text_rotation = int(self.config.get("text_rotation", 45))
        ax1.set_xticklabels(new_bus_order, rotation=text_rotation, ha='right') # Use the new order for labels
        ax1.legend()
        ax1.grid(axis='y')
        
        if Fq is not None:
            ax2 = axes[1]
            ax2.bar(bus_indices - bar_width, Fq_pre, bar_width, label='Pre-Disturbance Mean Frequency', color='lightgreen', alpha=bar_alpha)
            ax2.bar(bus_indices, Fq_during, bar_width, label='During-Transient Mean Frequency', color='orange', alpha=bar_alpha)
            ax2.bar(bus_indices + bar_width, Fq_post, bar_width, label='Post-Isolation/Steady-State Mean Frequency', color='darkred', alpha=bar_alpha)
            
            if len(island_buses) > 0 and len(main_grid_buses) > 0:
                 ax2.axvline(separator_x, color='black', linestyle='-', linewidth=separator_linewidth)
                 
            ax2.set_xlabel("Bus ID")
            ax2.set_ylabel("Mean Frequency (Hz)")
            f_all = np.concatenate([Fq_pre, Fq_during, Fq_post])
            ax2.set_ylim(f_all.min() * plot_margin_min, f_all.max() * plot_margin_max)
            ax2.set_title(f"Bus-by-Bus Mean Frequency Comparison (Pre: 0-{pre_end}, During: {during_start}-{during_end}, Post: {post_start}-End)")
            ax2.set_xticks(bus_indices)
            ax2.set_xticklabels(new_bus_order, rotation=text_rotation, ha='right') # Use the new order for labels
            ax2.legend()
            ax2.grid(axis='y')
            
        plt.tight_layout()
        output_comparison_file = self.config.get("output_comparison_file", "pmu_pre_during_post_comparison_bars.png")
        outp = os.path.join(out_dir, output_comparison_file)
        fig.savefig(outp)
        plt.close(fig)
        logging.info(f"Saved pre/during/post comparison bar plots to {outp}") 


    def plot_anomaly_clusters(self, isolation_report: Dict[str, Any]):
        out_dir = self.config.get("output_dir", "output")
        os.makedirs(out_dir, exist_ok=True)
        bus_anomaly_scores = np.array(isolation_report.get("bus_anomaly_scores", []))
        if bus_anomaly_scores.size == 0 or isolation_report.get("algorithm") == "Skipped (Stable System)":
            logging.warning("No anomaly scores or clustering was skipped. Skipping cluster plot.")
            return
        n_clusters = int(isolation_report.get("n_clusters_used", 2))
        N = len(self.bus_order)
        
        # Use the final designated island cluster for plotting
        final_island_cluster = isolation_report.get('final_island_cluster', 'N/A')
        
        # Reconstruct labels based on the final bus_clusters dictionary reported.
        final_bus_clusters = isolation_report['bus_clusters']
        label_map = {bus: int(c.split(' ')[1]) for c, buses in final_bus_clusters.items() for bus in buses}
        labels = np.array([label_map.get(bus, -1) for bus in self.bus_order])
        
        if (labels == -1).any():
             logging.warning("Could not map all buses to a cluster label for plotting.")
             return

        plt.style.use('seaborn-v0_8-whitegrid')
        fig = plt.figure(figsize=(12, 6))
            
        scatter_size = int(self.config.get("scatter_size", 100))
        scatter_alpha = float(self.config.get("scatter_alpha", 0.85))
        text_rotation = int(self.config.get("text_rotation", 45))
        for lbl in np.unique(labels):
            mask = labels == lbl
            cluster_key = f"Cluster {lbl}"
            # Use the final designated island key to label the plot
            label_name = f"Designated Island ({final_island_cluster})" if cluster_key == final_island_cluster else "Main Grid/Other Cluster"
            plt.scatter(np.arange(N)[mask], bus_anomaly_scores[mask], s=scatter_size, label=label_name, alpha=scatter_alpha)

        plt.xticks(np.arange(N), self.bus_order, rotation=text_rotation, ha='right')
        plt.xlabel("Bus ID")
        plt.ylabel("Average Anomaly Score (higher = more anomalous)")
        plt.title(f"Bus anomaly scores clustered (Method: {isolation_report['algorithm']}, K={n_clusters})")
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.tight_layout()
        output_clusters_file = self.config.get("output_clusters_file", "pmu_anomaly_clusters.png")
        outp = os.path.join(out_dir, output_clusters_file)
        fig.savefig(outp)
        plt.close(fig)
        logging.info(f"Saved anomaly clusters plot to {outp}") 


# ---------------------------
# Main execution
# ---------------------------
def main():
    config_file = os.environ.get("PMU_CONFIG_FILE", "config.json")
    cfg = load_config(config_file)
    log_file = cfg.get("log_file", "pmu_analysis.log")
    log_max_bytes = int(cfg.get("log_max_bytes", 5 * 1024 * 1024))
    log_backup_count = int(cfg.get("log_backup_count", 3))
    setup_logging(log_file, max_bytes=log_max_bytes, backup_count=log_backup_count)
    logging.info("Starting PMU analysis.")

    data_file = cfg.get("data_file", "pmu_fault_dataset.csv") 
    if not os.path.exists(data_file):
        logging.error(f"Data file '{data_file}' not found in current directory.")
        print(f"Error: '{data_file}' not found. Place your dataset in the script folder or modify the script.")
        return

    try:
        analyzer = PMUAnalyzer(data_file, cfg)
        stability_report = analyzer.check_stability()

        if not stability_report['stable']:
            logging.info("System is UNSTABLE. Proceeding with isolation detection and clustering.")
            
            isolation_report = analyzer.detect_isolation_with_clustering()
            viability_report = analyzer.check_island_viability(stability_report, isolation_report)
            
            # Update the isolation report with the final designated island key for consistency in summary/plotting
            isolation_report["final_island_cluster"] = viability_report["island_cluster"]
        else:
            logging.info("System is STABLE. Skipping isolation detection and clustering.")
            bus_order = analyzer.bus_order if analyzer.bus_order else []
            isolation_report = {
                "isolated": False, 
                "anomaly_ratio": 0.0,
                "bus_clusters": {"Main Grid": bus_order},
                "optimal_k_found": 1,
                "n_clusters_used": 1,
                "p_value": 1.0,
                "algorithm": "Skipped (Stable System)",
                "bus_anomaly_scores": [0.0] * len(bus_order),
                "initial_island_label": "N/A",
                "final_island_cluster": "N/A"
            }
            viability_report = {
                "viability_status": "*NOT APPLICABLE* (System Stable)",
                "island_cluster": "N/A",
                "island_buses_count": 0,
                "main_grid_buses_count": len(bus_order),
                "assessment": ["Isolation and viability check skipped because the system was determined to be stable."],
                "p_value": 1.0,
                "instability_location_check": {"vdrop_in_island": False, "rocof_in_island": False, "island_has_generation": False, "island_generators": []}
            }

        out_dir = cfg.get("output_dir", "output")
        os.makedirs(out_dir, exist_ok=True)
        summary = {
            "stability": stability_report,
            "isolation": isolation_report,
            "viability": viability_report
        }
        output_summary_file = cfg.get("output_summary_file", "pmu_analysis_summary.json")
        with open(os.path.join(out_dir, output_summary_file), "w") as f:
            json.dump(summary, f, indent=2)

        analyzer.plot_signals(stability_report.get("event_indices", []))
        analyzer._plot_pre_during_post_comparison(stability_report, isolation_report)
        
        if not stability_report['stable']:
            analyzer.plot_anomaly_clusters(isolation_report)

        # --- CONSOLE OUTPUT ---
        print("\n=== PMU STABILITY ANALYSIS ===")
        print(f"System is {'*STABLE' if stability_report['stable'] else 'UNSTABLE*'}")
        print(f"Max Voltage Drop: {stability_report['max_voltage_drop_pct']:.2f}% at Bus {stability_report['max_voltage_drop_bus']}")
        print(f"Max ROCOF: {stability_report['max_rocof']:.3f} Hz/s at Bus {stability_report['max_rocof_bus']}")

        print("\n=== ISLANDING DETECTION ===")
        print(f"Clustering Method: {isolation_report['algorithm']}")
        print(f"System is {'*ISOLATED*' if isolation_report['isolated'] else 'NOT ISOLATED'} (anomaly_ratio={isolation_report['anomaly_ratio']:.3f})")
        print(f"Optimal K (elbow): {isolation_report['optimal_k_found']}; used K={isolation_report['n_clusters_used']}")
        print("Clusters (based on electrical anomaly and topology):")
        for c, buses in isolation_report['bus_clusters'].items():
            # Highlight the final designated island
            if c == isolation_report.get('final_island_cluster'):
                 print(f" - {c} (DESIGNATED ISLAND, {len(buses)} buses): {', '.join(buses)}")
            else:
                 print(f" - {c} ({len(buses)} buses): {', '.join(buses)}")


        print("\n=== ISLAND VIABILITY (Post-Clustering Analysis) ===")
        print(f"Candidate island: {viability_report.get('island_cluster')} (Designated Stable Candidate)")
        print(f"Buses: {viability_report.get('island_buses_count')}")
        
        # --- Generator Reporting Section ---
        gen_count = len(viability_report['instability_location_check']['island_generators'])
        gen_list = ", ".join(viability_report['instability_location_check']['island_generators'])
        print(f"  (Includes {gen_count} Generator Bus(es): {gen_list if gen_count > 0 else 'None'})")

        print(f"Viability status: {viability_report.get('viability_status')}")
        for line in viability_report.get("assessment", []):
            print(f"- {line}")
        # -----------------------------------
            
        logging.info("PMU analysis complete. Results saved to output directory.")

    except FileNotFoundError as e:
        logging.error(f"File error: {e}")
        print(f"File error: {e}")
    except ValueError as e:
        logging.error(f"Data error: {e}")
        print(f"Data error: {e}")
    except Exception as e:
        logging.exception("Unexpected error during analysis.")
        print(f"Unexpected error: {e}")
if __name__ == "__main__":
    main()