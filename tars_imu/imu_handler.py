# imu_handler.py
import time
import os
import csv
import json
import threading
import numpy as np
from ahrs.filters import Madgwick
from scipy.spatial.transform import Rotation as R

# Try to import hardware library; if missing, we'll fallback to simulation
try:
    from mpu9250_jmdev.registers import *
    from mpu9250_jmdev.mpu_9250 import MPU9250
    HW_AVAILABLE = True
except Exception:
    HW_AVAILABLE = False

class IMUHandler:
    def __init__(self, cfg):
        """
        cfg: dict from config.json
        """
        self.cfg = cfg
        self.simulation = cfg.get("simulation", True) or not HW_AVAILABLE
        self.rate = float(cfg.get("rate_hz", 20))
        self.log_path = cfg.get("log_path", "logs/imu_log.csv")
        self.use_ros = cfg.get("use_ros", False)
        self.fuse = Madgwick()
        # AHRS Madgwick q format: [q0, q1, q2, q3] (q0 scalar). We store as returned.
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self._stop = False

        # If using hardware, initialize the MPU9250
        if not self.simulation and HW_AVAILABLE:
            try:
                self.mpu = MPU9250(
                    address_ak=AK8963_ADDRESS,
                    address_mpu_master=MPU9050_ADDRESS_68,
                    address_mpu_slave=None,
                    bus=int(cfg.get("i2c_bus", 1)),
                    gfs=GFS_1000,
                    afs=AFS_8G,
                    mfs=AK8963_BIT_16,
                    mode=AK8963_MODE_C100HZ
                )
                # Be careful: calibrate() will fail if the device isn't present.
                self.mpu.configure()
            except Exception as e:
                print("MPU init error, switching to simulation:", e)
                self.simulation = True

        # ensure log dir exists
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        # create CSV header if not exists
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "qx", "qy", "qz", "qw", "roll_deg", "pitch_deg", "yaw_deg"])

    # Utility: convert AHRS quaternion [q0,q1,q2,q3] to scipy format [x,y,z,w]
    @staticmethod
    def _to_scipy_quat(q):
        # ahrs returns [q0, q1, q2, q3] where q0 is scalar
        q0, q1, q2, q3 = q
        return np.array([q1, q2, q3, q0])

    def read_raw(self):
        """Return accel (m/s^2 or g), gyro (rad/s), mag (uT or raw) as numpy arrays"""
        if self.simulation or not HW_AVAILABLE:
            t = time.time()
            # simple slow rotation around Z plus small noise
            angle_deg = (t * 20.0) % 360.0
            angle_rad = np.deg2rad(angle_deg)
            # Simulated gyro: rad/s (derivative)
            gyr = np.array([0.0, 0.0, np.deg2rad(20.0)])  # constant 20 deg/s around Z
            # Simulated accel: body frame sees gravity vector rotated
            rot = R.from_euler('z', angle_deg, degrees=True)
            accel = rot.apply(np.array([0.0, 0.0, 1.0]))  # gravity as unit vector
            accel += np.random.normal(0, 0.02, 3)  # small noise
            mag = rot.apply(np.array([1.0, 0.0, 0.0])) + np.random.normal(0, 0.01, 3)
            return accel, gyr, mag
        else:
            # real sensor reads (example API)
            accel = np.array(self.mpu.readAccelerometerMaster(), dtype=float)
            gyro_deg = np.array(self.mpu.readGyroscopeMaster(), dtype=float)
            # convert gyro deg/s -> rad/s
            gyr = np.deg2rad(gyro_deg)
            mag = np.array(self.mpu.readMagnetometerMaster(), dtype=float)
            return accel, gyr, mag

    def update(self):
        """
        Read raw sensors, run Madgwick fusion, update internal quaternion,
        and return dict { 'quat':..., 'rpy': np.array([roll,pitch,yaw]) }
        """
        acc, gyr, mag = self.read_raw()
        # normalize accel to unit length (Madgwick expects normalized accelerometer)
        if np.linalg.norm(acc) != 0:
            acc_n = acc / np.linalg.norm(acc)
        else:
            acc_n = acc

        # Update Madgwick. prefer full 9-axis update when mag available
        try:
            # ahrs Madgwick expects arg names 'gyr', 'acc', 'mag', 'q'
            # Use 9-axis if mag not zero
            if mag is not None:
                q_new = self.fuse.updateIMU(gyr=gyr, acc=acc_n)

            else:
                q_new = self.fuse.updateIMU(gyr=gyr, acc=acc_n, q=self.q)
        except TypeError:
            # fallback if update signature differs; try IMU-only
            q_new = self.fuse.updateIMU(gyr=gyr, acc=acc_n, q=self.q)

        if q_new is None:
            q_new = self.q  # keep previous if filter didn't return
        self.q = q_new

        # Convert to roll/pitch/yaw (degrees) using scipy (scipy expects [x,y,z,w])
        sc_q = self._to_scipy_quat(self.q)
        rpy = R.from_quat(sc_q).as_euler('xyz', degrees=True)
        return {
            "timestamp": time.time(),
            "quat": self.q.copy(),
            "rpy": rpy.copy()
        }

    def log_row(self, data):
        """Append a row to CSV log. data from update()."""
        ts = data["timestamp"]
        q = data["quat"]
        rpy = data["rpy"]
        # ensure consistent order: qx,qy,qz,qw for CSV
        # ahrs q = [qw,qx,qy,qz] -> write qx,qy,qz,qw
        qw, qx, qy, qz = q
        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([ts, qx, qy, qz, qw, rpy[0], rpy[1], rpy[2]])

    def start_loop(self, callback=None):
        """
        Start a background loop that updates the filter at rate_hz and optionally calls callback(data).
        callback receives the data dict returned by update().
        """
        self._stop = False
        period = 1.0 / self.rate

        def loop():
            while not self._stop:
                data = self.update()
                if self.log_path:
                    self.log_row(data)
                if callback:
                    try:
                        callback(data)
                    except Exception as e:
                        print("Callback error:", e)
                time.sleep(period)

        t = threading.Thread(target=loop, daemon=True)
        t.start()
        return t

    def stop(self):
        self._stop = True
