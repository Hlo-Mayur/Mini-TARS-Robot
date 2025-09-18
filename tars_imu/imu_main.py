# imu_main.py
import json
import argparse
import os
import time
import socket
from imu_handler import IMUHandler
import numpy as np


# Try to import ROS2; if not available we'll fallback to UDP
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Imu
    ROS2_AVAILABLE = True
except Exception:
    ROS2_AVAILABLE = False

from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def load_config(path="config.json"):
    with open(path, "r") as f:
        return json.load(f)

def udp_broadcaster(cfg):
    host = cfg['udp_broadcast']['host']
    port = cfg['udp_broadcast']['port']
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    return sock, (host, port)

class RosImuNode:
    def __init__(self, topic):
        rclpy.init()
        self.node = rclpy.create_node("imu_publisher_node")
        self.pub = self.node.create_publisher(Imu, topic, 10)

    def publish_quat(self, q, rpy):
        msg = Imu()
        # orientation quaternion: IMUHandler's q is [qw,qx,qy,qz]; Ros Imu expects x,y,z,w
        qw, qx, qy, qz = q
        msg.orientation.x = qx
        msg.orientation.y = qy
        msg.orientation.z = qz
        msg.orientation.w = qw
        # We leave covariances as defaults (zeros) — tune later
        self.pub.publish(msg)

def run(cfg_path="config.json", override_sim=None):
    cfg = load_config(cfg_path)
    if override_sim is not None:
        cfg['simulation'] = override_sim

    imu = IMUHandler(cfg)

    # Prepare UDP fallback if requested
    udp = None
    if cfg.get("udp_broadcast", {}).get("enabled", True):
        udp = udp_broadcaster(cfg)

    # ROS2 node if requested and available
    ros_node = None
    if cfg.get("use_ros", False):
        if ROS2_AVAILABLE:
            ros_node = RosImuNode(cfg.get("ros_topic", "/imu/data"))
            print("ROS2 publisher ready.")
        else:
            print("ROS2 requested in config but rclpy not found; will broadcast UDP instead.")

    latest = {"rpy": [0,0,0], "quat": [1,0,0,0]}

    def callback(data):
        # called from IMUHandler loop
        latest['rpy'] = data['rpy']
        latest['quat'] = data['quat']
        # publish either via ROS2 or UDP
        if ros_node:
            try:
                ros_node.publish_quat(data['quat'], data['rpy'])
            except Exception as e:
                print("ROS publish error:", e)
        elif udp:
            msg = {
                "ts": data['timestamp'],
                "rpy": data['rpy'].tolist(),
                "quat": data['quat'].tolist()
            }
            try:
                udp[0].sendto(str(msg).encode("utf-8"), udp[1])
            except Exception as e:
                pass

    # Start background update loop and logging
    imu.start_loop(callback=callback)

    # Visualization (matplotlib) - show a cube + rpy lines
    if cfg.get("visualize", True):
        fig = plt.figure(figsize=(10,5))
        ax3d = fig.add_subplot(121, projection="3d")
        ax_rpy = fig.add_subplot(122)

        # cube setup
        cube_def = [[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]
        cube = np.array(cube_def)
        edges = [
            [0,1],[1,2],[2,3],[3,0],
            [4,5],[5,6],[6,7],[7,4],
            [0,4],[1,5],[2,6],[3,7]
        ]
        lines = [ax3d.plot([], [], [], 'b')[0] for _ in edges]
        ax3d.set_xlim(-2,2); ax3d.set_ylim(-2,2); ax3d.set_zlim(-2,2)
        # rpy plot setup
        ax_rpy.set_xlim(0, 10)
        ax_rpy.set_ylim(-180, 180)
        tdata, rdata, pdata, ydata = [], [], [], []
        ln_r, = ax_rpy.plot([], [], label="roll")
        ln_p, = ax_rpy.plot([], [], label="pitch")
        ln_y, = ax_rpy.plot([], [], label="yaw")
        ax_rpy.legend()

        def update_plot(frame):
            d = latest
            q = d['quat']
            # convert to scipy quat then rotation matrix
            sc_q = np.array([q[1], q[2], q[3], q[0]])
            rotm = R.from_quat(sc_q).as_matrix()
            rot_cube = cube @ rotm.T
            for edge, line in zip(edges, lines):
                pts = rot_cube[edge]
                line.set_data(pts[:,0], pts[:,1])
                line.set_3d_properties(pts[:,2])
            # rpy data
            tdata.append(time.time())
            rpy = d['rpy']
            rdata.append(rpy[0]); pdata.append(rpy[1]); ydata.append(rpy[2])
            # keep last N seconds
            maxlen = 200
            if len(tdata) > maxlen:
                tdata[:] = tdata[-maxlen:]; rdata[:] = rdata[-maxlen:]; pdata[:] = pdata[-maxlen:]; ydata[:] = ydata[-maxlen:]
            x = list(range(len(tdata)))
            ln_r.set_data(x, rdata); ln_p.set_data(x, pdata); ln_y.set_data(x, ydata)
            ax_rpy.relim(); ax_rpy.autoscale_view()
            return lines + [ln_r, ln_p, ln_y]

        ani = FuncAnimation(fig, update_plot, interval=100, blit=False)
        plt.show()
    else:
        # no visualization - just spin in main thread
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("Stopping...")
            imu.stop()
            time.sleep(0.2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--simulate", type=int, choices=[0,1], default=None,
                        help="override simulation flag (1 = simulate, 0 = hardware)")
    args = parser.parse_args()
    override = None
    if args.simulate is not None:
        override = True if args.simulate == 1 else False
    run(cfg_path=args.config, override_sim=override)
