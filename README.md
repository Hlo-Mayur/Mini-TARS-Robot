# Mini TARS Robot 🤖

A modular robotics project to build a miniature version of TARS from the movie Interstellar, featuring voice interaction, computer vision for obstacle avoidance, and robotic motion.

![TARS Robot in Action](https://via.placeholder.com/600x400.png?text=Add+A+GIF+Or+Image+Of+Your+Robot+Here)

## About The Project

This project integrates a Raspberry Pi with a custom-trained YOLOv8 model for real-time object detection. The goal is to create a functional prototype that can navigate an indoor environment and avoid obstacles. The system is designed to be modular, with separate components for vision, motion, and eventually, voice interaction.

### Built With

* Python
* ROS2 Humble
* OpenCV
* YOLOv8 (Ultralytics)
* Raspberry Pi 4/5

## Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

* Python 3.9+
* Git
* (Optional for full project) ROS2 Humble installation

### Installation

1.  **Clone the repo:**
    ```sh
    git clone [https://github.com/YourUsername/Mini-TARS-Robot.git](https://github.com/YourUsername/Mini-TARS-Robot.git)
    ```
2.  **Navigate to the project directory:**
    ```sh
    cd Mini-TARS-Robot
    ```
3.  **Create and activate a virtual environment:**
    ```sh
    python -m venv venv
    .\venv\Scripts\Activate
    ```
4.  **Install Python dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

## Usage

To run the vision system on your local machine using a webcam:

1.  Place your trained model file (e.g., `best1.pt`) inside the `yolo` folder.
2.  Run the main vision script from the terminal:
    ```sh
    python yolo/run_vision.py --model yolo/best1.pt
    ```

## Project Structure