# Autonomous Omni-Directional Robot Navigation System

Autobot is an autonomous omni-directional robotics project that integrates computer vision, machine learning, navigation, and embedded control into a single robotic system.

The project focuses on enabling an omni-directional robot to perceive its environment, detect obstacles using a YOLO-based vision pipeline, make navigation decisions, and execute movement through an embedded motor-control system.

The repository contains the navigation algorithms, computer-vision modules, YOLO training pipeline, dataset-processing utilities, Arduino firmware, camera-calibration utilities, and a small sample dataset used to demonstrate the overall workflow.

---

## Table of Contents

- [Overview](#overview)
- [Project Objectives](#project-objectives)
- [System Architecture](#system-architecture)
- [How the System Works](#how-the-system-works)
- [Computer Vision Pipeline](#computer-vision-pipeline)
- [Navigation System](#navigation-system)
- [Robot Control](#robot-control)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset Preparation](#dataset-preparation)
- [YOLO Training](#yolo-training)
- [Camera Calibration](#camera-calibration)
- [Arduino Firmware](#arduino-firmware)
- [Project Workflow](#project-workflow)
- [Results](#results)
- [Challenges](#challenges)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)

---

## Overview

Autonomous mobile robots need to combine perception, decision-making, and motion control in order to operate without continuous human intervention.

Autobot was developed as an experimental autonomous robotics platform to explore this integration.

The system combines:

- Computer vision for environmental perception
- YOLO-based object and obstacle detection
- Camera calibration
- Autonomous navigation logic
- Omni-directional robot movement
- Arduino-based embedded control
- Dataset preparation and annotation conversion

The project is organized into independent modules so that individual parts of the system can be developed, tested, and improved separately.

---

## Project Objectives

The main objectives of Autobot are:

1. Develop an autonomous navigation system for an omni-directional robot.
2. Use computer vision to obtain information about the robot's environment.
3. Train and integrate a YOLO-based detection model.
4. Detect obstacles and relevant objects from camera input.
5. Develop navigation and coordinate-based movement logic.
6. Control the robot's omni-directional movement using an embedded controller.
7. Develop reusable tools for dataset preparation and annotation conversion.
8. Create a modular architecture that can be extended for future autonomous and swarm applications.

---

## System Architecture

The overall Autobot system can be represented as:

<img width="960" height="540" alt="Block Diagram" src="https://github.com/user-attachments/assets/d4d57096-5e71-46a2-acbf-414705cb2cec" />



