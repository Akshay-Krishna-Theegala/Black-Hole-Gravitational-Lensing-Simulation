---

# 🌌 3D Black Hole & Gravitational Lensing Simulation

A Python-based simulation visualizing a black hole, an accretion disk, and gravitational lensing effects against a starfield background. Built using **VisPy** and custom **OpenGL shaders (GLSL)** as a physics capstone project.

---

## 🌟 Features

* **Gravitational Lensing:** Bends light rays from a procedural background starfield around the black hole's event horizon.
* **Accretion Disk:** Renders orbiting matter using particle sprites with rotational motion, velocity shifts, and inner-edge fade effects.
* **Interactive Controls:** Real-time camera navigation and runtime adjustment of simulation parameters (mass, star density, particle size, and disk visibility).

---

## 🚀 Getting Started

### Prerequisites

Make sure you have Python installed along with the necessary libraries:

```bash
pip install numpy vispy

```

### Running the Simulation

Clone or download this repository, then run:

```bash
python main.py

```

---

## 🎮 Controls & Customization

| Key / Action | Function |
| --- | --- |
| **Left Click + Drag** | Orbit around the black hole |
| **Right Click + Drag** | Pan the camera view |
| **Mouse Scroll** | Zoom in / out |
| **`+` / `-**` | Increase / Decrease Black Hole Mass ($M$) |
| **`[` / `]**` | Adjust Starfield Density / Threshold |
| **`p` / `P**` | Decrease / Increase Particle Size |
| **`D`** | Toggle Accretion Disk ON/OFF |
| **`Spacebar`** | Pause / Resume Simulation Time |
| **`Up` / `Down` Arrow** | Adjust Simulation Spin Speed |
| **`R`** | Reset to Default State |

*(Note: Disk width, thickness, and inner radius can also be easily configured directly in the code initialization block at the bottom of `main.py`).*

---

## 🛠️ Tech Stack

* **Python:** Core application logic and camera state management.
* **VisPy:** High-performance OpenGL rendering wrapper.
* **GLSL:** Vertex and fragment shaders handling real-time ray deflection and particle rendering.

---

## 📄 License

Distributed under the [MIT License](https://www.google.com/search?q=LICENSE).

---
