# orbital-mechanics-simulator
A fully featured GUI-based orbital mechanics simulator with perturbation modeling, JAX-Diffrax ODE solvers, and 3D trajectory visualization.

--

## Features

- **High-fidelity perturbation modeling**
  - J2 oblateness
  - Atmospheric drag
  - 3rd-body gravitational influences (e.g., Moon)
- **Multiple numerical solvers**
  - Euler, RK4, RK6, Verner RK7, RK8 (via `diffrax`)
- **Custom initial condition presets**
  - LEO, MEO, GEO, Molniya, hyperbolic escape, sun-synchronous, and more
- **Quantitative outputs**
  - Over 20 orbital parameters tracked (eccentricity, inclination, RAN, etc.)
  - 3D plots of position, velocity, and angular momentum
- **Animation**
  - Export animations of orbital motion
- **GUI**
  - Built with `Tkinter`, allows intuitive parameter entry and simulation control

---

## File Structure

```
OrbitalMechanicsSimulator/
├── main.py            # GUI application entry point
├── simulator.py       # Core numerical integration and physics engine
├── visuals.py         # Graphing, plotting, and animation utilities
├── icons/             # Custom icons used in GUI
├── output/            # Auto-generated folder with plots, animations, data
├── requirements.txt   # Required packages
├── .gitignore         # File exclusions
└── README.md          # This file
```

---

## Requirements

Install dependencies using pip:

```bash
pip install -r requirements.txt
```

Main dependencies:
- `jax`, `diffrax` – numerical integration
- `matplotlib`, `numpy`, `plotly` – plotting
- `tkinter` – GUI

---

## Running the Simulator

To launch the GUI:

```bash
python main.py
```

You can configure:
- Orbit type (select preset)
- Solver method
- Perturbation options
- Simulation duration

Results (graphs and animations) are saved in the `output/` folder.

---

## Screenshots

_TODO: Add screenshots of the GUI, 3D plots, and animations here._

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Future Plans

- Include mission planner and thrust modeling
- Expand to multi-body solver with Lagrange points

---

## Contributing

If you'd like to contribute, feel free to fork and submit a pull request! Issues and suggestions are welcome.
