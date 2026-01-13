# HVAC-MARL 6e43

> Multi-Agent Reinforcement Learning for HVAC Control in Buildings

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/BuildingGym/hvacmarl6e43
cd hvacmarl6e43

# Install the package and dependencies
python -m pip install -e .
python -m pip install --extra-index-url https://test.pypi.org/simple git+https://github.com/NTU-CCA-HVAC-OPTIM-a842a748/EnergyPlus-OOEP
pip install -r requirements.txt
```

## 📁 Project Structure

```
hvacmarl6e43/
├── packages/
│   ├── hvacmarl6e43/           # Core source code and utilities
│   │   ├── utils.py            # Utility functions
│   │   └── buildings/          # Building models (IDF files)
│   └── hvacmarl6e43_notebooks/ # Experiment notebooks
│       ├── run_baselines.ipynb # Run baseline experiments
│       ├── run_monoagent.ipynb # Run single-agent RL experiments
│       ├── run_multiagent.ipynb# Run multi-agent RL experiments
│       ├── viz_metrics.ipynb   # Visualize experiment metrics
│       └── viz_obs.ipynb       # Visualize observations
├── pyproject.toml
├── requirements.txt
└── README.md
```

## ▶️ Running Experiments

**Main experiment scripts are located in `packages/hvacmarl6e43_notebooks/`:**

| Notebook | Description |
|----------|-------------|
| `run_baselines.ipynb` | Run baseline experiments |
| `run_monoagent.ipynb` | Run single-agent reinforcement learning experiments |
| `run_multiagent.ipynb` | Run multi-agent reinforcement learning experiments |
| `viz_metrics.ipynb` | Visualize experiment metrics |
| `viz_obs.ipynb` | Visualize observation data |

```bash
# Open with Jupyter or VS Code to run experiments
```

## 📄 Citation

If you find this work useful, please cite our paper:

```bibtex
@article{guan2025adaptive,
  title={Adaptive Multi-Agent HVAC Control for Thermal Comfort Using Multi-Agent PPO with Population-Based Training},
  author={Guan, Songze and Chen, Ruotian and Liu, Jiuwei and Zhou, Kate Qi and Dai, Xilei and Li, Wentai and Somasundaram, Sivanand and Chong, Alex and Lee, Christopher HT and Yuen, Chau},
  journal={Energy and Buildings},
  pages={116882},
  year={2025},
  publisher={Elsevier}
}
```

## 📜 License

MIT License
