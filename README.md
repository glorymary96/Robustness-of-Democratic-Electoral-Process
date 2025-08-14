# Robustness of Democratic Electoral Processes to Computational Propaganda

This repository implements a computational social science framework to evaluate how robust different electoral systems are against targeted external influence (a proxy for “computational propaganda”). The model simulates dynamic voter opinions and quantifies how much external effort is needed to flip election outcomes across various systems.

---

## 🧠 Project Overview

- **Opinion Dynamics Model**: Simulates the evolution of voter opinions via a bounded-confidence consensus model (based on Taylor, Deffuant, Hegselmann–Krause).
- **Electoral Systems Modeled**:
  - **Single Representative (SR)**
  - **Winner-Takes-All (WTA)**
  - **Proportional Representation (PR)**
  - **Proportional Ranked Choice Voting (PRCV)**
- **External Influence Simulation**: Introduces perturbations to voter opinions until the electoral winner changes; quantifies required “effort.”
- **Validation**: Compares simulated *volatility* with historical data from U.S. House elections (2012–2020).

---

## 📂 Repository Structure
```
Robustness-of-Democratic-Electoral-Process/
├── DATA/                                   # Simulation outputs and input datasets
├── Python codes/                           # All code modules and utilities
│   ├── Data_analysis.ipynb                 # Generates figures & analysis in the paper
│   ├── Model_execution.py                  # Simple model execution
│   ├── Electoral_system_function.py        # Electoral outcome functions for different electoral systems
│   └── Other files..                       # Other sub files (read README_summary_files)
├── README_Python_codes/                    # Module-by-module usage instructions
└── README.md                               # Overview of the project
```

## 🎯 Getting Started


1. **Clone the repository**:

```bash
git clone https://github.com/glorymary96/Robustness-of-Democratic-Electoral-Process.git
cd Robustness-of-Democratic-Electoral-Process
```

2. **Create a virtual environment** (optional but recommended)
```
python -m venv rob_es
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. **Install dependencies**
```
pip install -r requirements.txt
```

4. **Usage**
- Detailed description of each pythin code is given in 'README_Python_Codes'
- All analysis and plots in 'Python_Codes/Data_analysis.ipynb'

5. **Results**
- Volatility matching: Model’s district-level volatility shows good correlation with real election data.

- Key Findings:
    - Proportional systems (PR, PRCV) generally require higher influence effort—most robust.

    - Bipartite systems are more robust in less polarized societies.

    - In multipartite settings, moderate openness maximizes resilience; extreme openness can sometimes benefit extremists.


6. **References**
- Givi et al., “On the robustness of democratic electoral processes to computational propaganda”, Scientific Reports (2024)