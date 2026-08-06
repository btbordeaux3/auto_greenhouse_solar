SOLAR GREENHOUSE DIGITAL TWIN
==============================

FOLDER STRUCTURE
----------------
greenhouse_digital_twin/
├── simulink/
│   └── greenhouse.m          ← run this in MATLAB to build the Simulink model
├── python/
│   ├── run_twin.py           ← main script, run this from terminal
│   ├── requirements.txt      ← Python dependencies
│   ├── optimizer/
│   │   └── stochastic_mpc.py
│   ├── weather/
│   │   └── scenario_generator.py
│   └── sensors/
│       └── state_estimator.py
└── results/                  ← output plots and CSV appear here after running


HOW TO RUN — PYTHON ONLY (no MATLAB needed)
--------------------------------------------
1. Open a terminal and navigate to the python folder:
      cd greenhouse_digital_twin/python

2. Install dependencies (only needed once):
      pip install -r requirements.txt

3. Run the simulation:
      python run_twin.py --no-matlab

4. When finished, open the results folder — you will find:
      twin_results.png   (plot of SoC, power, and load dispatch)
      twin_log.csv       (raw data for every timestep)


HOW TO RUN — WITH SIMULINK (needs MATLAB installed)
----------------------------------------------------
Step 1 — Build the Simulink model (only needed once):
   - Open MATLAB
   - In MATLAB, navigate to the simulink folder:
        cd('path/to/greenhouse_digital_twin/simulink')
   - Run:
        greenhouse
   - This generates greenhouse_twin.slx and opens it in Simulink
   - Leave MATLAB open

Step 2 — Run the digital twin:
   - Open a terminal, navigate to the python folder
   - Run:
        python run_twin.py
   - Python and Simulink will co-simulate together


USEFUL OPTIONS
--------------
python run_twin.py --no-matlab                  (Python plant, no MATLAB)
python run_twin.py --no-matlab --sim-days 3     (run 3 days)
python run_twin.py --no-matlab --scenarios 50   (more weather scenarios)
python run_twin.py --no-matlab --weather markov (Markov chain weather)
python run_twin.py                              (full Simulink co-sim)
