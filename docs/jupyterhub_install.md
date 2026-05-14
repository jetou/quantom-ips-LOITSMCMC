# Installation Guide

To simplify the installation process, we will utilize the Theory-Beta image on the iFarm. 
A similar process can be done without the image, but requires installing more `pip` packages like `torch`. 

To use the environment through SSH instead of Jupyter-Hub, you will need to load the Theory-Beta Apptainer container (not covered here).

## Step-by-Step Instructions
1. Log in to the iFarm using [Jupyter-Hub](https://jupyterhub.jlab.org/)
   - See this [link](https://jlab.servicenowservices.com/kb?id=kb_article&sysparm_article=KB0014668) for instructions to get access
   - Create a new session with the Theory-Beta Image
3. In a Jupyter-Hub terminal, create (using `mkdir`) or use an existing directory on `/work` and clone the quantom-ips repository.
   ([Repo Link](https://github.com/quantom-collab/quantom-ips))
    - Command: `git clone https://github.com/quantom-collab/quantom-ips.git`
    - If `git clone` hangs, ensure you have set up a personal access token
	    - Log in to GitHub
	    - Go to Settings->Developer Settings->Personal Access Tokens
	    - Set up a token with repository access.
    - If you plan to make changes to contribute, check out the most recent version (v0.0.2) in a new local branch 
    ```bash=
        cd quantom-ips
        git checkout tags/v0.0.2 -b <branch_name>
    ```
5. Create a virtual environment in the root of the quantom-ips repo:
    - If you haven't already, run `cd quantom-ips` to enter the root directory
    - Command: `python -m venv .venv --system-site-packages`
    - Explanation: `--system-site-packages` ensures that the virutal environment uses packages that are already installed in the Theory-Beta image.
      If you are not using the image, it is not necessary to use `--system-site-packages`
6. Activate the virtual environment
    - (bash) Command `source .venv/bin/activate`
        - If using a shell that is not bash, you may need a different activation script (also stored in `.venv/bin/`).
7.  Install `hydra-core` and `quantom-ips`
    - Command: `pip install hydra-core`
    - Explanation: 
        - [Hydra](hydra.cc) allows for configuration management and runtime type checking of input. 
    - Command: From the `quantom-ips` directory: `pip install -e .`
    - Explanation:
        - Installing `quantom-ips` allows the repository to use regular `import quantom_ips` calls instead of
          path based imports is brittle (can easily break) and may cause issues with other packages. 
 8. You are now ready to run the workflow!
    The current training script in `quantom_ips/src/quantom_ips/drivers` requires a path to the 2D proxy data,
    currently stored at `/work/data_science/quantom/data/events_2d_proxy_app_v0.npy`. 
    - Command: From the downloaded `quantom-ips` directory: 
    ```bash=
    cd src/quantom_ips/drivers
    python training_workflow.py environment.parser.path="['/work/data_science/quantom/data/events_2d_proxy_app_v0.npy']"
    ```
    
## Using Jupyter Notebooks (Optional)

You can set up a Jupyter-Hub Environment to allow importing packages from your new virtual environment. 
Following the following steps will add a new option in the Jupyter-Hub Launcher to open a new notebook or python session with this environment. 

- When the environment is activated (see 4.), run `python -m ipykernel install --user --name=<env_name>`
  - `<env_name>` can be a descriptive name of your choice. (e.g. `quantom-ips`)
  - You can see all installed kernels by running `jupyter kernelspec list` 
  - If you no longer need the environment, you can uninstall it with `jupyter kernelspec uninstall <env_name>`

## Future Use

In order to use the workflow through a Jupyter-Hub terminal, you must activate the environment (see 4.) each time you start a Jupyter-Hub session. 
No other actions are required.

