# Agentdb

## Description

[Briefly describe your project here.]

## Installation


1. **Initialize git:**
    - Windows: Run `scripts/000_init.bat`
    - Linux/macOS: Run `bash scripts/000_init.sh`

2. **Create a virtual environment:**
    - Windows: Run `scripts/001_env.bat`
    - Linux/macOS: Run `bash scripts/001_env.sh`

3. **Activate the virtual environment:**
    - Windows (cmd): Run `scripts/002_activate.bat`
    - PowerShell: `& .\.venv\Scripts\Activate.ps1`
    - Linux/macOS: `source scripts/002_activate.sh`

4. **Install dependencies:**
    - Windows: `scripts/003_setup.bat`
    - Linux/macOS: `bash scripts/003_setup.sh` (installs from `requirements.txt`).

5. **Deactivate the virtual environment:**
    - Windows: `scripts/008_deactivate.bat`
    - Linux/macOS: `bash scripts/008_deactivate.sh`

## Usage

1. **Run the main CLI app:**
    - Windows: Run `scripts/004_run.bat`
    - Linux/macOS: Run `bash scripts/004_run.sh`

2. **Run the Streamlit UI:**
    - Windows: Run `scripts/006_run_app.bat`
    - Linux/macOS: Run `bash scripts/006_run_app.sh`

    [Provide instructions on how to use your application.]

## Scripts

Cross-platform helpers live under `scripts/`.

- Windows (.bat):
    - `000_init.bat`: Initialize git and set global name/email.
    - `001_env.bat`: Create a virtual environment `.venv`.
    - `002_activate.bat`: Activate the `.venv` environment (cmd).
    - `003_setup.bat`: Install Python packages from `requirements.txt`.
    - `004_run.bat`: Run `main.py`.
    - `005_run_test.bat`: Run pytest and generate HTML report.
    - `005_run_code_cov.bat`: Run pytest with coverage and HTML report.
    - `008_deactivate.bat`: Deactivate the current virtual environment.

- Linux/macOS (.sh):
    - `000_init.sh`: Initialize git and set global name/email.
    - `001_env.sh`: Create a virtual environment `.venv`.
    - `002_activate.sh`: Source the `.venv` activation script.
    - `003_setup.sh`: Install Python packages from `requirements.txt`.
    - `004_run.sh`: Run `main.py`.
    - `005_run_test.sh`: Run pytest and generate HTML report.
    - `005_run_code_cov.sh`: Run pytest with coverage and HTML report.
    - `008_deactivate.sh`: Deactivate the current virtual environment.

Tip: On Unix-like systems, you might need to make scripts executable once:
`chmod +x scripts/*.sh`

## Contributing

[Explain how others can contribute to your project.]

## License

[Specify the project license, if any.]
