# agent-podcast-transcripts-configurable

Configurable podcast transcription agent for Compound / Bloomberg shows.
It transcribes selected podcast episodes with WhisperX, applies deterministic cleanup steps, runs an optional LLM cleanup pass, and emails finished transcripts.

## What you need before running

- Python 3.11
- A working Python environment with the packages in `requirements.txt`
- `ffmpeg` available on your machine
- A `.env` file with required API keys and email settings
- If you want to use the wrapper scripts, a working `conda` installation or adjust the launcher config to match your environment

## Repo layout

- `agent.py` — main transcription agent
- `run-agent.sh` — wrapper to run the agent through the configured environment
- `run-monitor.sh` — wrapper to run the monitor
- `runtime-operations-config/runtime-config.txt` — launcher settings
- `settings/` — editable runtime settings
- `settings/phrases-and-vocabulary/` — phrase lists, host names, and vocabulary corrections
- `requirements.txt` — Python package dependencies
- `.env.example` — environment variable template

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/SingingData/agent-podcast-transcripts-configurable.git
cd agent-podcast-transcripts-configurable
```

### 2. Create your Python environment

Use your preferred environment manager. Example with pip:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

If you use conda instead, install the same packages into the environment named in `runtime-operations-config/runtime-config.txt`, or change that file to match your environment.

### 3. Create your `.env`

Copy the template and fill in real values:

```bash
cp .env.example ../.env
```

By default, `agent.py` looks for `.env` one directory above the repo.
If you want the `.env` file somewhere else, update the `.env` path logic in `agent.py`.

Minimum likely required values:
- `LLM_PROVIDER`
- `LLM_MODEL`
- API key for the provider you selected
- SMTP/email settings
- recipient email

### 4. Check runtime launcher settings

Edit:

- `runtime-operations-config/runtime-config.txt`

Defaults are now portable:
- `CONDA_BIN=conda`
- `CONDA_ENV=fastai`
- `PYTHON_ENTRY=agent.py`
- `MONITOR_ENTRY=monitoring/monitor.py`

The wrapper scripts derive the repo working directory automatically, so you do not need to hard-code your local path.

If you are not using conda, either:
- update the wrapper scripts, or
- run `agent.py` directly from your activated environment

## Running the agent

### Run through the wrapper

```bash
./run-agent.sh
```

### Run directly

```bash
python3 agent.py
```

## Running the monitor

```bash
./run-monitor.sh
```

## Configuration

Main runtime settings live in:

- `settings/transcription-settings.txt`
- `settings/podcast_feeds.txt`
- `settings/service-endpoints.txt`
- `settings/phrases-and-vocabulary/`

Notable configurable items include:
- test mode and episode count
- Whisper model/device/batch size
- audio retention
- phrase lists for intro/ad stripping
- vocabulary correction mappings

## Notes

- The agent performs a fail-fast Python dependency check at startup.
- The wrapper scripts depend on `conda` unless you adapt them to your environment.
- Cached models and API access are still environment-specific concerns.
- This repo includes tests and example phrase/config files, but external services and credentials are still required for a full real run.
