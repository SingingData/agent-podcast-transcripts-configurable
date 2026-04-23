# agent-podcast-transcripts-configurable

Configurable podcast transcription agent for Compound shows.
This agent transcribes selected podcast episodes with WhisperX, applies deterministic cleanup steps, runs an optional LLM cleanup pass, and emails finished transcripts.

This agent is written to run locally on a CPU and with minimal token expense.  Note, it can be further configured to run the LLM clean-up lcoally and to add a diarization step to identify the speakers in-line.  

To run locally, you need to install the requirements listed below.  Note, I recommend installing from miniconda, and further install the fastai module (in a virtual environment in python if you can) to get most of yoru requirements loaded in one throw.  

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

## Running Agent in Test Mode

Test mode is controlled in:

- `settings/transcription-settings.txt`

Relevant settings:

```txt
TEST_MODE=true
TEST_MODE_EPISODE_COUNT=6
```

What test mode does:
- processes only the most recent `TEST_MODE_EPISODE_COUNT` matching episodes instead of the normal selection flow
- keeps intermediate transcript artifacts that are useful for inspection and debugging

Intermediate artifacts written only in test mode:
- `transcripts/raw_transcript_{safe_title}.txt`
  - initial transcript text produced from transcription output
- `transcripts/post_processed_{safe_title}.txt`
  - transcript text after vocabulary correction, `PODCAST START` handling, clipping after the last marker, and paragraph cleanup
- `transcripts/llm_cleaned_transcript_{safe_title}.txt`
  - plain transcript text after the LLM cleanup pass (or fallback text if LLM cleanup is skipped/fails)

Final deliverables written in all modes:
- `transcripts/final_cleaned_{safe_title}.txt`
- `transcripts/final_cleaned_{safe_title}.md`

Recommended workflow:
1. set `TEST_MODE=true`
2. set `TEST_MODE_EPISODE_COUNT` to the number of recent episodes you want to inspect
3. run `./run-agent.sh` or `python3 agent.py`
4. review the intermediate files in `transcripts/`
5. revert to normal mode when finished:

```txt
TEST_MODE=false
TEST_MODE_EPISODE_COUNT=1
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
