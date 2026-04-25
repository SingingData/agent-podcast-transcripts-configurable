# agent-podcast-transcripts-configurable

This repo contains a locally run, configurable podcast transcription agent.  Right now it's configured to transcribe all of the Compound shows.

**What does it do?**  This agent transcribes the latest podcast episodes with a local Python module, WhisperX, applies deterministic cleanup steps, runs an optional LLM cleanup pass configured to use Grok at the moment, drops a transcript to your local drive, and emails finished transcripts to the email account(s) of your choosing.  I'm currently running this agent in an OpenClaw harness, which gives me, amongst many other things, the ability to schedule cron jobs to check for new episodes and email new episode transcripts when it finds them.  Note, you don't need to run this within a harness like OpenClaw, but it makes it a lot easier.

**How is it designed?** This agent is written to run locally on a CPU and with minimal token expense. It uses WhisperX which is a relatively beefy local audio model, but if it's too big for your system, you can use faster-whisper model which is smaller.  The only time this agent calls external LLM service is to do a final LLM clean-up pass on the transcript. (WhisperX has done most of that already.) But this step is optional and you can disable.  Or, the agent can be configured to run the LLM cleanup locally rather than calling a service like Grok or OpenAI. I have it configured to use the most economical of the good LLM's - Grok - which still allows you to use the API calls on an all-you-can-eat plan.  Finally, you will note that speaker names are not yet identified in the body of the trasncript, a function called diarization.  I plan to extend this to add a diarization step to identify speakers inline.  There are some nice local models and local approaches available for this.  

To run locally, you need to install the requirements listed below. Note that if you don't already have Python on your machine, I strongly recommend installing Miniconda (details below) and then installing the fastai Python module (also below) to get most of your requirements loaded in one shot with the least trouble and overhead. Ask your LLM for help with these setup steps if needed.

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
- `run-corrections-ingest.sh` — wrapper to run the weekly correction-ingest job
- `runtime-operations-config/runtime-config.txt` — launcher settings
- `settings/` — editable runtime settings
- `phrases-and-vocabulary/` — phrase lists, host names, vocabulary corrections, and local correction-request files
- `phrases-and-vocabulary/correction-requests/` — local per-show correction submissions captured from reply emails (gitignored)
- `requirements.txt` — Python package dependencies
- `.env.example` — environment variable template

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/SingingData/agent-podcast-transcripts-configurable.git
cd agent-podcast-transcripts-configurable
```

### 2. Install Miniconda and create the Python environment

This repo is set up to run through a Conda environment named `fastai`.
Install Miniconda first, then create and populate that environment.

#### If you are running on macOS

If you use Homebrew:

```bash
brew install --cask miniconda
conda init zsh
exec zsh
```

If you are not using Homebrew, download Miniconda from the official installer page and complete the shell initialization step for your shell before continuing.

Create the `fastai` environment with Python 3.11:

```bash
conda create -n fastai python=3.11 -y
conda activate fastai
```

Install the required packages into `fastai`:

```bash
python -m pip install --upgrade pip
python -m pip install fastai
python -m pip install -r requirements.txt
```

#### If you are running on Windows

You can install Miniconda from the command line if you prefer not to download it manually in a browser.

If you are using Command Prompt, these commands download the latest 64-bit Miniconda installer, install it for your user account, and then remove the installer:

```bat
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe --output .\Miniconda3-latest-Windows-x86_64.exe
start /wait "" .\Miniconda3-latest-Windows-x86_64.exe /InstallationType=JustMe /RegisterPython=0 /S /D=%UserProfile%\Miniconda3
del .\Miniconda3-latest-Windows-x86_64.exe
```

If you prefer PowerShell, you can use:

```powershell
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe --output .\Miniconda3-latest-Windows-x86_64.exe
Start-Process .\Miniconda3-latest-Windows-x86_64.exe -ArgumentList '/InstallationType=JustMe','/RegisterPython=0','/S',"/D=$env:USERPROFILE\Miniconda3" -Wait
Remove-Item .\Miniconda3-latest-Windows-x86_64.exe
```

After installation finishes, open a new Anaconda Prompt (or a new Command Prompt/PowerShell window if `conda` is now available) and verify the install:

```bat
conda --version
conda list
```

Create the `fastai` environment with Python 3.11:

```bat
conda create -n fastai python=3.11 -y
conda activate fastai
```

Install the required packages into `fastai`:

```bat
python -m pip install --upgrade pip
python -m pip install fastai
python -m pip install -r requirements.txt
```

Install `ffmpeg` on Windows as well, and make sure it is available on your `PATH` before running the agent.

This matches the default wrapper configuration in:
- `runtime-operations-config/runtime-config.txt`

If you want to use a different Conda environment name, update that file to match.

#### If you are running on Linux

You can install Miniconda directly from the command line.

If `curl` is available, run:

```bash
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

If you prefer `wget`, run:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

During the installer prompts:
- review and accept the license
- confirm the install location
- allow the installer to initialize conda for your shell when prompted

After installation finishes, open a new terminal window, or reload your shell configuration, then verify the install:

```bash
conda --version
conda list
```

Create the `fastai` environment with Python 3.11:

```bash
conda create -n fastai python=3.11 -y
conda activate fastai
```

Install the required packages into `fastai`:

```bash
python -m pip install --upgrade pip
python -m pip install fastai
python -m pip install -r requirements.txt
```

Install `ffmpeg` on Linux as well, and make sure it is available on your `PATH` before running the agent.

This matches the default wrapper configuration in:
- `runtime-operations-config/runtime-config.txt`

If you want to use a different Conda environment name, update that file to match.

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
- email credentials and sender settings in `.env`
- `TRANSCRIPT_RECIPIENTS` for one or more regular-run recipients (comma-separated)
- optional `TEST_RUN_RECIPIENTS` for one or more test-run recipients (comma-separated; if not set, test runs automatically fall back to `TRANSCRIPT_RECIPIENTS`)
- for correction ingest, either `IMAP_USERNAME` / `IMAP_PASSWORD`, or reuse `GMAIL_ADDRESS` / `GMAIL_APP_PASSWORD`
- optional `IMAP_HOST` and `IMAP_PORT` if you are not using Gmail IMAP defaults

Email/service configuration is split across two places:
- `.env` holds provider keys, email credentials, sender info, recipient lists, and optional IMAP credentials for reply ingestion
- `settings/service-endpoints.txt` holds service endpoints such as `SMTP_HOST` and `SMTP_PORT`

### 4. Check runtime launcher settings

Edit:

- `runtime-operations-config/runtime-config.txt`

Defaults are now portable:
- `CONDA_BIN=conda`
- `CONDA_ENV=fastai`
- `PYTHON_ENTRY=agent.py`
- `MONITOR_ENTRY=monitoring/monitor.py`
- `CORRECTION_INGEST_ENTRY=monitoring/fetch_corrections.py`

The wrapper scripts derive the repo working directory automatically, so you do not need to hard-code your local path.

If you are not using conda, either:
- update the wrapper scripts, or
- run `agent.py` directly from your activated environment

## Running the agent

There are a number of ways you can run the agent.  You can run directly (see below).  Or you can run with the help of an agent harness.  I highly recommend running through an agent, as it's so much easier once you get the harness set up.  

### Run through an agent harness (OpenClaw, Hermes, or another launcher)

If you prefer, you can run this repo through an agent harness instead of calling the script manually. In that setup, the harness should use this repository as its working directory and launch `agent.py` with the same Python environment and `.env` settings described above.

For OpenClaw, see the official documentation at `docs.openclaw.ai` (or your local OpenClaw docs if it is already installed).

For Hermes or any other harness, follow that tool's installation and setup guide, then configure it to run this repo with `agent.py` as the entry point.

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
TEST_MODE_MONTH=
WEEKLY_SUMMARY_TEST_MODE=false
```

What test mode does:
- bypasses the normal new-episode selection flow and instead builds a shared candidate pool across all matching feeds
- processes only the most recent `TEST_MODE_EPISODE_COUNT` matching episodes from that shared pool (not per show)
- keeps intermediate transcript artifacts that are useful for inspection and debugging
- sends transcript emails to `TEST_RUN_RECIPIENTS` during test runs only; if that variable is unset, it falls back to `TRANSCRIPT_RECIPIENTS`

### Weekly Friday summary

On Friday runs, if `ENABLE_WEEKLY_SUMMARY=true`, the agent attempts the weekly summary after the main run path completes. This weekly summary attempt is independent of whether that day's transcript work succeeded, partially failed, or fully failed.

The agent checks transcripts this agent has already created and looks for episodes whose `processed_at` timestamp falls within the configured weekly summary lookback window. By default that window is the prior 7 days ending on Friday.

If any qualifying transcripts exist, the agent sends a single weekly roundup email to the same configured recipients as normal transcript emails using this subject format:

```txt
Compound Shows Summary - Week ending April 25, 2026
```

The weekly summary email body includes every qualifying episode from that lookback window. Each section contains the episode title and a no-more-than-4-paragraph summary intended to mention the important people, facts, companies, themes, claims, events, examples, and other notable details from that episode, with more detail than the earlier version of the feature. After each episode summary is generated, the agent also runs the summary text through `phrases-and-vocabulary/vocabulary-corrections.txt` before saving and emailing it. In the HTML version of the email, the summary body text is rendered at a larger font size for readability.

The agent records send history in `state/state.json` and will send at most one weekly summary email per `week_ending_YYYY-MM-DD` window, even if the agent is run multiple times that same day or retried after a failure.

After the summary is generated, the agent also saves local copies in `transcripts/`:

- `compound_shows_summary_week_ending_YYYY-MM-DD.txt`
- `compound_shows_summary_week_ending_YYYY-MM-DD.html`

### Weekly summary test mode

If you want to trigger only the weekly summary logic without fetching or transcribing new episodes, set this in `settings/transcription-settings.txt`:

```txt
WEEKLY_SUMMARY_TEST_MODE=true
```

In this mode, the agent skips RSS fetch/transcription work, runs only the weekly summary flow against already existing local transcripts, and then automatically resets `WEEKLY_SUMMARY_TEST_MODE=false` after the run completes.

Related settings in `settings/transcription-settings.txt`:

```txt
ENABLE_WEEKLY_SUMMARY=true
WEEKLY_SUMMARY_DAY=FRIDAY
WEEKLY_SUMMARY_LOOKBACK_DAYS=7
WEEKLY_SUMMARY_TEST_LOOKBACK_DAYS=7
WEEKLY_SUMMARY_MAX_PARAGRAPHS_PER_EPISODE=4
WEEKLY_SUMMARY_SAVE_TEXT_COPY=true
WEEKLY_SUMMARY_SAVE_HTML_COPY=true
WEEKLY_SUMMARY_EMAIL_SUBJECT_PREFIX=Compound Shows Summary
```

Optional selectors:
- leave `TEST_MODE_MONTH=` blank to use the default behavior and pull the most recent episodes
- set `TEST_MODE_MONTH` to a month name or abbreviation such as `January`, `Jan`, `February`, or `Feb` to pull test episodes from the most recent occurrence of that month within the last 12 months
- optionally set `TEST_MODE_SKIP_TITLE_CONTAINS` to exclude test candidates whose title contains a given substring
- after the run finishes successfully, test mode settings are automatically reset to:

```txt
TEST_MODE=false
TEST_MODE_EPISODE_COUNT=1
TEST_MODE_MONTH=
```

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
2. set `TEST_MODE_EPISODE_COUNT` to the number of recent episodes you want to inspect across all matching feeds
3. optionally set `TEST_MODE_MONTH` and/or `TEST_MODE_SKIP_TITLE_CONTAINS` if you want to narrow the candidate pool
4. run `./run-agent.sh` or `python3 agent.py`
5. review the intermediate files in `transcripts/`
6. confirm the settings were automatically restored after the run:

```txt
TEST_MODE=false
TEST_MODE_EPISODE_COUNT=1
TEST_MODE_MONTH=
```

If the run is interrupted before cleanup, reset those values manually.

## Correction reply capture

Outgoing transcript emails now include a correction request at the bottom of both the plain-text and HTML email bodies.

The exact text is:

```txt
Help us catch transcription errors.
Reply to this email. In the body of your reply email, start with the word "correction:", then type the actual transcript word or phrase that was mis-transcribed, a back slash "\", and the correct way to transcribe this word of phrase.
These will be human reviewed before implementing. Thank you!
```

A separate correction-ingest job reads reply emails and captures only replies whose first non-empty line starts with `correction:`.
Replies may come from any sender, including the same mailbox used to send or test transcript emails, as long as the reply matches a tracked transcript email.

What gets stored for each accepted correction:
- episode title
- sender name
- reply body

Storage location:
- `phrases-and-vocabulary/correction-requests/corrections-submission-animal-spirits.txt`
- `phrases-and-vocabulary/correction-requests/corrections-submission-the-compound-and-friends.txt`
- `phrases-and-vocabulary/correction-requests/corrections-submission-ask-the-compound.txt`
- `phrases-and-vocabulary/correction-requests/corrections-submission-masters-in-business.txt`
- `phrases-and-vocabulary/correction-requests/corrections-submission-at-the-money.txt`

Each accepted correction is appended as a single `|`-separated line. These files are local-only and are ignored by git.

### Running the correction-ingest job manually

```bash
./run-corrections-ingest.sh
```

Dry run:

```bash
./run-corrections-ingest.sh --dry-run
```

### Scheduled correction-ingest behavior

The correction-ingest job is intended to run weekly on Saturday.
It skips itself unless transcript emails were sent in the prior 7 days.

## Running the monitor

```bash
./run-monitor.sh
```

## Configuration

Main runtime settings live in:

- `settings/transcription-settings.txt`
- `settings/podcast_feeds.txt`
- `settings/service-endpoints.txt`
- `phrases-and-vocabulary/`

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
