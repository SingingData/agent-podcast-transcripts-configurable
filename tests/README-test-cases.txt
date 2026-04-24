Non-agent sanity harness test cases

Files:
- sample_segments.json: fixed transcript-segment fixtures for helper-only tests
- sanity_checks.py: runs helper-level validation without launching the full agent

Current checks:
- settings files exist and load
- opening ad stripping removes matched leading ad copy
- opening catch phrases help find the real podcast intro
- known hosts load from phrases-and-vocabulary/known-hosts-per-podcast.txt
- speaker mapping assigns known hosts and extracted guest names
- diarization import path resolves via whisperx.diarize.DiarizationPipeline
