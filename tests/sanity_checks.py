#!/usr/bin/env python3
import ast
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SETTINGS_DIR = BASE_DIR / "settings"
PHRASES_DIR = BASE_DIR / 'phrases-and-vocabulary'
FIXTURES_FILE = Path(__file__).resolve().parent / "sample_segments.json"
AGENT_FILE = BASE_DIR / "agent.py"

KEEP_FUNCTIONS = {
    'load_key_value_settings',
    'load_phrase_list',
    'load_known_hosts',
    'get_podcast_hosts',
    'extract_guest_names_from_text',
    'extract_guest_from_transcript_intro',
    'detect_hosts_from_intro_phrases',
    'strip_opening_ad_segments',
    'get_first_speaker',
    'map_speakers_to_names',
}

ASSIGN_NAMES = set()


class TestLog:
    def __init__(self):
        self.lines = []

    def info(self, *args, **kwargs):
        self.lines.append(('INFO', ' '.join(str(a) for a in args)))

    def warning(self, *args, **kwargs):
        self.lines.append(('WARNING', ' '.join(str(a) for a in args)))

    def error(self, *args, **kwargs):
        self.lines.append(('ERROR', ' '.join(str(a) for a in args)))



def load_agent_helpers():
    src = AGENT_FILE.read_text(encoding='utf-8')
    mod = ast.parse(src, filename=str(AGENT_FILE))
    new_body = []
    for node in mod.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import):
                kept = [n for n in node.names if n.name in {'os', 're', 'json'}]
                if kept:
                    node.names = kept
                    new_body.append(node)
            else:
                if node.module in {'pathlib'}:
                    new_body.append(node)
        elif isinstance(node, ast.Assign):
            names = {t.id for t in node.targets if isinstance(t, ast.Name)}
            if names & ASSIGN_NAMES:
                new_body.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in KEEP_FUNCTIONS:
            new_body.append(node)
    subset = ast.Module(body=new_body, type_ignores=[])
    ns = {}
    exec(compile(subset, str(AGENT_FILE), 'exec'), ns)
    ns['SETTINGS_DIR'] = str(SETTINGS_DIR)
    ns['KNOWN_HOSTS_FILE'] = str(PHRASES_DIR / 'known-hosts-per-podcast.txt')
    ns['OPENING_CATCH_PHRASES_FILE'] = str(PHRASES_DIR / 'opening-catch-phrases.txt')
    ns['OPENING_AD_PHRASES_FILE'] = str(PHRASES_DIR / 'opening-ad-phrases.txt')
    ns['OPENING_CATCH_PHRASES'] = ns['load_phrase_list'](ns['OPENING_CATCH_PHRASES_FILE'])
    ns['OPENING_AD_PHRASES'] = ns['load_phrase_list'](ns['OPENING_AD_PHRASES_FILE'])
    ns['log'] = TestLog()
    return ns



def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)



def run():
    helper = load_agent_helpers()
    fixtures = json.loads(FIXTURES_FILE.read_text(encoding='utf-8'))

    required_files = [
        PHRASES_DIR / 'known-hosts-per-podcast.txt',
        PHRASES_DIR / 'opening-catch-phrases.txt',
        PHRASES_DIR / 'opening-ad-phrases.txt',
        SETTINGS_DIR / 'service-endpoints.txt',
    ]
    for path in required_files:
        assert_true(path.exists() and path.stat().st_size > 0, f'Missing or empty settings file: {path.name}')

    hosts = helper['get_podcast_hosts']('Animal Spirits')
    assert_true(hosts == ['Ben Carlson', 'Michael Batnick'], f'Unexpected Animal Spirits hosts: {hosts}')

    case1 = fixtures['animal_spirits_intro_with_ad']
    stripped = helper['strip_opening_ad_segments'](case1['segments'], helper['OPENING_AD_PHRASES'], max_start_seconds=120)
    assert_true(len(stripped) == 3, f'Expected 3 segments after ad stripping, got {len(stripped)}')
    assert_true('sponsored by' not in stripped[0]['text'].lower(), 'Opening ad segment was not stripped')

    detected = helper['detect_hosts_from_intro_phrases'](stripped, helper['get_podcast_hosts'](case1['podcast_name']))
    assert_true(detected.get('SPEAKER_00') == 'Ben Carlson', f'Expected SPEAKER_00 -> Ben Carlson, got {detected}')
    assert_true(detected.get('SPEAKER_01') == 'Michael Batnick', f'Expected SPEAKER_01 -> Michael Batnick, got {detected}')

    mapped = helper['map_speakers_to_names'](
        stripped,
        case1['podcast_name'],
        episode_title=case1['episode_title'],
        episode_summary=case1['episode_summary'],
    )
    assert_true(mapped.get('SPEAKER_00') == 'Ben Carlson', f'Host map failed: {mapped}')
    assert_true(mapped.get('SPEAKER_01') == 'Michael Batnick', f'Second host map failed: {mapped}')
    assert_true(mapped.get('SPEAKER_99') is None, 'Stripped ad speaker should not remain in map')

    case2 = fixtures['compound_intro_no_ad']
    mapped2 = helper['map_speakers_to_names'](
        case2['segments'],
        case2['podcast_name'],
        episode_title=case2['episode_title'],
        episode_summary=case2['episode_summary'],
    )
    assert_true(mapped2.get('SPEAKER_00') == 'Josh Brown', f'Expected Josh Brown first: {mapped2}')
    assert_true(mapped2.get('SPEAKER_01') == 'Michael Batnick', f'Expected Michael Batnick second: {mapped2}')

    settings = helper['load_key_value_settings'](str(SETTINGS_DIR / 'transcription-settings.txt'))
    assert_true(settings.get('TEST_MODE', '').lower() == 'false', f"Expected TEST_MODE=false, got {settings.get('TEST_MODE')}")
    assert_true(settings.get('TEST_MODE_EPISODE_COUNT') == '1', f"Expected TEST_MODE_EPISODE_COUNT=1, got {settings.get('TEST_MODE_EPISODE_COUNT')}")

    print('sanity_checks: PASS')


if __name__ == '__main__':
    try:
        run()
    except Exception as e:
        print(f'sanity_checks: FAIL - {e}')
        sys.exit(1)
