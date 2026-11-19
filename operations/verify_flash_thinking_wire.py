#!/usr/bin/env python3
"""Check actual harness request fields through an ephemeral loopback proxy.

Only synthetic math prompts are sent; credentials and message content are not
printed or written to disk. The upstream endpoint remains the normal gateway.
"""
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading

import yaml

MODEL = 'ai-server/qwen3.8-flash-next-iq3_s'
CAPTURES = []


class Proxy(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get('Content-Length', '0')))
        req = json.loads(body)
        CAPTURES.append({k: req.get(k) for k in ('model', 'reasoning_effort', 'chat_template_kwargs',
                                                   'temperature', 'top_p', 'top_k', 'presence_penalty',
                                                   'max_tokens', 'max_completion_tokens')})
        conn = http.client.HTTPConnection('127.0.0.1', 8088, timeout=90)
        headers = {'Content-Type': 'application/json', 'Authorization': self.headers.get('Authorization', '')}
        conn.request('POST', self.path, body=body, headers=headers)
        response = conn.getresponse()
        self.send_response(response.status)
        self.send_header('Content-Type', response.getheader('Content-Type', 'application/json'))
        self.send_header('Connection', 'close')
        self.end_headers()
        try:
            while True:
                block = response.read1(8192)
                if not block:
                    break
                self.wfile.write(block)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # OpenCode may cancel its background title request on exit.
        finally:
            conn.close()
        self.close_connection = True

    def log_message(self, format, *args):
        pass


def main():
    token = os.environ.get('AI_SERVER_API_KEY')
    if not token:
        raise RuntimeError('AI_SERVER_API_KEY is not available in this shell')
    server = ThreadingHTTPServer(('127.0.0.1', 0), Proxy)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{server.server_address[1]}/v1'
    actual_models = yaml.safe_load(Path('/home/obenomar/.omp/agent/models.yml').read_text())
    actual_models['providers']['ai-server']['baseUrl'] = base
    rows = []
    try:
        with tempfile.TemporaryDirectory(prefix='flash-wire-', dir='/tmp/opencode') as temp:
            root = Path(temp)
            (root / 'models.yml').write_text(yaml.safe_dump(actual_models))
            (root / 'config.yml').write_text('modelRoles:\n  default: ' + MODEL + ':xhigh\n')
            env = {**os.environ, 'PI_CODING_AGENT_DIR': temp}
            for level in ('low', 'medium', 'high', 'xhigh', 'off', 'default'):
                before = len(CAPTURES)
                p = subprocess.run(['omp', '-p', '--mode', 'json', '--no-session', '--no-tools',
                    '--no-extensions', '--no-context-files', '--no-skills', '--max-time', '60s',
                    *([] if level == 'default' else ['--model', MODEL, '--thinking', level]),
                    'What is 17 times 19? Reply with the number.'],
                    cwd='/tmp/opencode', env=env, capture_output=True, text=True, timeout=80)
                if p.returncode or len(CAPTURES) != before + 1:
                    raise RuntimeError(f'OMP {level} failed: exit={p.returncode}, requests={len(CAPTURES)-before}, stderr={p.stderr[-500:]}')
                rows.append({'client': 'omp', 'level': level, 'request': CAPTURES[-1]})
            env = {**os.environ, 'OPENCODE_CONFIG_CONTENT': json.dumps({'provider': {'ai-server': {'options': {'baseURL': base}}}})}
            for level in ('low', 'medium', 'high', 'xhigh', 'none', 'default'):
                before = len(CAPTURES)
                p = subprocess.run(['opencode', '--standalone', 'run', '--format', 'json',
                    *([] if level == 'default' else ['-m', MODEL + '#' + level]),
                    'What is 17 times 19? Reply with the number.'],
                    cwd='/tmp/opencode', env=env, capture_output=True, text=True, timeout=90)
                if p.returncode or len(CAPTURES) <= before:
                    raise RuntimeError(f'OpenCode {level} failed: exit={p.returncode}, requests={len(CAPTURES)-before}, stderr={p.stderr[-500:]}')
                # OpenCode may also issue background title requests; retain all
                # requests and identify the one carrying the selected variant.
                choices = CAPTURES[before:]
                expected = 'xhigh' if level == 'default' else level
                chosen = next((r for r in choices if r.get('reasoning_effort') == expected), choices[0])
                rows.append({'client': 'opencode', 'level': level, 'request': chosen, 'requests': len(choices)})
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    for row in rows:
        req = row['request']
        kwargs = req.get('chat_template_kwargs') or {}
        level = row['level']
        if row['client'] == 'omp':
            actual = 'off' if kwargs.get('enable_thinking') is False else kwargs.get('reasoning_effort')
        else:
            actual = req.get('reasoning_effort')
        expected = 'xhigh' if level == 'default' else level
        row['wire_effort'] = actual
        row['passed'] = actual == expected
        print(json.dumps(row))
    if not all(row['passed'] for row in rows):
        raise SystemExit('Some clients did not transmit the intended thinking level')


if __name__ == '__main__':
    main()
