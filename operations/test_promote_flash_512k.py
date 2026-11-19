import json
from pathlib import Path
import unittest
from unittest.mock import patch
import urllib.error
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from email.message import Message

import promote_flash_512k as p


class PromotionTests(unittest.TestCase):
    def test_gateway_waits_for_connection_and_readiness(self):
        unavailable = urllib.error.HTTPError('http://gateway/readyz', 503, 'not ready', Message(), None)
        with patch.object(p, 'http', side_effect=[urllib.error.URLError('connection refused'),
                          unavailable, {'status': 'ready'}]) as request, patch.object(p.time, 'sleep'):
            value = p.wait_http('http://gateway/readyz', 'private', ready=lambda value: value['status'] == 'ready')
        self.assertEqual(value['status'], 'ready')
        self.assertEqual(request.call_count, 3)
        self.assertEqual(request.call_args.args, ('http://gateway/readyz', 'private'))

    def test_gateway_auth_errors_are_not_hidden_as_startup_delays(self):
        unauthorized = urllib.error.HTTPError('http://gateway/readyz', 401, 'unauthorized', Message(), None)
        with patch.object(p, 'http', side_effect=unauthorized), patch.object(p.time, 'sleep') as sleep:
            with self.assertRaises(urllib.error.HTTPError):
                p.wait_http('http://gateway/readyz', 'wrong')
            sleep.assert_not_called()

    def test_readiness_wait_has_a_deadline(self):
        with patch.object(p.time, 'monotonic', side_effect=[10, 10, 12, 12]), \
             patch.object(p.time, 'sleep'), patch.object(p, 'http', side_effect=ConnectionRefusedError('not listening')):
            with self.assertRaisesRegex(TimeoutError, 'did not become ready'):
                p.wait_http('http://gateway/healthz', seconds=1)

    def test_real_delayed_http_listener_reproduces_original_startup_race(self):
        with socket.socket() as reserve:
            reserve.bind(('127.0.0.1', 0))
            port = reserve.getsockname()[1]
        ready = threading.Event()
        holder = {}
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(b'{"status":"ok"}')
            def log_message(self, format, *args):
                pass
        def serve():
            time.sleep(0.1)
            server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
            holder['server'] = server
            ready.set()
            server.serve_forever(poll_interval=0.01)
        thread = threading.Thread(target=serve, daemon=True)
        thread.start()
        try:
            value = p.wait_http(f'http://127.0.0.1:{port}/healthz', seconds=3, poll_seconds=0.02)
            self.assertEqual(value['status'], 'ok')
        finally:
            if ready.wait(timeout=2):
                holder['server'].shutdown()
                holder['server'].server_close()
            thread.join(timeout=2)

    def test_gateway_credentials_and_unrelated_lines_are_preserved(self):
        text = ('# operator configuration\nLAN_GATEWAY_CLIENT_TOKEN="private-value"\n'
                'LAN_INFERENCE_BACKENDS=old\nEXTRA_SETTING=keep\n')
        gateway, _ = p.profile_values()
        updated = p.patch_env(text, gateway)
        self.assertIn('LAN_GATEWAY_CLIENT_TOKEN="private-value"\n', updated)
        self.assertIn('EXTRA_SETTING=keep\n', updated)
        self.assertEqual(updated.count('LAN_INFERENCE_BACKENDS='), 1)
        self.assertNotIn('LAN_INFERENCE_BACKENDS=old', updated)

    def test_duplicate_public_setting_does_not_override_new_value(self):
        updated = p.patch_env('VALUE=old\nVALUE=older\n', {'VALUE': 'new'})
        self.assertEqual(updated, 'VALUE=new\n')

    def test_models_and_gpu_ownership_match_topology(self):
        gateway, metrics = p.profile_values()
        backends = json.loads(gateway['LAN_INFERENCE_BACKENDS'])
        self.assertEqual({b['base_url'] for b in backends},
                         {'http://127.0.0.1:8081', 'http://127.0.0.1:8082'})
        self.assertNotIn('muse', gateway['LAN_INFERENCE_BACKENDS'])
        pools = json.loads(gateway['LAN_INFERENCE_POOLS'])
        self.assertEqual(pools[0]['replicas'], ['qwen_atx_gpu2'])
        ownership = [g for backend in json.loads(metrics['AI_METRICS_BACKENDS']).values() for g in backend['gpus']]
        self.assertCountEqual(ownership, ['0', '1', '2'])
        self.assertEqual(len(ownership), len(set(ownership)))

    def test_production_model_flags_equal_validated_512k_flags(self):
        production = json.loads((p.OPS / 'config/flash-512k.json').read_text())
        baseline = json.loads((p.ROOT / 'research/flash-reliability-2026-11-05/full-matrix/2gpu-512k/mtp-effective-config.json').read_text())
        self.assertEqual(production['args'], baseline['args'])
        self.assertEqual(production['parallel'], 1)
        self.assertEqual(production['gpu'], [0, 1])
        self.assertEqual(production['env']['CUDA_VISIBLE_DEVICES'].split(','), [p.EXPECTED['0'], p.EXPECTED['1']])
        self.assertNotIn(p.EXPECTED['2'], production['env']['CUDA_VISIBLE_DEVICES'])

    def test_flash_service_prevents_conflicting_boot_services(self):
        text = (p.OPS / 'systemd/strata-flash-512k.service').read_text()
        conflicts = next(line for line in text.splitlines() if line.startswith('Conflicts=')).split('=', 1)[1].split()
        self.assertCountEqual(conflicts, p.OLD)
        self.assertIn('WantedBy=multi-user.target', text)
        self.assertIn('KillMode=control-group', text)


if __name__ == '__main__':
    unittest.main()
