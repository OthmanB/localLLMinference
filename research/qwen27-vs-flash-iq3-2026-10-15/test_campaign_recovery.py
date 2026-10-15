"""No-service-control tests for verified process restart reconciliation."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_campaign as campaign


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        campaign.STOP.clear()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)

    def test_identical_configuration_accepts_restart_and_records_it(self):
        with patch.object(campaign, 'SERVICE_TEXTS', {'svc': 'same'}), \
             patch.object(campaign, 'RUN_FOR_NOTES', self.folder), \
             patch.object(campaign, 'unit', side_effect=lambda name, prop: '200' if prop == 'MainPID' else 'active'), \
             patch.object(campaign.subprocess, 'check_output', return_value='same'):
            pids = {'svc': '100'}
            campaign.protect(pids)
            self.assertEqual(pids['svc'], '200')
            events = json.loads((self.folder / 'service-restarts.json').read_text())
            self.assertTrue(events[0]['configuration_unchanged'])

    def test_configuration_change_still_aborts(self):
        with patch.object(campaign, 'SERVICE_TEXTS', {'svc': 'original'}), \
             patch.object(campaign, 'RUN_FOR_NOTES', self.folder), \
             patch.object(campaign, 'unit', return_value='different'), \
             patch.object(campaign.subprocess, 'check_output', return_value='changed'):
            with self.assertRaisesRegex(RuntimeError, 'configuration changed'):
                campaign.protect({'svc': '100'})

    def test_repeated_restarts_are_not_ignored(self):
        (self.folder / 'service-restarts.json').write_text(json.dumps([{}, {}, {}]))
        with patch.object(campaign, 'SERVICE_TEXTS', {'svc': 'same'}), \
             patch.object(campaign, 'RUN_FOR_NOTES', self.folder), \
             patch.object(campaign, 'unit', side_effect=lambda name, prop: '200' if prop == 'MainPID' else 'active'), \
             patch.object(campaign.subprocess, 'check_output', return_value='same'):
            with self.assertRaisesRegex(RuntimeError, 'Repeated'):
                campaign.protect({'svc': '100'})


if __name__ == '__main__':
    unittest.main()
