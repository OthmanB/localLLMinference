import copy
import json
from pathlib import Path
import unittest

import set_flash_thinking as s


class ThinkingConfigTests(unittest.TestCase):
    def test_only_default_effort_and_old_sampler_change(self):
        old = {'args': ['--max-context', '524288'], 'gpu': [0, 1],
               'reasoning_effort': 'none', 'sampling': copy.deepcopy(s.OLD_SAMPLING)}
        original = copy.deepcopy(old)
        new = s.desired_config(old)
        self.assertEqual(new, {'args': ['--max-context', '524288'], 'gpu': [0, 1],
                               'reasoning_effort': 'high'})
        self.assertEqual(old, original)

    def test_custom_sampler_is_not_silently_overwritten(self):
        with self.assertRaisesRegex(ValueError, 'Custom sampling'):
            s.desired_config({'sampling': {'temperature': 0.2}})

    def test_prepared_production_config_uses_supported_high_alias(self):
        cfg = json.loads((Path(__file__).parent / 'config/flash-512k.json').read_text())
        self.assertEqual(cfg['reasoning_effort'], 'high')
        self.assertNotIn('sampling', cfg)
        self.assertEqual(cfg['gpu'], [0, 1])

    def test_shared_settings_path_matches_strata(self):
        self.assertEqual(str(s.SHARED), '/etc/ai-server/flash-512k.shared-settings.json')


if __name__ == '__main__':
    unittest.main()
