#!/usr/bin/env python3
"""Publish allowlisted original40/additional40/combined80 reports only."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

from publish_results import git

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--public-repo',type=Path)
    p.add_argument('--wait-unit')
    p.add_argument('--commit-and-push',action='store_true')
    a = p.parse_args()
    if a.wait_unit:
        while subprocess.run(['systemctl','--user','show',a.wait_unit,'-p','ActiveState','--value'],
                             text=True,capture_output=True).stdout.strip() in ('active','activating','deactivating'):
            time.sleep(60)
    subprocess.run(['python3',str(HERE/'combined_swe_report.py'),'--study-root',str(HERE),
                    '--output-dir',str(HERE/'combined-report')],check=True)
    if a.public_repo is None:
        return
    if a.commit_and_push:
        if git(a.public_repo,'status','--porcelain').strip():
            raise RuntimeError('Other public changes exist; private combined reports are preserved')
        git(a.public_repo,'fetch','origin')
        git(a.public_repo,'merge','--ff-only','origin/main')
    destination = a.public_repo/'research'/HERE.name
    paths = []
    report = json.loads((HERE/'combined-report/combined-swe.json').read_text())
    ready = report['integration_gate']['ready_for_complete_comparison']
    report['report_kind'] = 'published_complete_cohort' if ready else 'published_incomplete_snapshot'
    report['final_claim'] = bool(ready)
    text = (HERE/'combined-report/COMBINED_SWE.md').read_text()
    text = text.replace('— unpublished snapshot','— expanded evaluation',1)
    text = text.replace('**Not a final accuracy or publication claim.**',
                        '**Complete selected-cohort evaluation; see uncertainty and scope below.**' if ready else
                        '**Incomplete snapshot: pending outcomes are not a final accuracy score.**',1)
    (destination/'COMBINED_SWE.md').write_text(text)
    (destination/'combined-swe.json').write_text(json.dumps(report,indent=2)+'\n')
    for name in ('COMBINED_SWE.md','combined-swe.json'):
        paths.append(f'research/{HERE.name}/{name}')
    manifest_path = destination/'EXTRA_PUBLICATION.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        manifest['updated_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
        manifest['integration_gate'] = report['integration_gate']
        for item in manifest.get('included',[]):
            item['sha256'] = hashlib.sha256((destination/item['path']).read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
        paths.append(f'research/{HERE.name}/EXTRA_PUBLICATION.json')
    if a.commit_and_push:
        git(a.public_repo,'status','--short')
        git(a.public_repo,'diff','--check')
        git(a.public_repo,'diff','--',*paths)
        git(a.public_repo,'log','--oneline','-10')
        git(a.public_repo,'add','--',*paths)
        if git(a.public_repo,'diff','--cached','--name-only').splitlines() != sorted(paths):
            raise RuntimeError('Unexpected staged changes; no commit made')
        git(a.public_repo,'commit','-m','Report expanded paired SWE-bench cohort')
        git(a.public_repo,'push','origin','main')


if __name__ == '__main__':
    main()
