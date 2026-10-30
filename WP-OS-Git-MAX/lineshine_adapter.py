"""LineShine HPC connection readiness checker. No remote execution by default."""
import argparse
import json
import os
from pathlib import Path

REQUIRED = ('LINESHINE_HOST', 'LINESHINE_USER', 'LINESHINE_SCHEDULER')
ALLOWED_SCHEDULERS = ('slurm', 'pbs')


def evaluate(environ=None):
    env = dict(os.environ if environ is None else environ)
    missing = [key for key in REQUIRED if not env.get(key, '').strip()]
    scheduler = env.get('LINESHINE_SCHEDULER', '').strip().lower()
    errors = []
    if scheduler and scheduler not in ALLOWED_SCHEDULERS:
        errors.append('Unsupported scheduler; expected slurm or pbs')
    if env.get('LINESHINE_HOST', '').strip() in ('localhost', '127.0.0.1'):
        errors.append('Remote host cannot be localhost')
    return {'provider': 'LineShine (planned)', 'remote_connected': False,
            'job_submitted': False, 'execution_verified': False,
            'configured': not missing and not errors,
            'missing_environment_variables': missing, 'errors': errors,
            'note': 'Configuration is not proof of authorization or connection. No network call is made.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    status = evaluate()
    report = json.dumps(status, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + '\n', encoding='utf-8')
    print(report)


if __name__ == '__main__':
    main()
