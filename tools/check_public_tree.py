"""Check staged Git blobs for publication regressions; never print matched data."""
import ast
import json
from pathlib import Path, PurePosixPath
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_CSV = 'examples/synthetic_messages.csv'
ALLOWED_CSV = (
    'message,sentiment_score\n'
    'Synthetic example: the test completed.,0.5\n'
    'Synthetic example: the task is queued.,0.0\n'
    'Synthetic example: the build failed.,-0.5\n'
    'Synthetic example: all checks passed.,1.0\n'
)
PATTERNS = {
    'credential_shape': re.compile(rb'(?:\b\d{7,12}:[A-Za-z0-9_-]{30,}\b|\bsk-[A-Za-z0-9_-]{20,}\b|\bgh[pousr]_[A-Za-z0-9]{25,}\b|\x2d{5}BEGIN [A-Z ]*PRIVATE KEY\x2d{5})'),
    'password_in_database_uri': re.compile(rb'(?i)postgres(?:ql)?://[^\s/:]+:[^\s/@]+@'),
    'machine_path': re.compile(rb'(?i)(?:[a-z]:[\\/](?:Users|proj)[\\/]|/(?:home|Users|content|kaggle)/)'),
}
BLOCKED_SUFFIXES = {'.jsonl', '.parquet', '.db', '.sqlite', '.sqlite3', '.log',
                    '.pkl', '.pickle', '.pt', '.pth', '.npy', '.npz', '.xlsx', '.zip'}


def inspect_file(name, content):
    issues = []
    path = PurePosixPath(name)
    if path.suffix.lower() == '.csv':
        if name != SYNTHETIC_CSV or content.replace(b'\r\n', b'\n') != ALLOWED_CSV.encode():
            issues.append('unapproved_csv')
    if name.startswith('model/datasets/data/') and name != 'model/datasets/data/README.md':
        issues.append('bundled_data')
    if path.suffix.lower() in BLOCKED_SUFFIXES:
        issues.append('generated_data_or_binary')
    if path.name == '.env' or (path.name.startswith('.env.') and path.name != '.env.example'):
        issues.append('environment_file')
    issues.extend(kind for kind, pattern in PATTERNS.items() if pattern.search(content))
    if path.suffix == '.ipynb':
        try:
            doc = json.loads(content)
            if doc.get('metadata', {}).keys() - {'kernelspec', 'language_info'}:
                issues.append('notebook_metadata')
            for cell in doc['cells']:
                if cell.get('outputs') or cell.get('execution_count') is not None:
                    issues.append('notebook_saved_output')
                if cell.get('attachments') or cell.get('metadata'):
                    issues.append('notebook_cell_metadata_or_attachment')
                if cell['cell_type'] == 'code':
                    tree = ast.parse(''.join(cell['source']))
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Dict):
                            fields = {k.value: v for k, v in zip(node.keys, node.values)
                                      if isinstance(k, ast.Constant) and isinstance(k.value, str)}
                            if fields.keys() & {'sender', 'sender_id', 'from_id', 'msg_content', 'user', 'message'}:
                                synthetic = all(isinstance(fields.get(k), ast.Constant)
                                                and fields[k].value == value
                                                for k, value in {'user': 'synthetic-user',
                                                                 'message': 'Synthetic sample'}.items())
                                if not synthetic:
                                    issues.append('notebook_embedded_record')
        except (ValueError, KeyError, TypeError, AttributeError, SyntaxError):
            issues.append('invalid_notebook')
    elif path.suffix == '.py':
        try:
            ast.parse(content)
        except (SyntaxError, ValueError):
            issues.append('python_syntax')
    return sorted(set(issues))


def main():
    paths = subprocess.check_output(['git', '-C', str(ROOT), 'ls-files', '-z']).decode().split('\0')
    findings = []
    count = 0
    for name in filter(None, paths):
        content = subprocess.check_output(['git', '-C', str(ROOT), 'show', ':' + name])
        count += 1
        kinds = inspect_file(name, content)
        if kinds:
            findings.append({'path': name, 'kinds': kinds})
    print(json.dumps({'staged_files': count, 'findings': findings}))
    return int(bool(findings))


if __name__ == '__main__':
    raise SystemExit(main())
