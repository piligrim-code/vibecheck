import ast
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from examples.synthetic_demo import summarize, weighted_vibe
from tools.check_public_tree import ALLOWED_CSV, inspect_file

ROOT = Path(__file__).resolve().parents[1]


class PublicationTests(unittest.TestCase):
    def notebook(self, **cell_changes):
        cell = {'cell_type': 'code', 'metadata': {}, 'source': ['x = 1\n'],
                'outputs': [], 'execution_count': None}
        cell.update(cell_changes)
        return json.dumps({'cells': [cell], 'metadata': {}}).encode()

    def test_only_synthetic_csv_allowed(self):
        self.assertEqual(inspect_file('examples/synthetic_messages.csv', ALLOWED_CSV.encode()), [])
        self.assertEqual(inspect_file('examples/synthetic_messages.csv', ALLOWED_CSV.replace('\n', '\r\n').encode()), [])
        for path in ('data.csv', 'model/datasets/data/data1.csv', 'examples/another.csv'):
            with self.subTest(path=path):
                self.assertIn('unapproved_csv', inspect_file(path, b'message,score\n'))

    def test_synthetic_fixture_cannot_be_replaced_with_real_data(self):
        self.assertIn('unapproved_csv', inspect_file('examples/synthetic_messages.csv', b'message,score\n'))

    def test_data_directory_rejects_even_unknown_formats(self):
        self.assertIn('bundled_data', inspect_file('model/datasets/data/export.txt', b'example'))
        self.assertEqual(inspect_file('model/datasets/data/README.md', b'Local only'), [])

    def test_generated_artifacts_and_environment_files(self):
        for name in ('export.parquet', 'local.db', 'weights.pth', 'records.jsonl', '.env', '.env.production'):
            with self.subTest(name=name):
                self.assertTrue(inspect_file(name, b''))

    def test_credential_patterns_without_echoing_matches(self):
        samples = [b'123456789:' + b'A' * 35, b'sk-' + b'B' * 30,
                   b'ghp_' + b'C' * 30, b'-' * 5 + b'BEGIN RSA PRIVATE KEY' + b'-' * 5]
        for sample in samples:
            findings = inspect_file('example.txt', sample)
            self.assertEqual(findings, ['credential_shape'])
            self.assertNotIn(sample.decode(), json.dumps(findings))

    def test_password_database_uri(self):
        uri = b'postgresql://' + b'example:placeholder@localhost/db'
        self.assertIn('password_in_database_uri', inspect_file('example.txt', uri))

    def test_machine_paths(self):
        for value in (b'C:' + b'\\Users\\example\\file', b'/' + b'home/example/file'):
            with self.subTest(kind='machine path'):
                self.assertIn('machine_path', inspect_file('example.txt', value))

    def test_author_attribution_is_not_a_secret(self):
        self.assertEqual(inspect_file('source.py', b'# Author: contributor@example.com\nx = 1\n'), [])

    def test_clean_notebook(self):
        self.assertEqual(inspect_file('notebook.ipynb', self.notebook()), [])

    def test_saved_outputs(self):
        raw = self.notebook(outputs=[{'output_type': 'stream', 'text': ['synthetic output']}])
        self.assertIn('notebook_saved_output', inspect_file('notebook.ipynb', raw))

    def test_execution_counter(self):
        self.assertIn('notebook_saved_output', inspect_file('notebook.ipynb', self.notebook(execution_count=1)))

    def test_cell_metadata_and_attachments(self):
        for change in ({'metadata': {'execution': 'old'}}, {'attachments': {'image': 'placeholder'}}):
            self.assertIn('notebook_cell_metadata_or_attachment', inspect_file('notebook.ipynb', self.notebook(**change)))

    def test_widget_metadata(self):
        doc = json.loads(self.notebook())
        doc['metadata']['widgets'] = {'state': 'placeholder'}
        self.assertIn('notebook_metadata', inspect_file('notebook.ipynb', json.dumps(doc).encode()))

    def test_embedded_record_rejected(self):
        raw = self.notebook(source=["records = [{'sender': 'example', 'message': 'example'}]\n"])
        self.assertIn('notebook_embedded_record', inspect_file('notebook.ipynb', raw))

    def test_invalid_notebook_and_python(self):
        self.assertIn('invalid_notebook', inspect_file('bad.ipynb', b'not json'))
        self.assertIn('invalid_notebook', inspect_file('bad.ipynb', self.notebook(source=['x ='])))
        self.assertIn('python_syntax', inspect_file('bad.py', b'x ='))

    def test_malformed_notebook_structure_is_a_finding(self):
        for payload in (b'[]', b'null', b'{"cells": [], "metadata": []}'):
            self.assertIn('invalid_notebook', inspect_file('bad.ipynb', payload))

    def test_historical_csvs_absent(self):
        names = ('cleaned_dataset.csv', 'data1.csv', 'data2.csv', 'korean_students.csv', 'maistudents.csv')
        for name in names:
            self.assertFalse((ROOT / 'model/datasets/data' / name).exists())

    def test_every_notebook_has_no_saved_data(self):
        notebooks = list(ROOT.rglob('*.ipynb'))
        self.assertEqual(len(notebooks), 5)
        for path in notebooks:
            with self.subTest(path=path.relative_to(ROOT)):
                self.assertEqual(inspect_file(path.name, path.read_bytes()), [])

    def test_notebook_csv_inputs_are_explicit(self):
        reads = 0
        for path in ROOT.rglob('*.ipynb'):
            for cell in json.loads(path.read_text(encoding='utf-8'))['cells']:
                if cell['cell_type'] != 'code':
                    continue
                for node in ast.walk(ast.parse(''.join(cell['source']))):
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'read_csv':
                        reads += 1
                        self.assertEqual(ast.unparse(node.args[0]), "os.environ['VIBECHECK_INPUT_CSV']")
        self.assertEqual(reads, 3)

    def test_database_notebook_requires_explicit_configuration(self):
        path = ROOT / 'metric/query.ipynb'
        sources = [''.join(c['source']) for c in json.loads(path.read_text())['cells'] if c['cell_type'] == 'code']
        source = '\n'.join(sources)
        self.assertIn('os.environ["VIBECHECK_DATABASE_URL"]', source)
        self.assertIn('readonly=True', source)
        self.assertIn('SELECT COUNT(*)', source)
        self.assertNotIn('fetchall', source)


class SyntheticDemoTests(unittest.TestCase):
    def test_known_single_scores(self):
        self.assertEqual(weighted_vibe([1]), 0)
        self.assertEqual(weighted_vibe([0]), 1)
        self.assertEqual(weighted_vibe([-1]), 3)

    def test_invalid_scores(self):
        for values in ([], [2], [-2], [float('nan')], [float('inf')], ['1'], [True]):
            with self.subTest(values_type=str(type(values[0])) if values else 'empty'):
                with self.assertRaises(ValueError):
                    weighted_vibe(values)

    def test_more_recent_negative_score_has_higher_weight(self):
        self.assertGreater(weighted_vibe([1, -1]), weighted_vibe([-1, 1]))

    def test_fixture_summary(self):
        result = summarize(ROOT / 'examples/synthetic_messages.csv')
        self.assertEqual(result['rows'], 4)
        self.assertGreater(result['weighted_vibe'], 0)

    def test_wrong_csv_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'synthetic.csv'
            path.write_text('wrong,columns\n1,2\n')
            with self.assertRaises(ValueError):
                summarize(path)

    def test_demo_from_another_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            output = subprocess.check_output([sys.executable, str(ROOT / 'examples/synthetic_demo.py')],
                                             cwd=directory, text=True)
        self.assertEqual(json.loads(output)['rows'], 4)
        self.assertNotIn('Synthetic example', output)

    def test_metric_notebook_core_uses_only_synthetic_records(self):
        doc = json.loads((ROOT / 'metric/metric_and_visualisation.ipynb').read_text(encoding='utf-8'))
        namespace = {}
        with contextlib.redirect_stdout(io.StringIO()):
            for cell in doc['cells']:
                if cell['cell_type'] != 'code':
                    continue
                source = ''.join(cell['source'])
                if 'matplotlib' in source:
                    break
                exec(compile(source, '<synthetic-notebook>', 'exec'), namespace)
        self.assertEqual(len(namespace['metric']), 20)
        self.assertEqual(len(namespace['json_dicts']), 100)
        self.assertTrue(all(d['user'] == 'synthetic-user' and d['message'] == 'Synthetic sample'
                            for d in namespace['json_dicts']))
        for scores, value in zip(namespace['sentiment_lists'], namespace['metric']):
            self.assertAlmostEqual(weighted_vibe(scores), value)


if __name__ == '__main__':
    unittest.main()
