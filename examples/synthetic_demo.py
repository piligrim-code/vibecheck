"""Offline arithmetic example using assigned synthetic scores, not a model."""
import argparse
import csv
import json
import math
from pathlib import Path


def weighted_vibe(scores):
    """Apply the historical notebook formula to scores ordered oldest first."""
    scores = list(scores)
    if not scores:
        raise ValueError('At least one score is required')
    if any(isinstance(s, bool) or not isinstance(s, (int, float))
           or not math.isfinite(s) or not -1 <= s <= 1 for s in scores):
        raise ValueError('Scores must be finite numbers between -1 and 1')
    n = len(scores)
    return sum((2 ** (1 - s) - 1) / math.log2(n - i + 1)
               for i, s in enumerate(scores)) / n


def summarize(path):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['message', 'sentiment_score']:
            raise ValueError('Expected message,sentiment_score columns')
        scores = [float(row['sentiment_score']) for row in reader]
    return {'rows': len(scores), 'weighted_vibe': round(weighted_vibe(scores), 6)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path,
                        default=Path(__file__).with_name('synthetic_messages.csv'))
    args = parser.parse_args(argv)
    print(json.dumps(summarize(args.input)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
