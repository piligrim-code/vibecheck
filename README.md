# Vibecheck

Historical Telegram sentiment-monitoring prototype. The source includes a
bot, PostgreSQL helpers, an experimental aggregate metric and BERT research
notebooks. It is not a production-ready moderation system, and this revision
does not establish model accuracy or reliability.

## Data cleanup

At the owner's request, the five historical CSV datasets, embedded notebook
records, saved notebook outputs/metadata and old chat screenshot references
were removed from this revision. The database notebook now requires an
explicit environment variable instead of an embedded connection string.
Its example query returns a count, not message records.

**This is current-tree cleanup, not a Git-history purge.** Older commits,
existing clones, forks and externally hosted attachments may still contain
previous content. Old credentials must be assessed and revoked/rotated by
their owner if they were real; age does not establish that a credential is
inactive. No credentials were tested as part of this cleanup.

Do not reintroduce real chat exports or notebook outputs. Use local reviewed
data outside Git. Automated checks reject unapproved CSVs, common generated
data files, saved notebook outputs and selected credential/path patterns.
These checks are not comprehensive PII or secret detection, and do not prove
consent or publication rights for arbitrary new content.

## Runnable offline example

Python 3.12 or newer; no third-party packages, services or credentials needed:

```console
python examples/synthetic_demo.py
python -m unittest discover -s tests -v
python tools/check_public_tree.py
```

The demo applies the notebook's arithmetic formula to four hand-written
synthetic messages with manually assigned scores. It prints only a row count
and aggregate. It does not run a sentiment model, measure accuracy or assess
a person's mood. The metric notebook also uses explicitly synthetic records.

`check_public_tree.py` examines the Git index, not history or unstaged edits.
Run it after staging intended changes and before committing. CI runs the same
check on checkout, plus synthetic tests on Linux and Windows.

## Research sources

- `main.py` and `psycopg.py`: historical bot and database integration. Running
  the bot can send message text to OpenAI and persist sender/message details.
  Do not use it on real chats without reviewing data handling and permissions.
- `metric/`: aggregate-metric exploration and PostgreSQL source. The database
  notebook requires `VIBECHECK_DATABASE_URL` and explicit execution; it does
  not include credentials or contact a service during the offline tests.
- `model/`: BERT architecture and historical training/evaluation fragments.
  Notebook CSV reads require `VIBECHECK_INPUT_CSV`; the cleaning notebook's
  export requires `VIBECHECK_OUTPUT_CSV`. Input schemas differ by notebook.
  Running legacy cells may download models or contact experiment tracking.
- `model/datasets/parse_tg.py`: attributed third-party export parser. Its
  output includes identifying fields; keep it local. Attribution is retained
  and is not a claim that third-party code/data rights have been verified.

## Known limitations

The bot imports an absent `alive` module. Training scripts depend on notebook
globals/checkpoints and contain unresolved runtime issues. `model/predict.py`
has legacy import/path/variable mismatches. The bot repeats model calls for
one message and lacks robust response validation; database identifier handling
and personal-data logging also need review. The historical requirements are
not a tested deployment environment.

The offline tests qualify only publication guards, synthetic arithmetic and
selected sanitized notebook behavior. They do not qualify Telegram, OpenAI,
PostgreSQL, BERT training or the complete notebooks. Reproducible evaluation,
dependency repair, permission/retention controls and service integration tests
are required before presenting this as a usable product.

No new repository license or rights to historical data are granted by this
cleanup. Code and tokenizer provenance still need a separate licensing review.
