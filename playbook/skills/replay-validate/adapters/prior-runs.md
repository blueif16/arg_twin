# replay-validate: prior tested runs

- **Start from the frozen replay params version** and its scored distribution (`references/reassemble.md` Lessons, B2 rounds); a change needs a new run, because `record_check` judges against validation.yaml as it is (`playbook/orchestrator.md` Watching 1).
- **Parameters marked not identifiable stay sampled** in cousins (`sim-integrity`), never fixed to a fitted value.
