# read-footage: prior tested runs and lessons

For a source this skill has already read and scored (REASSEMBLE today).

- **Start from what was frozen, not from scratch.** Reuse the source's conventions, the hole/part tables and the rules in `references/<source>.md`; each rule there carries the held-out number it earned (REASSEMBLE: segment F1 0.926 → 0.961, success bacc 0.649 → 0.892 on 29 held-out recordings, `references/reassemble.md`).
- **Re-score before trusting on new data.** Run the frozen rules on the new recordings' tuning part first; a rule that was tuned on train is only evidence on held-out recordings (`references/reassemble.md` Protocol).
- **Read the open items first.** "Still missed" and "Broke or misled" entries name the cases the rules do not cover; label those by hand and mark them unverified.
- **Never read exam recordings** to check a rule (rules.md; `playbook/exam_burned.json`).
- **Add, don't overwrite.** A new finding is a dated Lessons entry with its evidence (`append_lesson`), and the source reference gets the detail.
