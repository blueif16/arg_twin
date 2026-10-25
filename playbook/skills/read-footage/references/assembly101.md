# read-footage: Assembly101 (human toy assembly, video only)

Evidence: `runs/argon/probes/longvideo/` (CC BY-NC 4.0 sample under `data/assembly101-sample/`; 4 recordings up to 13.5 min, frontal exocentric view C10118).

- **Scores:** every variant F1 0.45–0.49, task accuracy 0.13–0.42, mistake bacc 0.64–0.74; single pass caught 14 of 18 mistakes with 15 false alarms; zooming did not help.
- **Why:** dataset conventions, not perception. Four wheel attachments are labelled as one step; part names differ (speaker vs sound module).
- **Do next time:** put the step definitions and a part glossary with one crop per part in the prompt; ask for a restatement before labelling (longvideo proposals 4–5).
