# Localizing a fixture from robot contacts

## Question

How do we place a workpiece or fixture (here, the task board and its holes) in the robot frame from the robot's own contact data, when the grasped part's pose in the hand is uncertain and changes with every grasp?

## What is established, with numbers

- **Contact points plus force normals register a known CAD model.** A robust cost does it: 0.51 mm and 0.55° mean error on a real setup with Geman–McClure [kim-2023].
- **Peg insertions can serve as registration fiducials.**
  - NIST recorded robot poses at manual peg insertions and registered the nominal hole layout to them with a rigid point-based fit [cheok-2018].
  - Judge the fit at targets that were not used to fit it (TRE), not at the fiducials (FRE) [cheok-2018].
  - Their correction reduced failed insertions from 51% to 13% for tight holes [cheok-2018].
- **The gripper pose is not the part pose.**
  - Misalignment torque during insertion makes the part slip in the gripper [chen-2022].
  - In-hand pose can be estimated to under 1 mm by touching a known environment with a particle filter [drigalski-2020].
- **Estimate the in-hand offset as its own variable.** Factor-graph estimators treat it as a per-grasp rest pose, shared by every contact made during that grasp [vandermerwe-2026, kimrodriguez-2022].
- **Use particle filters when you don't know which feature was touched.** They work over contact configurations or probing hypotheses [taguchi-2010; chhatpar-2005].

## What works, what doesn't, what's efficient

- **Works:**
  - robust least squares (Cauchy or Geman–McClure) over many contact events, with a rigid model of the fixture [kim-2023];
  - scoring at held-out targets [cheok-2018].
- **Doesn't work:** treating the TCP as the part position at each event when the in-hand pose varies per grasp [chen-2022, drigalski-2020].
- **Efficient:** when every event is labelled with the feature it belongs to, the problem is a small nonlinear least-squares fit. It takes milliseconds per recording, and no particle filter is needed.

## Disagreements and open questions

- Kim et al. use force-derived surface normals at sliding contacts [kim-2023]. Insertion end-states carry no useful normal, because the part is inside the hole. Whether onsets or failed attempts add information here is open.

## For us

- For us: the per-grasp offset reduces to one dimension. Parallel coupled fingers centre the part across the pads, so an event constrains the hole to a line along the pad axis.
- For us, on REASSEMBLE: this point-to-line model gives a 1.23 mm held-out across-pad error on 24 gate recordings, against 4.08 mm for the mocap board pose. Method and numbers: `playbook/skills/board-from-contacts`; tools: `board_err`, `board_from_contacts`.
