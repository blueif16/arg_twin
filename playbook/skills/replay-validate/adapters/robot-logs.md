# replay-validate: robot logs + cameras

- **REASSEMBLE:** Cartesian impedance Franka; rebuild x_d from the log, tare F/T in the tool frame on the window's first 10 frames, leave joint friction out (the rebuilt x_d already holds it), grasp as a force-limited tendon servo released by the logged finger width; window from onset − 1.5 s to release + 1 s; onset at |F − F_ref| > 2 N (`references/reassemble.md` §1–§4).
- **SO-101:** position-controlled servos driven by the logged leader-arm commands; no F/T, so outcome and timing are the checks (place vs drop), with the T2 negative control: brick start 20 mm off (`references/so101.md`).
- **Read outcomes after release:** operators release once the tip is in and gravity finishes (`references/reassemble.md` Lessons).
