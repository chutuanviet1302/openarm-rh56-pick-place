# custom_ros_messages — PLACEHOLDER, not the lab's real package

This is a **guessed, minimal stand-in** for the lab's internal `custom_ros_messages`
package, written only so `openarm_pick_place.ros2_nodes` can be exercised under
Docker while the real package is unavailable. It is **not** authoritative.

## What it's based on

Nothing here was invented from scratch: the two fields on `MotorCmd` are exactly
the two fields `openarm_pick_place/ros2_nodes.py` assigns before publishing
(`command.mode = 0`, `command.q = position`) — see `_hand()` in that file. That is
the only evidence available in this repo about the message shape. Everything else
about the real message (extra fields such as `dq`/`tau`/`kp`/`kd` common in
motor-command messages of this style, field types, units, an `id` per motor,
whether `mode` means something specific to the RH56 driver) is **unknown** and
not guessed here.

## Do not use this for anything but local Docker smoke-testing

- Field types/ranges may not match the real message.
- There is no guarantee `mode=0` means the same thing on real hardware.
- Replace this whole directory with the lab's real `custom_ros_messages` (or a
  `git submodule`/`vcs import` pointing at it) as soon as it's available, and
  delete this stub. See question 1 in `BAO_CAO_MENTOR.md`.
