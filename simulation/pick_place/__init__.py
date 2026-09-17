"""Pick-and-place demo, split by responsibility:

config     -> every tunable
kinematics -> rotations, FK probes, IK
scene      -> model/data, index tables, geometry and contact queries
planner    -> wrist targets and the IK waypoint chain
executor   -> stepping physics through ctrl, collision aborts, viewer pacing
episode    -> TrialResult and the timestamped log
demo       -> the phases (perceive, plan, ready, reach, grasp, carry, release)
cli        -> command line
"""
