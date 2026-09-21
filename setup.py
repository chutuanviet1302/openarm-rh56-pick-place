from setuptools import find_packages, setup


setup(
    name="openarm_pick_place",
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/openarm_pick_place"]),
        ("share/openarm_pick_place", ["package.xml", "config.example.json", "calibration.example.json"]),
    ],
    install_requires=["setuptools", "numpy"],
    zip_safe=True,
    entry_points={
        "console_scripts": [
            "d435_perception = openarm_pick_place.ros2_nodes:perception_main",
            "motion_planning = openarm_pick_place.ros2_nodes:motion_main",
            "mujoco_bridge = openarm_pick_place.mujoco_bridge:main",
        ]
    },
)
