import os
from setuptools import setup, find_packages
import sys


if sys.version_info.major != 3:
    print(
        "This Python is only compatible with Python 3, but you are running "
        "Python {}. The installation will likely fail.".format(sys.version_info.major)
    )

package_name = "sunblaze_envs"
authors = ["UC Berkeley", "Intel Labs", "and other contributors"]
url = "https://github.com/sunblaze-ucb/rl-generalization"
description = "Modifiable OpenAI Gym environments for studying generalization in RL"

setup_py_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = os.path.join(setup_py_dir, package_name)

# Discover asset files.
ASSET_EXTENSIONS = ["png", "wad", "txt"]
assets = []
for root, dirs, files in os.walk(package_dir):
    for filename in files:
        extension = os.path.splitext(filename)[1][1:]
        if extension and extension in ASSET_EXTENSIONS:
            filename = os.path.join(root, filename)
            assets.append(filename[1 + len(package_dir) :])

setup(
    name=package_name,
    version="0.1.0",
    description=description,
    author=", ".join(authors),
    # maintainer_email="",
    url=url,
    packages=find_packages(exclude=("examples",)),
    package_data={"": assets},
    dependency_links=(
        # "git+https://github.com/kostko/omgifol.git@master#egg=omgifol-0.1.0",
        "git+https://github.com/openai/gym.git@094e6b8e6a102644667d53d9dac6f2245bf80c6f#egg=gym-0.10.8r1",
        "git+https://github.com/openai/baselines.git@2b0283b9db18c768f8e9fa29fbedc5e48499acc6#egg=baselines-0.1.5r1",
    ),
    install_requires=[
        "gym==0.10.8r1",
        #'gym==0.10.5',
        "Box2D==2.3.2",
        "cocos2d==0.6.5",
        "numpy==1.14.2",
        "scipy==1.0.0",
        #'vizdoom==1.1.2',
        #'omgifol>=0.1.0',
    ],
    extras_require={
        "examples": [
            "baselines==0.1.5r1",  # use dep_link for specific commit
            "PyYAML==3.12",
            "opencv-python==3.4.0.12",
            "cloudpickle==0.4.1",
            "natsort==5.1.0",
            "chainer==3.3.0",
            "chainerrl==0.3.0",
        ],
    },
)
