from setuptools import setup, find_packages

setup(
    name="fairis",
    version="1.0.0",
    packages=find_packages(include=["fairis_tools", "fairis_tools.*", "fairis_lib", "fairis_lib.*","reinforcement_lib","reinforcement_lib.*"]),
    install_requires=[
        "numpy",
        "opencv-python",
        "scikit-learn",
        "scipy",
        "matplotlib",
        "astar",
        "pandas",
        "jupyter",
        "shapely",
        "pygame==2.1.3",
        "tensorflow==2.15.0",
        "tf-agents==0.19.0",
        "gym",

    ],
    author="Chance J Hamilton",
    description="FAIRIS: A framework for cognitive mapping and navigation using Webots simulator.",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/biorobaw/FAIRIS",  # Replace with your GitHub repo URL
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.11',
)
