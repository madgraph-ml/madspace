from setuptools import setup, find_packages

HTTPS_GITHUB_URL = "https://github.com/madgraph-ml/MadSpace"

with open("README.md", "r") as fh:
    long_description = fh.read()

requirements = ["numpy", "pandas", "scipy", "tables", "torch", "torchtestcase", "matplotlib", "vegas"]

setup(
    name="madspace",
    version="0.2.0",
    author="Ramon Winterhalder, Theo Heimel",
    author_email="ramon.winterhalder@uclouvain.be",
    description="Differentiable and GPU ready phase-space mappings",
    long_description=long_description,
    long_description_content_type="text/md",
    url=HTTPS_GITHUB_URL,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    packages=find_packages(),
    install_requires=requirements,
)
