from setuptools import find_packages, setup

import versioneer

with open("README.md", "r") as fp:
    LONG_DESCRIPTION = fp.read()

setup(
    name="maze_image_processing_pipeline",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author="Simon-Martin Schroeder",
    author_email="sms@informatik.uni-kiel.de",
    description="(Re-)Segmentation and meta-data collection of LOKI images",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/MOSAiC-Zooplankton-Image-Analyses/loki_pipeline",
    packages=find_packages(),
    install_requires=[
        "parse",
        "exceptiongroup",
        "natsort",
        "pathlib_abc",
        "click",
        "rich",
        "pydantic",
    ],
    # python_requires=">=3.7",
    # extras_require={
    #     "tests": [
    #         # Pytest
    #         "pytest",
    #         "pytest-cov",
    #         "timer-cm",
    #         # Coverage
    #         "codecov",
    #         # Optional dependencies
    #         "parse",
    #     ],
    # },
    entry_points={"console_scripts": ["maze-ipp=maze_ipp.cli:cli"]},
    # classifiers=[
    #     "Programming Language :: Python :: 3",
    #     "Programming Language :: Python :: 3.7",
    #     "Programming Language :: Python :: 3.8",
    #     "Programming Language :: Python :: 3.9",
    #     "Programming Language :: Python :: 3.10",
    #     "License :: OSI Approved :: MIT License",
    #     "Operating System :: OS Independent",
    #     "Development Status :: 3 - Alpha",
    #     "Intended Audience :: Science/Research",
    # ],
)
