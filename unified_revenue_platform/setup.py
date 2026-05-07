"""Setup configuration for the Unified Revenue Platform package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip()
        for line in fh.readlines()
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="unified-revenue-platform",
    version="0.1.0",
    author="Shreyas Karwa",
    author_email="shreyasrkarwa@gmail.com",
    description=(
        "A production-grade framework for building unified revenue "
        "intelligence systems in B2B enterprise environments"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/shreyasrkarwa/Analytics",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "spark": ["pyspark>=3.3.0", "delta-spark>=2.0.0"],
        "mlflow": ["mlflow>=2.0.0"],
        "salesforce": ["simple-salesforce>=1.12.0"],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "urp-demo=demo_pipeline:main",
        ],
    },
)
