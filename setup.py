from setuptools import setup, find_packages

setup(
    name="bioscreen",
    version="1.0.0",
    description="Function-aware DNA synthesis screening against AI-designed biological threats",
    author="AIxBio Hackathon Team",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.2.0",
        "transformers>=4.36.0",
        "biopython>=1.83",
        "fastapi>=0.109.0",
        "uvicorn>=0.27.0",
        "numpy>=1.26.0",
        "scikit-learn>=1.4.0",
    ],
    entry_points={
        "console_scripts": [
            "bioscreen-server=bioscreen.api.server:app",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.12",
    ],
)
