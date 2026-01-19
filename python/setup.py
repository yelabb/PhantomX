"""PhantomX: LaBraM-POYO Neural Foundation Model Setup"""

from setuptools import setup, find_packages

setup(
    name="phantomx",
    version="0.1.0",
    description="Population-geometry BCI decoding with electrode-dropout robustness",
    author="Youssef El abbassi",
    author_email="youssef@elabbassi.com",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "pynwb>=2.5.0",
        "einops>=0.7.0",
        "matplotlib>=3.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.7.0",
            "ruff>=0.0.280",
        ],
        "onnx": [
            "onnx>=1.14.0",
            "onnxruntime>=1.15.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
