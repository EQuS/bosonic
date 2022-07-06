"""
Setup for bosonic-jax
"""
import os

from setuptools import setup, find_namespace_packages

REQUIREMENTS = [
    "numpy",
    "scipy",
    "matplotlib",
    "qutip",
    "tqdm",
    "cython>=0.29.20",
    "jax",
]

EXTRA_REQUIREMENTS = {
    "dev": [
        "jupyterlab>=3.1.0",
        "mypy",
        "pylint",
        "black",
        "mkdocs",
        "mkdocs-material",
        "mkdocs-gen-files",
        "mkdocs-literate-nav",
        "mkdocs-section-index",
        "mkdocstrings-python",
    ],
}

# Read long description from README.
README_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "README.md")
with open(README_PATH) as readme_file:
    README = readme_file.read()

version_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "bosonic_jax", "VERSION.txt")
)

with open(version_path, "r") as fd:
    version_str = fd.read().rstrip()

setup(
    name="bosonic-jax",
    version=version_str,
    description="Bosonic JAX",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/Phionx/bosonic-jax",
    author="Shantanu Jha, Shoumik Chowdhury, Max Hays",
    author_email="shantanu.rajesh.jha@gmail.com",
    license="Apache 2.0",
    packages=find_namespace_packages(exclude=["tutorials*"]),
    install_requires=REQUIREMENTS,
    extras_require=EXTRA_REQUIREMENTS,
    classifiers=[
        "Environment :: Console",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering",
    ],
    keywords="bosonic jax qubits cQED QEC GKP quantum error correction",
    python_requires=">=3.7",
    project_urls={
        "Bug Tracker": "https://github.com/Phionx/bosonic-jax/issues",
        "Documentation": "https://github.com/Phionx/bosonic-jax",
        "Source Code": "https://github.com/Phionx/bosonic-jax",
        "Tutorials": "https://github.com/Phionx/bosonic-jax/tutorials",
        "Tests": "https://github.com/Phionx/bosonic-jax/test",
    },
    include_package_data=True,
)
