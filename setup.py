from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="gmda",
    version="0.1.0",
    author="Alice Lacan",
    author_email="alice.b.lacan@gmail.com",
    description="Generative Modeling Density Alignement",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ablacan/gmda",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        "matplotlib==3.5.1",
        "numpy==1.21.2",
        "numba==0.55.1",
        "pandas==1.4.1",
        "scikit_learn==1.0.2",
        "seaborn==0.12.2",
        "setuptools==58.0.4",
        "umap_learn==0.5.2",
        "rich==13.3.5",
    ],
    extras_require={
        'conda': ['torch==1.12.1', 'torchvision==0.13.1', 'torchaudio==0.12.1'],
    },
    entry_points={
        "console_scripts": [
            "gmda=main:main",
        ],
    },
)
