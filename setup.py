from setuptools import setup, find_packages

setup(
    name="transpath",
    version="0.1.0",
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'transpath-train=transpath.train:main',  # запуск функции main() из train.py
        ],
    },
    install_requires=[
        "torch",
        "lightning",
        "pyyaml",
        "tensorboard",
        "tqdm",
        "einops",
        "numpy",
    ],
    python_requires=">=3.8",
)