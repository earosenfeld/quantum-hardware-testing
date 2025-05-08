from setuptools import setup, find_packages

setup(
    name="cryocooler_test",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pytest>=7.0.0",
        "matplotlib>=3.5.0",
        "pandas>=1.4.0",
        "numpy>=1.21.0",
        "reportlab>=3.6.0",
        "python-dateutil>=2.8.2"
    ],
    python_requires=">=3.8",
    entry_points={
        'console_scripts': [
            'cryocooler-test=src.cryocooler.cryocooler:main',
        ],
    },
) 