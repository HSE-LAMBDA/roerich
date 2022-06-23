import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="roerich",  # Replace with your own username
    version="0.3.0",
    author=["Mikhail Hushchyn", "Kenenbek Arzymatov"],
    author_email=["hushchyn.mikhail@gmail.com", "kenenbek@gmail.com"],
    description="Roerich is a python library for online and offline change point detection in time series data based on machine learning.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/HSE-LAMBDA/roerich",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD 2-Clause License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.23.0',
        'pandas>=1.4.2',
        'scipy>=1.8.1',
        'scikit-learn>=1.1.1',
        'joblib>=1.1.0',
        'matplotlib>=3.5.2',
        'torch>=1.11.0',
        'torchaudio>=0.11.0',
        'torchvision>=0.12.0'
    ],
)
