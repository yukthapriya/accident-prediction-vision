from setuptools import setup, find_packages

setup(
    name="accident_prediction_vision",
    version="0.1.0",
    author="yukthapriya",
    author_email="yukthapriya@example.com",
    description="Real-time multi-sensor fusion for accident prediction",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yukthapriya/accident-prediction-vision",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        line.strip() for line in open("requirements.txt")
        if line.strip() and not line.startswith("#")
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)