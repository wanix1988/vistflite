#!/usr/bin/env python3
#-*- coding: utf-8 -*-

import setuptools

with open('README.md', 'r', encoding='UTF-8') as df:
    long_description = df.read()
    
setuptools.setup(
    name = "TFLite Analyzer",
    version = "0.0.1",
    author = "wanglinwei",
    author_email = "wenix@live.cn",
    description = "Visualize the tflite model",
    long_description = "Visualize the tflite model",
    long_description_content_type = "markdown",
    url = "https://github.com/wanix1988/vistflite",
    packages = setuptools.find_packages(),
    install_requires = ["flask", "flask-bootstrap", "flatbuffers"],
    entry_points = {},
    classifiers = (
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT Licens",
        "Operating System :: OS Independent",
    )
)