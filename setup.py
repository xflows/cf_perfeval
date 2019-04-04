from setuptools import setup, find_packages
import os

CLASSIFIERS = [
    'Development Status :: 5 - Production/Stable',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
]

dist = setup(
    name='cf_perfeval',
    version='0.2',
    author='Borut Sluban',
    description='Package providing performance evaluation utilities and widgets for ClowdFlows 3.0',
    url='https://github.com/xflows/cf_perfeval',
    license='MIT License',
    classifiers=CLASSIFIERS,
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=[],
)
