from setuptools import setup, find_packages
import os


lib_folder = os.path.dirname(os.path.realpath(__file__))
req_path = os.path.join(lib_folder, "requirements.txt")
install_requires = []
if os.path.exists(req_path):
    with open(req_path, "r") as f:
        install_requires = f.read().splitlines()

setup(
    name="yacgo",
    version="0.1",
    packages=find_packages(),
    install_requires=install_requires,
)
