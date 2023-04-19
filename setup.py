# ye  setup.py mere machine learning model ko 1 package me install krta h aur hm iss package ko pypi pe deploy kr skte h aur vha 
# koii bhii use kr skta h by downloading this setup.py

from setuptools import find_packages , setup
from typing import List # I have imported this because of 'def get_requirements(file_path:str)-> List[str]' this

HYPEN_E_DOT = '-e .'

def get_requirements(file_path:str)-> List[str]:

    """ this function return list of requirements"""
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        
        # I would not had to do this things but if this function go through \n will get added so remove \n I have written this code..
        requirements=[req.replace('\n','') for req in requirements]

        # I would not had to do this things but when this function go through requirements.txt there I have written HYPEN_E_DOT this to remove this I have written this code..
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
            
    return requirements

setup(
name = 'End-To-End_ML_project' , 
version='0.0.1',
author='Gyan Prakash Kushwaha',
author_email='gyanp7880@gmail.com',
packages=find_packages(),
install_requires = get_requirements('requirements.txt'),

)
