from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'so101_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.json')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'skills'), glob('skills/*.pt')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='csrobot',
    maintainer_email='john.brann4315@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'simple_move = so101_control.joint_command_publisher:main',
            'object_pose_estimation = so101_control.object_pose_publisher:main',
            'rl_control = so101_control.rl_control:main',
            'move_home = so101_control.move_home:main',
        ],
    },
)
