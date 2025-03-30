from setuptools import find_packages, setup

package_name = 'fatigue_classifier'

setup(
    name=package_name,
    version='0.0.0',
    # packages=find_packages(exclude=['test']),
    packages = [package_name, f"{package_name}/scripts", f"{package_name}/python_utils/ros2_utils/comms"],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='gkouretas',
    maintainer_email='gkouretas@scu.edu',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            f'sim = {package_name}.scripts.real_time_fatigue_simulator:main'
        ],
    },
)
