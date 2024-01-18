from setuptools import setup

with open('README.md') as f:
    long_description = f.read()

setup(
    name='statistical-plot',
    version='0.8',
    license='MIT',
    author='Chen Liu',
    author_email='chen.liu.cl2482@yale.edu',
    packages={'statistical_plot'},
    # package_dir={'': ''},
    description='Statistical plotting with good aesthetics.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/ChenLiu-1996/StatsPlot',
    keywords='plotting, plot, statistical plotting, statistical plot',
    install_requires=['numpy', 'pandas', 'seaborn', 'matplotlib'],
    classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    ],
)