from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    name='b2b-revenue-forecasting',
    version='0.2.0',
    description='A Python framework for hierarchical B2B sales quota cascading and pipeline reconciliation.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/shreyasrkarwa/Analytics/tree/main/hierarchical_sales_forecasting',
    author='Shreyas Karwa',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Financial and Insurance Industry',
        'Topic :: Office/Business :: Financial :: Spreadsheet',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    keywords='sales, forecasting, revops, quota, pipeline, b2b',
    packages=find_packages(include=['b2b_revenue_forecasting', 'b2b_revenue_forecasting.*']),
    python_requires='>=3.8, <4',
    install_requires=[
        'pandas>=1.0.0',
        'networkx>=2.5',
        'numpy>=1.19.0'
    ],
    project_urls={
        'Bug Reports': 'https://github.com/shreyasrkarwa/Analytics/issues',
        'Source': 'https://github.com/shreyasrkarwa/Analytics/tree/main/hierarchical_sales_forecasting',
    },
)
