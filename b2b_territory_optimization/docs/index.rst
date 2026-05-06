.. b2b-territory-optimization documentation master file

B2B Territory Optimization
==========================

**b2b-territory-optimization** is an open-source Python framework for Enterprise
RevOps and Data Strategy teams to mathematically design, carve, and manage B2B
sales territories using Operations Research algorithms.

Key Capabilities
----------------

- **Taxonomy Schema**: Define strict hierarchical boundaries preventing
  cross-contamination (e.g., an SMB account can never leak into an Enterprise territory).

- **Territory Allocator**: Balance accounts across K territories using a Greedy
  heuristic inspired by the Longest Processing Time (LPT) multiprocessor
  scheduling algorithm, consistently achieving < 0.1% TAM variance.

- **Capacity-Driven Prediction**: Automatically predict the required number of
  territories per segment using ``k = ceil(Total_TAM / Target_Capacity)``
  instead of manually guessing headcount.

- **Seller Assignment Matrix**: Map custom roles (AE, SE, SDR, Manager) to
  carved territories using configurable coverage ratios (1:1, 1:N, N:1).

- **Intelligent Bipartite Matching**: Optimally pair real human sellers to
  territories via the Hungarian Algorithm (``scipy.optimize.linear_sum_assignment``),
  enforcing hard taxonomy constraints and optimizing for seniority and domain
  expertise fit.

- **Manager Override Rebalancing**: Track manual account moves, quantify the
  resulting TAM imbalance, and suggest optimal 1-, 2-, or 3-account counter-swaps
  using ``itertools.combinations``.

Quick Start
-----------

.. code-block:: python

   from b2b_territory_optimization import (
       TaxonomySchema, TerritoryAllocator, B2BDataGenerator
   )

   # Generate synthetic B2B accounts
   df = B2BDataGenerator(random_seed=42).generate_accounts(1000)

   # Define strict taxonomy boundaries
   schema = TaxonomySchema(df, ['Region', 'Account_Segment'])

   # Allocate using capacity-driven prediction ($1.5M per territory)
   allocator = TerritoryAllocator(target_metric='Estimated_TAM')
   result = allocator.allocate_by_capacity(schema, target_capacity=1_500_000)

Installation
------------

.. code-block:: bash

   pip install b2b-territory-optimization

Integration
-----------

This package feeds directly into `b2b-revenue-forecasting <https://pypi.org/project/b2b-revenue-forecasting/>`_
for quota cascading after territories are defined.

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   modules
