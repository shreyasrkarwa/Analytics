# The Unified Revenue Platform: A System-Agnostic Reference Architecture for Integrated Revenue Decision Support in B2B Enterprises

**Author:** Shreyas Karwa

**Affiliation:** Independent Researcher, Sunnyvale, California, USA

**Corresponding author:** Shreyas Karwa, Sunnyvale, California, USA. Email: karwa.s@northeastern.edu. ORCID: 0009-0002-4103-6811

---

## Abstract

Enterprise B2B sales organizations run revenue decision-making across disconnected tools: CRM systems hold raw activity, financial-planning systems hold approved targets, and forecasting and quota work happen in proprietary spreadsheets. This fragmentation, not poor data, drives low forecast accuracy and slow executive review. We address it from a design-science perspective by proposing the Unified Revenue Platform, a system-agnostic four-layer reference architecture — ingestion, dimension engineering, intelligence, and presentation — that integrates CRM ingestion, fiscal and pipeline dimension engineering, hierarchical target allocation, behavioral commit reconciliation, and machine-learning forecasting into one coherent decision-support system. The architecture is organized by three design principles: single source of truth, layered transformation, and compensation-plan alignment. We specify each layer's responsibilities, interfaces, and latency expectations, show how previously separate revenue-intelligence capabilities compose as interchangeable modules, and introduce a structured weekly forecast-submission workflow that absorbs existing manual processes rather than adding a parallel one. We demonstrate it through an open-source reference implementation exercised on a calibrated 100-quarter simulation of a 200-contributor organization, evaluating the design against derived requirements and reporting pipeline scalability to 10,000 nodes. The platform shows how revenue decision support can be made coherent, reversible across CRM vendors, and aligned with the incentives that drive adoption.

**Keywords:** revenue decision support; reference architecture; design science; sales analytics; enterprise systems; forecast workflow

---

## 1. Introduction

Revenue forecasting, quota setting, and pipeline management are among the most consequential recurring decisions in a B2B enterprise, and they are quintessential decision-support problems: they are made under uncertainty, embedded in an organizational hierarchy, repeated every quarter, and tied directly to compensation. Yet the infrastructure that supports them is typically fragmented. A single enterprise commonly maintains between 15 and 50 distinct dashboards, spreadsheets, and reporting tools for sales performance management (Davenport & Harris, 2007), and forecast accuracy in manual processes rarely exceeds 50–60% at the start of a quarter (Mentzer & Moon, 2005). The binding constraint is usually not data quality but architecture: pipeline mechanics live in the CRM, approved targets live in a financial-planning tool, behavioral commit history lives in managers' heads and spreadsheets, and no single system composes them. Executives then spend review cycles reconciling sources rather than acting on them.

This paper takes a design-science stance (Hevner et al., 2004; Peffers et al., 2007) and asks an architectural question: *what reference architecture allows the full revenue decision-support stack — ingestion, transformation, intelligence, and presentation — to be integrated into one coherent, vendor-reversible system whose outputs align with the decisions and incentives of its users?* Our answer is the Unified Revenue Platform (URP), a system-agnostic four-layer architecture together with three design principles and an open-source reference implementation.

We deliberately separate the *architecture* contribution of this paper from the *algorithmic* contributions it integrates. Two of the platform's intelligence modules — a directed-acyclic-graph (DAG) hierarchical target-allocation method and a manager-level bias-reconciliation method — are formalized and evaluated in detail in the companion paper Karwa (2026a). Here they appear as interchangeable components of the intelligence layer, and we cite the companion work for their formal specification and component-level evaluation. The contribution of the present paper is the integrating design: the layer decomposition and interfaces, the design principles that govern them, the workflow that drives adoption, and a reproducible end-to-end implementation and demonstration.

### 1.1 Contributions

This paper makes four design-science contributions to the decision-support literature.

1. **A four-layer reference architecture** for enterprise revenue decision support — ingestion, dimension engineering, intelligence, and presentation — that is system-agnostic with respect to CRM, data-platform, and financial-planning tool choice, and that admits independent evolution of each layer (Section 4).
2. **Three governing design principles** — single source of truth, layered transformation, and compensation-plan alignment — derived from requirements for enterprise analytics adoption and made operational in the architecture (Sections 3–4).
3. **A composable intelligence layer** that integrates hierarchical target allocation, behavioral commit reconciliation, and machine-learning forecasting as interchangeable modules, together with a structured weekly forecast-submission workflow that replaces rather than supplements manual processes (Section 4).
4. **An open-source reference implementation and demonstration**, exercised on a calibrated simulation and evaluated against the derived design requirements, including an end-to-end scalability characterization to 10,000 organizational nodes (Section 5).

Section 2 reviews related work. Section 3 derives design requirements and principles. Section 4 specifies the architecture. Section 5 presents the reference implementation and its evaluation. Section 6 discusses implications and limitations, and Section 7 concludes.

---

## 2. Related work

### 2.1 Decision support systems and architecture

The decision-support systems (DSS) literature has long held that system adoption is driven less by raw analytical capability than by the alignment between system outputs and the user's decision context (Arnott & Pervan, 2005; Power et al., 2015). This motivates an architectural treatment: the value of revenue intelligence depends on whether ingestion, transformation, allocation, forecasting, and presentation are composed coherently and surfaced where decisions are made. Our design-science framing follows Hevner et al. (2004) and Peffers et al. (2007), and we present the artifact and its evaluation in the style recommended by Gregor and Hevner (2013), distinguishing the architectural artifact from the algorithmic kernels it hosts.

### 2.2 Enterprise data architectures

The platform's lower layers build on modern data-architecture patterns. The lakehouse (Armbrust et al., 2020, 2021) provides the substrate for ingestion and engineering: Delta Lake (Armbrust et al., 2020) supplies the ACID merge-upsert semantics required by a source-faithful Bronze layer, and the medallion progression (Bronze → Silver → Gold) maps onto our ingestion → engineering → intelligence layers. The data-mesh perspective (Dehghani, 2022) reinforces a complementary requirement: domain-aligned ownership of derived revenue metrics inside the engineering layer avoids the central-data-team bottleneck typical of monolithic warehouses. These works supply infrastructure patterns; they do not prescribe a revenue-specific decision-support architecture, which is our concern.

### 2.3 Revenue intelligence platforms

Commercial revenue-intelligence platforms (e.g., Clari, Gong, Aviso, BoostUp, InsightSquared) apply machine learning to CRM pipeline data to predict end-of-quarter revenue. Peer-reviewed evaluation of such approaches is comparatively scarce; Bohanec et al. (2017) provide one of the few, reporting that interpretable machine-learning models on CRM features can outperform managerial judgment, particularly when behavioral signals are included. Architecturally, commercial systems are typically deployed as a *thick analytical layer above the CRM*: forecast logic is embedded inside a vendor-controlled model that lacks authoritative access to board-approved targets in the planning system, forcing parallel reconciliation. The URP inverts this relationship — treating the CRM as a source-of-record-only ingestion layer and centralizing intelligence in an enterprise-owned layer — which is what makes vendor reversibility (Section 6) a meaningful property rather than a marketing claim.

### 2.4 Forecasting and allocation methods integrated by the platform

The intelligence layer hosts forecasting and allocation methods rather than inventing them. Hierarchical forecasting and reconciliation (Hyndman & Athanasopoulos, 2021; Wickramasuriya et al., 2019) address statistical coherence across aggregation levels; quota *allocation*, by contrast, must encode intentional business decisions such as managerial hedges, which the companion paper's DAG method addresses (Karwa, 2026a). Behavioral correction of judgmental forecasts (Goodwin et al., 2011; Lawrence et al., 2006) motivates the bias-reconciliation module, also formalized in Karwa (2026a). The machine-learning forecaster uses gradient boosting (Friedman, 2001) via scikit-learn (Pedregosa et al., 2011), and the DAG operations use NetworkX (Hagberg et al., 2008). In this paper these are described at the level of interfaces and responsibilities; their internal specification and component-level evaluation are deferred to the companion work.

---

## 3. Design requirements and principles

Following design-science practice (Hevner et al., 2004), we first make explicit the requirements an enterprise revenue decision-support architecture must satisfy, then state the principles that the architecture uses to meet them.

**Design requirements.** From the fragmentation problem and the DSS adoption literature we derive five requirements. (R1) *Source fidelity*: raw operational records must be captured faithfully and remain auditable. (R2) *Semantic enrichment*: organization-specific concepts — fiscal calendars, pipeline coverage, segmented win rates — must be engineered once and reused. (R3) *Composable intelligence*: allocation, behavioral reconciliation, and forecasting must be swappable modules over a shared semantic layer, not hard-wired logic. (R4) *Coherent, role-aware presentation*: every stakeholder must see a single consistent hierarchy and the metrics relevant to their decisions. (R5) *Vendor reversibility*: the enterprise must be able to change CRM, planning, or BI tools without losing decision continuity.

**Principle 1 — Single source of truth.** Each data element has exactly one authoritative system: the CRM owns raw events, the financial-planning system owns approved targets and headcount, and the data platform owns derived metrics and business logic. Violations — for example, computing metrics inside the visualization tool — produce the fragmentation that drives parallel spreadsheets. This principle operationalizes R1 and R3.

**Principle 2 — Layered transformation.** Data flows unidirectionally through layers, each adding semantic richness, so that each layer can evolve independently behind a stable interface. This principle operationalizes R2 and R5: because intelligence consumes an engineered semantic layer rather than CRM-specific schemas, the CRM beneath can be replaced.

**Principle 3 — Compensation-plan alignment.** The presentation layer is organized around the metric that most directly affects each user's compensation. In a sales organization the most reliable driver of organic adoption is visibility into the number that determines pay; designing for the individual contributor's compensation plan pulls every other role onto the platform. This principle operationalizes R4 and is the architecture's adoption forcing-function.

---

## 4. The Unified Revenue Platform architecture

The URP is a four-layer architecture (Figure 1, Table 1). Data flows from source systems through ingestion (L1), dimension engineering (L2), and intelligence (L3) to a read-only presentation layer (L4). The architecture is system-agnostic; our reference implementation uses Salesforce, Databricks/Delta Lake, and Anaplan, but any CRM, data platform, and planning tool can fill these roles.

**Table 1.** URP layer responsibilities, reference implementations, and target latencies.

| Layer | Responsibility | Reference implementation | Latency |
|:------|:---------------|:-------------------------|:--------|
| L1: Ingestion | Faithful extraction from source systems | CRM API (e.g., Salesforce SOQL), lakehouse (Delta Lake; Armbrust et al., 2020) | Near real-time |
| L2: Engineering | Dimension transformation, fiscal mapping | Distributed compute (e.g., PySpark) | Minutes |
| L3: Intelligence | Target allocation, bias reconciliation, ML forecast | NetworkX (Hagberg et al., 2008), scikit-learn (Pedregosa et al., 2011), MLflow | Minutes |
| L4: Presentation | Role-based visualization, forecast submission | BI tool (e.g., Tableau), custom web app | Real-time |

![URP system architecture](figures/png/fig1_system_architecture.png)

*Figure 1.* URP system architecture. Source systems (L1) feed dimension engineering (L2) and intelligence (L3); a read-only presentation layer (L4) surfaces results. The dashed boundary marks the logic layer where all computation occurs. *(For submission, also upload as a separate file: `figures/pdf/fig1_system_architecture.pdf`.)*

### 4.1 Layer 1: ingestion

The ingestion layer extracts core CRM objects (Opportunity, Account, User, OpportunityHistory) through the CRM API and persists them to a source-faithful Bronze table using merge-upsert semantics (e.g., Delta Lake `MERGE INTO`), with the CRM record identifier as the natural merge key. A key engineering decision is to extract the full organizational lineage *within* the source query — for example, using Salesforce SOQL relationship traversal (`Owner.Manager.Manager.Name`) to stamp every opportunity with its complete reporting chain at extraction time — eliminating expensive post-hoc joins downstream. The Bronze layer applies no transformations and no business logic; its sole responsibility is fidelity to the source (satisfying R1).

### 4.2 Layer 2: dimension engineering

The engineering (Silver) layer transforms raw CRM dimensions into analytically meaningful fields that downstream modules share (satisfying R2). *Fiscal-calendar mapping* converts calendar dates to fiscal quarters for any fiscal-year start month, handling the common enterprise case where the fiscal year does not align with the calendar year. *Pipeline coverage ratio* is defined as open pipeline divided by quota and classified into Healthy (≥ 3.0×), Moderate (1.5–3.0×), and At Risk (< 1.5×); the 3.0× threshold reflects qualified-opportunity win rates clustering around 25–35%. *Segmented win rates* are computed per Segment × Region over trailing quarters, because Enterprise, Mid-Market, and SMB segments exhibit structurally different conversion rates. Engineering these dimensions once, in a domain-owned layer, is the data-mesh principle (Dehghani, 2022) applied to revenue.

### 4.3 Layer 3: intelligence

The intelligence layer composes three interchangeable modules over the shared semantic layer (satisfying R3). We describe each functionally; the allocation and bias methods are formally specified and evaluated in Karwa (2026a).

*Target allocation.* A DAG-based cascader distributes a board-approved root target down the reporting hierarchy using capacity-weighted, level-specific hedge multipliers and operator-supplied locked-node overrides, producing coherent, auditable quotas in a single pass. The method and its coherence guarantees are developed in Karwa (2026a); within the platform it exposes a target-in, quota-tree-out interface and emits a cascade summary for audit.

*Behavioral commit reconciliation.* A bias-reconciliation module summarizes each manager's trailing actual-versus-commit history into a single coefficient and an archetype label (conservative, calibrated, or optimistic), which the forecasting module can consume as a feature and the presentation layer can surface for review. The estimator and its evaluation are given in Karwa (2026a); here it is one pluggable signal source.

*Machine-learning forecasting.* A gradient-boosting ensemble (Friedman, 2001; Pedregosa et al., 2011) predicts quarterly revenue from a small feature vector spanning pipeline mechanics, historical velocity, organizational capacity, and the behavioral coefficient (Table 2). Time-series cross-validation prevents future-data leakage, and models are tracked and versioned via MLflow. Because the forecaster consumes engineered features and the behavioral signal through stable interfaces, alternative learners can be substituted without changing other layers.

**Table 2.** Feature interface consumed by the forecasting module. The behavioral coefficient is one pluggable signal among mechanical, historical, and organizational inputs.

| Feature | Signal type | Description |
|:--------|:-----------|:------------|
| open_pipeline_acv | Mechanical | Total open pipeline ACV |
| pipeline_coverage_ratio | Mechanical | Pipeline / quota ratio |
| avg_deal_age_days | Historical | Mean age of open opportunities |
| pct_stage4_plus | Mechanical | Fraction of pipeline in late stages |
| avg_win_rate_trailing | Historical | Trailing 4-quarter win rate (Segment × Region) |
| manager_bias_index | Behavioral | Commit-bias coefficient (Karwa, 2026a) |
| headcount_quota_ratio | Organizational | Active reps / total quota capacity |

### 4.4 Layer 4: presentation and the weekly forecast workflow

The presentation layer exposes a single drill-down hierarchy (Company → Segment → Region → Manager → IC → Deal) visible to every user (satisfying R4), and replaces manual forecasting spreadsheets with a structured weekly submission workflow. Each submission is timestamped, attributed, versioned, and immediately visible to the full management chain, so the forecasting process itself — not just its outputs — lives on the platform. Per Principle 3, the workflow is built around the individual contributor's real-time pacing against compensation-relevant KPIs; designing for that view is intended to make participation organic rather than mandated. The workflow absorbs an existing process rather than adding a parallel one, which the DSS adoption literature identifies as decisive for sustained use (Arnott & Pervan, 2005; Power et al., 2015).

---

## 5. Reference implementation and evaluation

### 5.1 Reference implementation

We provide an open-source reference implementation of the architecture (Appendix A) whose package structure mirrors the four layers: `ingestion`, `dimension_engineering`, an intelligence tier (`quota_cascader`, `bias_reconciler`, `revenue_forecaster`), `weekly_forecast`, and a `data_generator` for testing and demonstration. Each module exposes the interfaces described in Section 4, and the modules are independently testable. The implementation requires only four core packages (pandas, numpy, networkx, scikit-learn), with optional connectors (pyspark, delta-spark, mlflow, simple-salesforce) for production deployment; this keeps the reference artifact runnable without proprietary infrastructure.

### 5.2 Evaluation approach

Because no real customer data is used, the evaluation is a design-science demonstration (Hevner et al., 2004; Peffers et al., 2007): we instantiate the architecture on a calibrated synthetic organization and assess whether it satisfies the design requirements of Section 3 and operates within the latency expectations of Table 1. Component-level accuracy and coherence claims for the allocation and bias modules are established separately in Karwa (2026a) and are not re-derived here; this section evaluates the *integrated artifact*.

**Simulated organization.** Using the bundled `data_generator`, we construct a five-level hierarchy (CRO → VP → Director → Manager → IC) with 200+ individual contributors across three regions and three segments, an annual root target of \$500M with quarterly seasonality, and 100 quarters of opportunity, commit, and outcome history. Generator parameters (branching factors, win rates, deal-age and seasonality distributions) are calibrated to values reported in the sales-forecasting literature (Bohanec et al., 2017; Lawrence et al., 2006; Mentzer & Moon, 2005) rather than tuned to flatter the platform.

### 5.3 Design-requirement evaluation

Table 3 summarizes how the instantiated architecture satisfies each requirement. The end-to-end run exercises all four layers: ingestion materializes a source-faithful table; engineering derives fiscal, coverage, and win-rate dimensions; the intelligence modules produce a coherent quota tree, behavioral coefficients, and a forecast; and the presentation workflow records versioned submissions against the unified hierarchy.

**Table 3.** Evaluation of the instantiated architecture against the design requirements (Section 3).

| Requirement | Architectural mechanism | Demonstrated outcome |
|:------------|:------------------------|:---------------------|
| R1 Source fidelity | Bronze merge-upsert, no business logic | Reproducible source-faithful table with audit key |
| R2 Semantic enrichment | Engineered fiscal/coverage/win-rate dimensions | Shared semantic layer consumed by all L3 modules |
| R3 Composable intelligence | Interface-based allocation/bias/forecast modules | Modules run independently and in composition |
| R4 Coherent presentation | Single drill-down hierarchy + submission workflow | One consistent hierarchy across roles; versioned commits |
| R5 Vendor reversibility | Intelligence consumes engineered layer, not CRM schema | CRM swap leaves L2–L4 interfaces unchanged |

### 5.4 End-to-end scalability

A reference architecture must remain interactive at enterprise scale. We executed the full pipeline on synthetic hierarchies from 100 to 10,000 nodes and measured single-thread wall-clock time (Table 4, Figure 2). The allocation pass is linear in the number of nodes and the bias reconciliation is linear in the number of managers, consistent with the complexity results in Karwa (2026a); forecasting time is dominated by feature aggregation and grows sub-linearly. At 10,000 nodes — comparable to a very large enterprise sales organization — end-to-end runtime per cascade event remains under three seconds, comfortably within an interactive planning workflow's latency budget and validating the latency expectations of Table 1.

**Table 4.** End-to-end pipeline runtime by organization size (single thread, mean of 10 runs).

| Organization size (nodes) | ICs | Allocation | Bias reconciliation | ML train | ML inference |
|:--|:--|:--|:--|:--|:--|
| 100 | ~63 | 0.012 s | 0.004 s | 0.32 s | 0.008 s |
| 500 | ~315 | 0.058 s | 0.019 s | 0.41 s | 0.011 s |
| 1,000 | ~630 | 0.114 s | 0.037 s | 0.49 s | 0.014 s |
| 5,000 | ~3,150 | 0.578 s | 0.183 s | 0.83 s | 0.025 s |
| 10,000 | ~6,300 | 1.156 s | 0.367 s | 1.17 s | 0.041 s |

![End-to-end pipeline runtime by organization size](figures/png/fig6_scalability.png)

*Figure 2.* End-to-end pipeline runtime as a function of organization size. Allocation and bias reconciliation scale linearly; ML training is dominated by feature aggregation. At 10,000 nodes, total runtime is under three seconds. *(For submission, also upload as a separate file: `figures/pdf/fig6_scalability.pdf`.)*

---

## 6. Discussion

### 6.1 Design implications

Three implications follow for enterprise decision-support design. First, **process centralization drives adoption more than data centralization**: by absorbing the weekly forecast-submission workflow rather than adding another dashboard, the platform aims to become where work happens rather than one more place to check. Second, **the intelligence layer should be composable**: treating allocation, behavioral reconciliation, and forecasting as interchangeable modules over a shared semantic layer lets each evolve — or be replaced by future methods — without disturbing ingestion, engineering, or presentation. Third, **compensation alignment is an adoption forcing-function** that likely generalizes to any enterprise setting where users have measurable performance incentives.

### 6.2 Vendor reversibility

The single-source-of-truth and layered-transformation principles together yield a property that commercial revenue-intelligence platforms generally lack. Because the intelligence layer consumes an engineered semantic layer rather than CRM-specific schemas, an enterprise can change CRM vendors without losing forecast or allocation continuity; the model state and business logic remain in the enterprise-owned layer. Commercial systems that embed forecast logic inside the CRM create migration dependencies that this inversion avoids.

### 6.3 Limitations

The evaluation is a simulation-based demonstration. The synthetic generator reproduces patterns reported in the literature, but real-world behavior will vary with organization size, sales motion, data maturity, and market conditions; field instantiation across organizations is the most important next step and is supported by the reference implementation's thin connector layer. The architecture has been exercised in simulation to 10,000 nodes; in production the CRM API extraction step may bottleneck at very high opportunity volumes depending on rate limits. Finally, this paper evaluates the *integration*; the accuracy and coherence of the hosted allocation and bias methods are established in the companion work (Karwa, 2026a) and are properties of those modules rather than of the architecture.

### 6.4 Threats to validity

We note threats following Wohlin et al. (2012). *Internal validity*: calibration choices embed assumptions that field data could refute; the generator is deterministic given a seed, so the demonstration is reproducible, but calibrated demonstration is not field proof. *External validity*: results derive from a single calibrated template (200-IC, five-level, three-region B2B SaaS); the architectural separation generalizes more readily than any specific runtime figure. *Construct validity*: design-requirement satisfaction (Table 3) is assessed qualitatively against stated requirements; alternative requirement framings could yield different judgments. *Conclusion validity*: scalability figures are single-machine wall-clock means and indicate trends rather than absolute production performance.

### 6.5 Future work

Four directions follow from the reference implementation: field instantiation across multiple organizations using the connector layer; extension of the presentation layer with role-specific decision aids; substitution experiments in the intelligence layer to compare hosted methods under a common interface; and integration of natural-language commit narratives as an additional engineered signal. Each is enabled by, rather than requiring changes to, the architecture presented here.

---

## 7. Conclusion

Fragmented tooling, not poor data, is a principal obstacle to effective revenue decision-making in B2B enterprises. We presented the Unified Revenue Platform, a system-agnostic four-layer reference architecture — ingestion, dimension engineering, intelligence, and presentation — governed by three design principles (single source of truth, layered transformation, compensation-plan alignment) and realized as an open-source reference implementation. The architecture integrates previously separate revenue-intelligence capabilities, including the hierarchical target-allocation and behavioral-reconciliation methods of the companion paper, as interchangeable modules over a shared semantic layer, and surfaces them through a workflow designed to align with the incentives that drive organic adoption. A calibrated demonstration shows the instantiated architecture satisfying its design requirements and remaining interactive to 10,000 nodes. By making revenue decision support coherent, composable, and reversible across vendors, the platform offers a reusable architectural foundation for enterprise revenue intelligence and a basis for future field studies.

---

## Acknowledgements

None.

## Declaration of Interest Statement

The author reports there are no competing interests to declare.

## Data Availability Statement

The open-source reference implementation and the synthetic data generator that produce all demonstration results in this article are openly available on GitHub at https://github.com/shreyasrkarwa/Analytics/tree/main/unified_revenue_platform and are permanently archived on Zenodo at https://doi.org/10.5281/zenodo.20517567 (Karwa, 2026b). No real customer or revenue data were used.

## Funding

This research received no specific grant from funding agencies in the public, commercial, or not-for-profit sectors.

## Disclosure of Generative AI Use

During the preparation of this manuscript the author used Claude (Anthropic), a large language model (Claude Opus 4 generation, 2026), to assist with drafting and editing the text, formatting the manuscript to the journal's requirements, and developing and documenting the accompanying open-source reference implementation and figures. The tool was used to improve clarity and to accelerate routine drafting and formatting. All data and quantitative results reported in this article derive from the author's openly available code, not from the language model. The author reviewed, verified, and edited all AI-assisted content and takes full responsibility for the content of the article.

---

## Appendix A. Reproducibility

The reference implementation is released openly (Karwa, 2026b; archived at https://doi.org/10.5281/zenodo.20517567). **Software environment:** Python 3.10 with pandas ≥ 1.5, numpy ≥ 1.23, networkx ≥ 2.8, scikit-learn ≥ 1.1; optional pyspark, delta-spark, mlflow, simple-salesforce. The published demonstration requires only the four core packages. **Reproduction:** after installing requirements, `demo_pipeline.py` runs the end-to-end demonstration and `experiments/scalability.py` reproduces the runtime characterization in Section 5.4; module test suites run via `pytest tests/`. **Data:** no real customer or revenue data is used; the bundled `data_generator` is the canonical source of all demonstration inputs, seeded (seed = 42) for reproducibility.

---

## References

Armbrust, M., Das, T., Sun, L., Yavuz, B., Zhu, S., Murthy, M., Torres, J., van Hovell, H., Ionescu, A., Łuszczak, A., Świtakowski, M., Szafrański, M., Li, X., Ueshin, T., Mokhtar, M., Boncz, P., Ghodsi, A., Paranjpye, S., Senster, P., … Zaharia, M. (2020). Delta Lake: High-performance ACID table storage over cloud object stores. *Proceedings of the VLDB Endowment, 13*(12), 3411–3424.

Armbrust, M., Ghodsi, A., Xin, R., & Zaharia, M. (2021). Lakehouse: A new generation of open platforms that unify data warehousing and advanced analytics. *Conference on Innovative Data Systems Research (CIDR)*.

Arnott, D., & Pervan, G. (2005). A critical analysis of decision support systems research. *Journal of Information Technology, 20*(2), 67–87.

Bohanec, M., Robnik-Šikonja, M., & Kljajić Borštnar, M. (2017). Decision-making framework with double-loop learning through interpretable black-box machine learning models. *Industrial Management & Data Systems, 117*(7), 1389–1406.

Davenport, T. H., & Harris, J. G. (2007). *Competing on analytics: The new science of winning.* Harvard Business School Press.

Dehghani, Z. (2022). *Data mesh: Delivering data-driven value at scale.* O'Reilly Media.

Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *Annals of Statistics, 29*(5), 1189–1232.

Goodwin, P., Önkal, D., & Lawrence, M. (2011). Improving the role of judgment in economic forecasting. In M. P. Clements & D. F. Hendry (Eds.), *The Oxford handbook of economic forecasting* (pp. 163–190). Oxford University Press.

Gregor, S., & Hevner, A. R. (2013). Positioning and presenting design science research for maximum impact. *MIS Quarterly, 37*(2), 337–355.

Hagberg, A. A., Schult, D. A., & Swart, P. J. (2008). Exploring network structure, dynamics, and function using NetworkX. *Proceedings of the 7th Python in Science Conference*, 11–15.

Hevner, A. R., March, S. T., Park, J., & Ram, S. (2004). Design science in information systems research. *MIS Quarterly, 28*(1), 75–105.

Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and practice* (3rd ed.). OTexts.

Karwa, S. (2026a). *Graph-theoretic approaches to hierarchical revenue target allocation in B2B enterprises: A methodological framework* [Working paper, under review, Journal of Revenue and Pricing Management]. SSRN. https://ssrn.com/abstract=6626318

Karwa, S. (2026b). *Unified Revenue Platform: Reference implementation v1.0.0* [Data set/software]. Zenodo. https://doi.org/10.5281/zenodo.20517567

Lawrence, M., Goodwin, P., O'Connor, M., & Önkal, D. (2006). Judgmental forecasting: A review of progress over the last 25 years. *International Journal of Forecasting, 22*(3), 493–518.

Mentzer, J. T., & Moon, M. A. (2005). *Sales forecasting management: A demand management approach* (2nd ed.). Sage Publications.

Peffers, K., Tuunanen, T., Rothenberger, M. A., & Chatterjee, S. (2007). A design science research methodology for information systems research. *Journal of Management Information Systems, 24*(3), 45–77.

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.

Power, D. J., Sharda, R., & Burstein, F. (2015). Decision support systems. In *Wiley encyclopedia of management.* Wiley.

Wickramasuriya, S. L., Athanasopoulos, G., & Hyndman, R. J. (2019). Optimal forecast reconciliation for hierarchical and grouped time series through trace minimization. *Journal of the American Statistical Association, 114*(526), 804–819.

Wohlin, C., Runeson, P., Höst, M., Ohlsson, M. C., Regnell, B., & Wesslén, A. (2012). *Experimentation in software engineering.* Springer.
