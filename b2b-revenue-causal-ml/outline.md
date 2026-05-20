# Outline: Why A/B Testing Fails in B2B Revenue Optimization — And How Causal ML Saves It

## 1. Introduction: The B2B Experimentation Trap
- **The Scenario**: A RevOps director wants to test if offering a 15% discount on enterprise contracts increases Year 1 Net Revenue Retention (NRR).
- **The Pitfall**: Running a standard A/B test.
  - *Small Sample Sizes*: Enterprise deals are low volume (e.g., dozens or hundreds per quarter, not millions like B2C).
  - *Long Sales Cycles*: A single deal takes 3–9 months to close.
  - *Statistical Power Problem*: Mathematically demonstrating that frequentist A/B testing is impractical (showing sample size calculations).
- **The Core Problem**: Selection Bias.
  - Sales reps don't hand out discounts randomly. They discount deals that are hard to close, highly competitive, or very large. 
  - Consequence: Observational CRM data shows that heavily discounted accounts have *lower* NRR or *higher* churn, leading to the false conclusion that discounts *cause* customer loss.

## 2. Decoupling Correlation from Causation: The RevOps Confounding DAG
- Introduce the concept of **Confounders** in B2B sales.
- **DAG (Directed Acyclic Graph) Visualization**:
  - `Account Size` & `Industry Vertical` are confounders.
  - They affect both:
    1. The likelihood of receiving a `Discount` (Treatment).
    2. The resulting `Net Revenue Retention (NRR)` (Outcome).
- Explain **Backdoor Paths**: How omitting these confounders in naive analysis leads to highly biased estimates of the discount's true impact.

## 3. The Causal Inference Toolkit
Explain the two chosen methods in an intuitive, accessible way:
- **Method 1: Propensity Score Matching (PSM)**
  - *Concept*: Predict the probability of a deal receiving a discount based on confounders (propensity score). Match each discounted deal with an undiscounted deal that has an almost identical propensity score.
  - *Intuition*: We are reconstructing a pseudo-randomized controlled trial (RCT) from purely historical data.
- **Method 2: Regression Adjustment (Doubly Robust Estimation Concept)**
  - *Concept*: Controlling for confounders directly in a multivariate regression model to block backdoor paths.

## 4. The Practical Blueprint (Python Code Walkthrough)
Outline the Python script we will write:
- **Synthetic Data Generation**: Creating a realistic B2B CRM dataset with 1,000 deals where:
  - Larger enterprise accounts in competitive industries (e.g., Tech) are more likely to get discounts.
  - Larger accounts naturally have higher NRR due to deeper integration, creating strong selection bias.
  - The *true* causal effect of a discount on NRR is set to a known positive value (e.g., +5.0% NRR).
- **Naive Analysis**: Running a simple OLS of NRR on Discount (which yields a biased, possibly negative coefficient due to selection bias).
- **Causal Estimation (Propensity Score Matching)**:
  - Step 1: Fit a Logistic Regression to estimate propensity scores.
  - Step 2: Perform 1-to-1 matching (nearest neighbor) on propensity scores.
  - Step 3: Run OLS on the matched cohort to recover the true causal effect.
- **Causal Estimation (Multivariate Regression)**:
  - Run OLS controlling for `Account Size` and `Industry`.
  - Compare results to show how controlling for confounders recovers the true +5.0% effect.

## 5. Conclusion & Actionable Advice for B2B Teams
- How data science teams can partner with Finance and Sales to implement causal frameworks.
- Best practices for CRM data cleanliness to support causal models (e.g., tracking "reason for discount" or "competitor presence").
