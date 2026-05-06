import pytest
import pandas as pd
import numpy as np
from b2b_territory_optimization.intelligent_assignment import IntelligentAssigner
from b2b_territory_optimization.reassignment import ReassignmentEngine
from b2b_territory_optimization.allocator import TerritoryAllocator
from b2b_territory_optimization.taxonomy import TaxonomySchema


# ---------------------------------------------------------------------------
# IntelligentAssigner Tests
# ---------------------------------------------------------------------------

def _build_allocated_df():
    """Helper: creates a small allocated DataFrame for testing."""
    return pd.DataFrame({
        'Account_ID': ['A1', 'A2', 'A3', 'A4', 'A5', 'A6'],
        'Territory_ID': ['AMER_Ent_T1', 'AMER_Ent_T1', 'AMER_Ent_T1',
                         'EMEA_Ent_T2', 'EMEA_Ent_T2', 'EMEA_Ent_T2'],
        'Estimated_TAM': [500_000, 400_000, 300_000, 600_000, 200_000, 100_000],
        'Region': ['AMER'] * 3 + ['EMEA'] * 3,
        'Account_Segment': ['Enterprise'] * 6,
        'Industry': ['Technology', 'Technology', 'Healthcare',
                     'Finance', 'Finance', 'Retail']
    })


def test_intelligent_assigner_hard_constraints():
    """Sellers should only match territories in their own Region."""
    allocated_df = _build_allocated_df()

    sellers_df = pd.DataFrame({
        'Seller_ID': ['AE-1', 'AE-2'],
        'Region': ['AMER', 'EMEA'],
        'Account_Segment': ['Enterprise', 'Enterprise'],
        'Seniority': [1, 2],
        'Domain_Expertise': ['Technology', 'Finance']
    })

    assigner = IntelligentAssigner(sellers_df, allocated_df, ['Region', 'Account_Segment'])
    assignments = assigner.assign_sellers()

    assert len(assignments) == 2

    amer_match = assignments[assignments['Seller_ID'] == 'AE-1']
    emea_match = assignments[assignments['Seller_ID'] == 'AE-2']

    # AMER seller must match AMER territory
    assert amer_match.iloc[0]['Territory_ID'] == 'AMER_Ent_T1'
    assert bool(amer_match.iloc[0]['Is_Valid_Match']) is True

    # EMEA seller must match EMEA territory
    assert emea_match.iloc[0]['Territory_ID'] == 'EMEA_Ent_T2'
    assert bool(emea_match.iloc[0]['Is_Valid_Match']) is True


def test_intelligent_assigner_domain_expertise_preference():
    """When two sellers compete for the same region, domain match should win."""
    allocated_df = pd.DataFrame({
        'Account_ID': ['A1', 'A2', 'A3', 'A4'],
        'Territory_ID': ['T1', 'T1', 'T2', 'T2'],
        'Estimated_TAM': [500_000, 500_000, 500_000, 500_000],
        'Region': ['AMER'] * 4,
        'Account_Segment': ['Enterprise'] * 4,
        'Industry': ['Technology', 'Technology', 'Finance', 'Finance']
    })

    sellers_df = pd.DataFrame({
        'Seller_ID': ['AE-Tech', 'AE-Fin'],
        'Region': ['AMER', 'AMER'],
        'Account_Segment': ['Enterprise', 'Enterprise'],
        'Seniority': [2, 2],  # Same seniority to isolate domain effect
        'Domain_Expertise': ['Technology', 'Finance']
    })

    assigner = IntelligentAssigner(sellers_df, allocated_df, ['Region', 'Account_Segment'])
    assignments = assigner.assign_sellers()

    tech_match = assignments[assignments['Seller_ID'] == 'AE-Tech'].iloc[0]
    fin_match = assignments[assignments['Seller_ID'] == 'AE-Fin'].iloc[0]

    # Tech seller should get T1 (Technology-dominant), Finance seller should get T2
    assert tech_match['Territory_ID'] == 'T1'
    assert fin_match['Territory_ID'] == 'T2'


def test_intelligent_assigner_invalid_match_flagged():
    """When a seller has no valid territory, the assignment should be flagged invalid."""
    allocated_df = pd.DataFrame({
        'Account_ID': ['A1', 'A2'],
        'Territory_ID': ['AMER_T1', 'AMER_T2'],
        'Estimated_TAM': [500_000, 500_000],
        'Region': ['AMER', 'AMER'],
        'Account_Segment': ['Enterprise', 'Enterprise'],
        'Industry': ['Technology', 'Technology']
    })

    # Two sellers: one AMER (valid), one EMEA (no matching territory)
    sellers_df = pd.DataFrame({
        'Seller_ID': ['AE-1', 'AE-2'],
        'Region': ['AMER', 'EMEA'],
        'Account_Segment': ['Enterprise', 'Enterprise'],
        'Seniority': [1, 1],
        'Domain_Expertise': ['Technology', 'Technology']
    })

    assigner = IntelligentAssigner(sellers_df, allocated_df, ['Region', 'Account_Segment'])
    assignments = assigner.assign_sellers()

    # The EMEA seller is forced into an assignment but it should be flagged as invalid
    # because the cost is at the infinity threshold (1,000,000)
    emea_match = assignments[assignments['Seller_ID'] == 'AE-2']
    assert len(emea_match) == 1
    assert bool(emea_match.iloc[0]['Is_Valid_Match']) is False


# ---------------------------------------------------------------------------
# ReassignmentEngine Tests
# ---------------------------------------------------------------------------

def _build_balanced_df():
    """Helper: creates a perfectly balanced 2-territory DataFrame."""
    return pd.DataFrame({
        'Account_ID': ['A1', 'A2', 'A3', 'A4', 'A5', 'A6'],
        'Account_Name': ['Acme', 'Globex', 'Initech', 'Cyberdyne', 'Oscorp', 'Wayne'],
        'Territory_ID': ['T1', 'T1', 'T1', 'T2', 'T2', 'T2'],
        'Estimated_TAM': [300, 200, 100, 300, 200, 100]
    })


def test_reassignment_move_account():
    """Moving an account should update the territory and track history."""
    df = _build_balanced_df()
    engine = ReassignmentEngine(df, 'Estimated_TAM')

    engine.move_account('A1', 'T2')

    assert len(engine.history) == 1
    assert engine.history[0]['From'] == 'T1'
    assert engine.history[0]['To'] == 'T2'
    assert engine.history[0]['Value'] == 300

    # Verify T2 now has 4 accounts
    t2_count = (engine.df['Territory_ID'] == 'T2').sum()
    assert t2_count == 4


def test_reassignment_imbalance_detection():
    """After an unbalanced move, imbalance should be detected."""
    df = _build_balanced_df()
    engine = ReassignmentEngine(df, 'Estimated_TAM')

    # Both start at 600 total, move A1 (300) from T1 to T2
    engine.move_account('A1', 'T2')

    imbalance = engine.get_imbalance()

    t1_row = imbalance[imbalance['Territory_ID'] == 'T1']
    t2_row = imbalance[imbalance['Territory_ID'] == 'T2']

    # T1 should be below mean, T2 above
    assert t1_row.iloc[0]['Variance_From_Mean'] < 0
    assert t2_row.iloc[0]['Variance_From_Mean'] > 0


def test_reassignment_suggest_rebalance_returns_suggestions():
    """suggest_rebalance should return actionable suggestions."""
    df = _build_balanced_df()
    engine = ReassignmentEngine(df, 'Estimated_TAM')

    engine.move_account('A1', 'T2')

    suggestions = engine.suggest_rebalance(over_territory='T2', under_territory='T1')

    assert len(suggestions) > 0
    assert suggestions[0]['From'] == 'T2'
    assert suggestions[0]['To'] == 'T1'
    assert suggestions[0]['Delta_Improvement'] > 0


def test_reassignment_exclusions_work():
    """Excluded accounts and recently moved accounts should not appear in suggestions."""
    df = _build_balanced_df()
    engine = ReassignmentEngine(df, 'Estimated_TAM')

    engine.move_account('A1', 'T2')

    suggestions = engine.suggest_rebalance(
        over_territory='T2',
        under_territory='T1',
        excluded_account_ids=['A4', 'A5', 'A6']
    )

    # A1 is recently moved (auto-excluded), A4/A5/A6 are manually excluded
    # Only A2 and A3 remain — but they are in T1, not T2
    # So no valid candidates from T2 should remain
    # Actually A4, A5, A6 are in T2 but excluded; A1 was moved to T2 but is history-excluded
    all_suggested_ids = []
    for s in suggestions:
        for acc in s['Accounts']:
            all_suggested_ids.append(acc['Account_ID'])

    assert 'A1' not in all_suggested_ids  # Recently moved
    assert 'A4' not in all_suggested_ids  # Manually excluded
    assert 'A5' not in all_suggested_ids  # Manually excluded
    assert 'A6' not in all_suggested_ids  # Manually excluded


def test_reassignment_move_nonexistent_raises():
    """Moving an account that doesn't exist should raise ValueError."""
    df = _build_balanced_df()
    engine = ReassignmentEngine(df, 'Estimated_TAM')

    with pytest.raises(ValueError, match="not found"):
        engine.move_account('NONEXISTENT', 'T2')


def test_reassignment_three_account_combos():
    """Suggestions should include 3-account combinations when available."""
    # Create a scenario with enough accounts to generate 3-combos
    df = pd.DataFrame({
        'Account_ID': [f'A{i}' for i in range(1, 11)],
        'Account_Name': [f'Company_{i}' for i in range(1, 11)],
        'Territory_ID': ['T1'] * 5 + ['T2'] * 5,
        'Estimated_TAM': [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
    })
    engine = ReassignmentEngine(df, 'Estimated_TAM')

    # Move A1 (100) from T1 to T2, creating imbalance
    engine.move_account('A1', 'T2')

    suggestions = engine.suggest_rebalance(
        over_territory='T2', under_territory='T1', top_n=20
    )

    combo_types = [s['Type'] for s in suggestions]
    assert 'Combination (3 Accounts)' in combo_types


# ---------------------------------------------------------------------------
# Edge Case Tests
# ---------------------------------------------------------------------------

def test_allocator_handles_empty_dataframe():
    """Allocator should handle empty DataFrame gracefully."""
    allocator = TerritoryAllocator()
    result = allocator.allocate_bucket(pd.DataFrame(), 3, "Test")
    assert len(result) == 0


def test_allocator_handles_nan_values():
    """Allocator should handle NaN TAM values without crashing."""
    df = pd.DataFrame({
        'Account_ID': ['A1', 'A2', 'A3', 'A4'],
        'Estimated_TAM': [100, np.nan, 200, None]
    })
    allocator = TerritoryAllocator()
    result = allocator.allocate_bucket(df, 2, "Test")

    assert len(result) == 4
    assert result['Territory_ID'].notna().all()


def test_allocator_single_territory():
    """When num_territories=1, all accounts go to the same territory."""
    df = pd.DataFrame({
        'Account_ID': ['A1', 'A2', 'A3'],
        'Estimated_TAM': [100, 200, 300]
    })
    allocator = TerritoryAllocator()
    result = allocator.allocate_bucket(df, 1, "Single")

    assert result['Territory_ID'].nunique() == 1
