from sksurv.metrics import concordance_index_censored
import numpy as np

def compute_c_index(event_indicator, event_time, estimate):
    """
    Computes the Concordance Index for evaluating survival models.
    """
    event_indicator = np.asarray(event_indicator).astype(bool)
    event_time = np.asarray(event_time)
    c_index, concordant, discordant, tied_risk, tied_time = concordance_index_censored(
        event_indicator, event_time, estimate
    )
    return {
        'C-Index': round(c_index, 4),
        'Concordant Pairs': concordant,
        'Discordant Pairs': discordant
    }
