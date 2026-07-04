"""
Regime Model Module
Implements regime detection models, primarily Hidden Markov Models (HMM).
"""

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from typing import Dict, Any, Tuple


class RegimeHMM:
    """
    Wrapper for Hidden Markov Model (HMM) from hmmlearn to identify market regimes.
    """

    def __init__(
        self,
        n_regimes: int = 3,
        covariance_type: str = "full",
        random_state: int = 42,
    ):
        """
        Initialize the Regime HMM.

        Parameters
        ----------
        n_regimes : int, default 3
            Number of hidden market regimes to identify (e.g., Bull, Bear, Crisis).
        covariance_type : str, default 'full'
            Covariance type for emissions ('full', 'diag', 'spherical', 'tied').
        random_state : int, default 42
            Seed for random state initialization.
        """
        self.n_regimes = n_regimes
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.model_ = None
        self.state_map_ = None  # Maps integer states to 'Bull', 'Bear', 'Crisis'

    def fit(self, features: pd.DataFrame) -> "RegimeHMM":
        """
        Fit the HMM to the features matrix.

        Parameters
        ----------
        features : pd.DataFrame
            DataFrame of features (e.g., returns, volatility) used to detect regimes.

        Returns
        -------
        self : RegimeHMM
            Fitted estimator.
        """
        self.model_ = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type=self.covariance_type,
            n_iter=1000,
            random_state=self.random_state,
        )
        self.model_.fit(features)
        
        # Decode states to label them
        predicted_states = self.model_.predict(features)
        self._create_state_mapping(features, predicted_states)
        
        return self

    def predict_regimes(self, features: pd.DataFrame) -> pd.Series:
        """
        Predict the most likely regime sequence for the given features.
        
        This method decodes the most likely state sequence using the Viterbi
        algorithm (which is the default decoding algorithm used by the `predict`
        method of GaussianHMM in hmmlearn under the hood).

        Parameters
        ----------
        features : pd.DataFrame
            DataFrame of features.

        Returns
        -------
        pd.Series
            Series of predicted regimes (0, 1, 2) indexed by date.
        """
        if self.model_ is None:
            raise ValueError("Model must be fitted before predicting.")
        states = self.model_.predict(features)
        return pd.Series(states, index=features.index, name="regime")

    def predict_labeled_regimes(self, features: pd.DataFrame) -> pd.Series:
        """
        Predict the regime sequence and map the integer labels to their string
        representations ('Bull', 'Bear', 'Crisis').

        Parameters
        ----------
        features : pd.DataFrame
            DataFrame of features.

        Returns
        -------
        pd.Series
            Series of labeled regimes ('Bull', 'Bear', 'Crisis') indexed by date.
        """
        states = self.predict_regimes(features)
        if self.state_map_ is None:
            return states.astype(str).rename("regime_label")
        return states.map(self.state_map_).rename("regime_label")

    def predict_probabilities(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Predict the posterior probabilities of each regime.

        Parameters
        ----------
        features : pd.DataFrame
            DataFrame of features.

        Returns
        -------
        pd.DataFrame
            DataFrame containing probability of each regime for each date.
        """
        if self.model_ is None:
            raise ValueError("Model must be fitted before predicting probabilities.")
        probs = self.model_.predict_proba(features)
        
        # Label the columns using our state map if available
        cols = [f"state_{i}" for i in range(self.n_regimes)]
        if self.state_map_ is not None:
            # Sort columns so they correspond to 0, 1, 2, ...
            cols = [self.state_map_[i] for i in range(self.n_regimes)]
            
        return pd.DataFrame(probs, index=features.index, columns=cols)

    def _create_state_mapping(self, features: pd.DataFrame, predicted_states: np.ndarray) -> None:
        """
        Map HMM state integers (0, 1, 2) to ('Bull', 'Bear', 'Crisis') based on state features:
        - Crisis: State with the highest volatility (nifty_volatility_21 or vix_level).
        - Bull: State with the lowest volatility and positive/highest momentum.
        - Bear: State with moderate volatility and negative/lower momentum.
        """
        df = features.copy()
        df["state"] = predicted_states
        
        state_means = df.groupby("state").mean()
        
        # Identify the volatility and momentum column names
        vol_col = next((c for c in features.columns if "volatility" in c or "vix" in c), None)
        mom_col = next((c for c in features.columns if "momentum" in c or "return" in c), None)
        
        if vol_col is None or mom_col is None:
            # Fallbacks
            vol_col = features.columns[0]
            mom_col = features.columns[0]
            
        print(f"\n--- HMM State Feature Means ---")
        print(state_means[[vol_col, mom_col]])
        
        # Labeling Logic:
        # 1. Crisis: Highest volatility
        crisis_state = state_means[vol_col].idxmax()
        
        # Remaining states
        remaining_states = [s for s in range(self.n_regimes) if s != crisis_state]
        
        if len(remaining_states) == 2:
            s1, s2 = remaining_states
            # 2. Bull: Highest momentum / positive return among the remaining states
            if state_means.loc[s1, mom_col] > state_means.loc[s2, mom_col]:
                bull_state = s1
                bear_state = s2
            else:
                bull_state = s2
                bear_state = s1
        elif len(remaining_states) == 1:
            bull_state = remaining_states[0]
            bear_state = -1
        else:
            bull_state = 0
            bear_state = 1
            
        self.state_map_ = {
            crisis_state: "Crisis",
            bull_state: "Bull",
            bear_state: "Bear"
        }
        
        # Enforce completeness for non-3-regime cases if they occur
        for s in range(self.n_regimes):
            if s not in self.state_map_:
                self.state_map_[s] = f"State_{s}"
                
        print("Discovered State Mapping:")
        for state_idx, label in self.state_map_.items():
            print(f"  State {state_idx} -> {label}")
        print("--------------------------------\n")

    def get_transition_matrix(self) -> pd.DataFrame:
        """
        Return the transition probability matrix as a DataFrame with labeled indices and columns.

        Returns
        -------
        pd.DataFrame
            Transition probability matrix.
        """
        if self.model_ is None:
            raise ValueError("Model must be fitted first.")
        trans_mat = self.model_.transmat_
        
        # Labeled names
        labels = [self.state_map_.get(i, f"State_{i}") for i in range(self.n_regimes)]
        return pd.DataFrame(trans_mat, index=labels, columns=labels)
