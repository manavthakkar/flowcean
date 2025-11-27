import logging

import polars as pl
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier
from typing_extensions import override

from flowcean.core.learner import SupervisedLearner
from flowcean.core.model import Model
from flowcean.sklearn import SciKitModel
from flowcean.utils.random import get_seed

logger = logging.getLogger(__name__)


class RandomForestClassifierLearner(SupervisedLearner):
    """Wrapper class for sklearn's RandomForestClassifier."""

    classifier: RandomForestClassifier

    def __init__(
        self,
        n_estimators: int = 200,
        *,
        criterion: str = "gini",
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_features: str | float | None = "sqrt",
        max_leaf_nodes: int | None = None,
        min_impurity_decrease: float = 0.0,
        bootstrap: bool = True,
        oob_score: bool = False,
        n_jobs: int | None = None,
        random_state: int | None = None,
        verbose: int = 0,
        warm_start: bool = False,
        class_weight: dict | str | None = "balanced",
        ccp_alpha: float = 0.0,
        max_samples: float | None = None,
        monotonic_cst: NDArray | None = None,
    ) -> None:
        """Initialize the random forest classifier."""
        self.classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state or get_seed(),
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples,
            monotonic_cst=monotonic_cst,
        )

    @override
    def learn(
        self,
        inputs: pl.LazyFrame,
        outputs: pl.LazyFrame,
    ) -> Model:
        """Fit the classifier."""
        dfs = pl.collect_all([inputs, outputs])
        collected_inputs = dfs[0]
        collected_outputs = dfs[1].to_series()

        self.classifier.fit(collected_inputs, collected_outputs)
        logger.info("Using Random Forest Classifier")

        return SciKitModel(
            self.classifier,
            output_names=outputs.columns,
        )
