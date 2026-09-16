from scripts.analyze_core32_results import failure_categories, validate_expected


def test_failure_categories_can_overlap_without_fabricating_manual_label():
    categories = failure_categories(
        {"vector": 1.0, "hybrid": 0.5, "rerank": 0.9}, 0.2
    )
    assert categories == ["hybrid_vs_vector_drop", "rule_not_recovered_or_worse", "rule_improves_hybrid"]


def test_validate_expected_compares_displayed_four_decimals():
    validate_expected(
        {"vector": 0.8734375, "hybrid": 0.796875, "rerank": 0.825},
        {"vector": 0.8734, "hybrid": 0.7969, "rerank": 0.8250},
    )
