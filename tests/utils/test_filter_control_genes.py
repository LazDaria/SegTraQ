import numpy as np

from segtraq.utils import _filter_control_and_low_quality_transcripts


CONTROL_PREFIXES = (
    "NegControlProbe_",
    "antisense_",
    "NegControlCodeword",
    "BLANK_",
    "Blank-",
    "NegPrb",
    "DeprecatedCodeword_",
    "UnassignedCodeword_",
)


def _to_array(X):
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)


def test_filter_control_and_low_quality_transcripts(sdata_new):
    sdata = _filter_control_and_low_quality_transcripts(
        sdata_new,
        min_qv=30.0,
        control_prefixes=CONTROL_PREFIXES,
        points_key="transcripts",
        tables_key="table",
        points_gene_key="feature_name",
        inplace=False,
    )

    # Transcripts should have been removed.
    assert len(sdata.points["transcripts"]) < len(sdata_new.points["transcripts"])

    points = sdata.points["transcripts"].compute()

    # Minimum quality threshold should be respected.
    assert points["qv"].min() >= 30.0

    # No retained genes should match a control prefix.
    assert not points["feature_name"].str.startswith(CONTROL_PREFIXES).any()

    # No control genes should remain in the expression table.
    table_genes = sdata.tables["table"].var_names.astype(str)
    assert not table_genes.str.startswith(CONTROL_PREFIXES).any()

    # Point transformation should be preserved.
    assert sdata.points["transcripts"].attrs["transform"] == sdata_new.points["transcripts"].attrs["transform"]


def test_filter_control_prefixes(sdata_new):
    points_before = sdata_new.points["transcripts"].compute()

    # Use a prefix known to occur in both points and the table.
    table_genes = set(sdata_new.tables["table"].var_names.astype(str))
    common_genes = points_before.loc[
        points_before["feature_name"].astype(str).isin(table_genes),
        "feature_name",
    ].astype(str)

    gene = common_genes.iloc[0]
    prefix = gene

    sdata = _filter_control_and_low_quality_transcripts(
        sdata_new,
        min_qv=None,
        control_prefixes=(prefix,),
        points_key="transcripts",
        tables_key="table",
        points_gene_key="feature_name",
        inplace=False,
    )

    points_after = sdata.points["transcripts"].compute()

    assert not points_after["feature_name"].astype(str).str.startswith(prefix).any()
    assert len(points_after) < len(points_before)

    # Matching control genes should also be removed from the table.
    assert not sdata.tables["table"].var_names.astype(str).str.startswith(prefix).any()


def test_filter_control_and_low_quality_transcripts_no_filtering(sdata_new):
    sdata = _filter_control_and_low_quality_transcripts(
        sdata_new,
        min_qv=None,
        control_prefixes=(),
        points_key="transcripts",
        tables_key="table",
        points_gene_key="feature_name",
        inplace=False,
    )

    # No transcripts should have been removed.
    assert len(sdata.points["transcripts"]) == len(sdata_new.points["transcripts"])

    # Table should remain unchanged.
    assert sdata.tables["table"].shape == sdata_new.tables["table"].shape

    np.testing.assert_array_equal(
        _to_array(sdata.tables["table"].X),
        _to_array(sdata_new.tables["table"].X),
    )


def test_filter_control_and_low_quality_transcripts_no_transcripts_remain(
    sdata_new,
):
    sdata = _filter_control_and_low_quality_transcripts(
        sdata_new,
        min_qv=100.0,
        control_prefixes=(),
        points_key="transcripts",
        tables_key="table",
        points_gene_key="feature_name",
        inplace=False,
    )

    # No transcripts should remain.
    assert len(sdata.points["transcripts"].compute()) == 0

    # QV filtering does not modify the expression table.
    assert sdata.tables["table"].shape == sdata_new.tables["table"].shape

    np.testing.assert_array_equal(
        _to_array(sdata.tables["table"].X),
        _to_array(sdata_new.tables["table"].X),
    )


def test_filter_control_and_low_quality_transcripts_inplace_false(
    sdata_new,
):
    n_points_before = len(sdata_new.points["transcripts"])
    X_before = _to_array(sdata_new.tables["table"].X).copy()
    var_names_before = sdata_new.tables["table"].var_names.copy()

    _filter_control_and_low_quality_transcripts(
        sdata_new,
        min_qv=30.0,
        control_prefixes=CONTROL_PREFIXES,
        points_key="transcripts",
        tables_key="table",
        points_gene_key="feature_name",
        inplace=False,
    )

    # Original object should not be modified.
    assert len(sdata_new.points["transcripts"]) == n_points_before

    np.testing.assert_array_equal(
        _to_array(sdata_new.tables["table"].X),
        X_before,
    )

    assert sdata_new.tables["table"].var_names.equals(var_names_before)


def test_filter_control_and_low_quality_transcripts_min_qv_none(sdata_new):
    sdata = _filter_control_and_low_quality_transcripts(
        sdata_new,
        min_qv=None,
        control_prefixes=(),
        points_key="transcripts",
        tables_key="table",
        points_gene_key="feature_name",
        inplace=False,
    )

    # With both filters disabled, all points should remain.
    assert len(sdata.points["transcripts"]) == len(sdata_new.points["transcripts"])