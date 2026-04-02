from segtraq.utils import _filter_control_and_low_quality_transcripts


def test_filter_control_genes(sdata_new):
    sdata = _filter_control_and_low_quality_transcripts(
        sdata_new,
        min_qv=30.0,
        control_genes=("A2ML1", "AAMP", "AAR2", "AARSD1", "ABAT", "ABCA1", "ABCA10", "ABCA3"),
        points_key="transcripts",
        tables_key="table",
        points_gene_key="feature_name",
        recompute_expression=False,
        inplace=False,
    )

    # check that the control genes have been removed in transcripts
    assert len(sdata.points["transcripts"]) < len(sdata_new.points["transcripts"]), (
        f"Expected fewer transcripts after filtering, "
        f"but got {len(sdata.points['transcripts'])} vs {len(sdata_new.points['transcripts'])}"
    )
    # check that the control genes have been removed in tables
    assert sdata.tables["table"].shape[1] < sdata_new.tables["table"].shape[1], (
        f"Expected fewer genes after filtering, "
        f"but got {sdata.tables['table'].shape[1]} vs {sdata_new.tables['table'].shape[1]}"
    )
    # assert that the minimum quality value is respected
    assert sdata.points["transcripts"]["qv"].compute().min() >= 30.0, (
        f"Expected all quality values to be >= 30.0, but got min {sdata.points['transcripts']['qv'].min()}"
    )


def test_filter_control_genes_recompute_expression_matrix(sdata_new):
    sdata = _filter_control_and_low_quality_transcripts(
        sdata_new,
        min_qv=30.0,
        control_genes=("A2ML1", "AAMP", "AAR2", "AARSD1", "ABAT", "ABCA1", "ABCA10", "ABCA3"),
        points_key="transcripts",
        tables_key="table",
        points_gene_key="feature_name",
        inplace=False,
    )

    # check that the control genes have been removed in transcripts
    assert len(sdata.points["transcripts"]) < len(sdata_new.points["transcripts"]), (
        f"Expected fewer transcripts after filtering, "
        f"but got {len(sdata.points['transcripts'])} vs {len(sdata_new.points['transcripts'])}"
    )
    # check that the control genes have been removed in tables
    assert sdata.tables["table"].shape[1] < sdata_new.tables["table"].shape[1], (
        f"Expected fewer genes after filtering, "
        f"but got {sdata.tables['table'].shape[1]} vs {sdata_new.tables['table'].shape[1]}"
    )
    # assert that the minimum quality value is respected
    assert sdata.points["transcripts"]["qv"].compute().min() >= 30.0, (
        f"Expected all quality values to be >= 30.0, but got min {sdata.points['transcripts']['qv'].min()}"
    )


def test_filter_control_genes_no_filtering(sdata_new):
    sdata = _filter_control_and_low_quality_transcripts(
        sdata_new,
        min_qv=0.0,
        control_prefixes=(),
        points_key="transcripts",
        tables_key="table",
        points_gene_key="feature_name",
        recompute_expression=True,
        inplace=False,
    )

    # check that no transcripts have been removed
    assert len(sdata.points["transcripts"]) == len(sdata_new.points["transcripts"]), (
        f"Expected same number of transcripts after filtering, "
        f"but got {len(sdata.points['transcripts'])} vs {len(sdata_new.points['transcripts'])}"
    )
    # check that no genes have been removed in tables
    assert sdata.tables["table"].shape[1] == sdata_new.tables["table"].shape[1], (
        f"Expected same number of genes after filtering, "
        f"but got {sdata.tables['table'].shape[1]} vs {sdata_new.tables['table'].shape[1]}"
    )
    # assert that the expression matrix is unchanged
    assert (sdata.tables["table"].X != sdata_new.tables["table"].X).nnz == 0, (
        f"Expected expression matrix to be unchanged, but got differences in {sdata.tables['table'].X.nnz} entries"
    )


def test_filter_control_genes_no_transcripts_remain(sdata_new):
    sdata = _filter_control_and_low_quality_transcripts(
        sdata_new,
        min_qv=100.0,  # setting a high threshold to filter out all transcripts
        points_key="transcripts",
        tables_key="table",
        points_gene_key="feature_name",
        recompute_expression=False,
        inplace=False,
    )

    # check that no transcripts remain
    assert len(sdata.points["transcripts"].compute()) == 0, (
        f"Expected no transcripts after filtering, but got {len(sdata.points['transcripts'])}"
    )
    # check that the tables remain unchanged
    assert sdata.tables["table"].shape[1] == sdata_new.tables["table"].shape[1], (
        f"Expected the same number of genes after filtering, "
        f"but got {sdata.tables['table'].shape[1]} vs {sdata_new.tables['table'].shape[1]}"
    )


def test_filter_control_genes_no_transcripts_remain_with_recompute(sdata_new):
    sdata = _filter_control_and_low_quality_transcripts(
        sdata_new,
        min_qv=100.0,  # setting a high threshold to filter out all transcripts
        points_key="transcripts",
        tables_key="table",
        points_gene_key="feature_name",
        inplace=False,
    )

    # check that no transcripts remain
    assert len(sdata.points["transcripts"].compute()) == 0, (
        f"Expected no transcripts after filtering, but got {len(sdata.points['transcripts'])}"
    )
    # check that the tables remain unchanged
    assert sdata.tables["table"].X.sum() == 0, (
        f"Expected no counts after filtering with recompute_expression, but got {sdata.tables['table'].shape[1]}"
    )
