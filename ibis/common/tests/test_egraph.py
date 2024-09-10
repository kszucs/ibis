from __future__ import annotations

import pytest

from ibis.common.egraph import DisjointSet


def test_disjoint_set():
    ds = DisjointSet()
    ds.add(1)
    ds.add(2)
    ds.add(3)
    ds.add(4)

    ds1 = DisjointSet([1, 2, 3, 4])
    assert ds == ds1
    assert ds[1] == {1}
    assert ds[2] == {2}
    assert ds[3] == {3}
    assert ds[4] == {4}

    assert ds.union(1, 2) is True
    assert ds[1] == {1, 2}
    assert ds[2] == {1, 2}
    assert ds.union(2, 3) is True
    assert ds[1] == {1, 2, 3}
    assert ds[2] == {1, 2, 3}
    assert ds[3] == {1, 2, 3}
    assert ds.union(1, 3) is False
    assert ds[4] == {4}
    assert ds != ds1
    assert 1 in ds
    assert 2 in ds
    assert 5 not in ds

    assert ds.find(1) == 1
    assert ds.find(2) == 1
    assert ds.find(3) == 1
    assert ds.find(4) == 4

    assert ds.connected(1, 2) is True
    assert ds.connected(1, 3) is True
    assert ds.connected(1, 4) is False

    # test mapping api get
    assert ds.get(1) == {1, 2, 3}
    assert ds.get(4) == {4}
    assert ds.get(5) is None
    assert ds.get(5, 5) == 5
    assert ds.get(5, default=5) == 5

    # test mapping api keys
    assert set(ds.keys()) == {1, 2, 3, 4}
    assert set(ds) == {1, 2, 3, 4}

    # test mapping api values
    assert tuple(ds.values()) == ({1, 2, 3}, {1, 2, 3}, {1, 2, 3}, {4})

    # test mapping api items
    assert tuple(ds.items()) == (
        (1, {1, 2, 3}),
        (2, {1, 2, 3}),
        (3, {1, 2, 3}),
        (4, {4}),
    )

    # check that the disjoint set doesn't get corrupted by adding an existing element
    ds.verify()
    ds.add(1)
    ds.verify()

    with pytest.raises(RuntimeError, match="DisjointSet is corrupted"):
        ds._parents[1] = 1
        ds._classes[1] = {1}
        ds.verify()

    # test copying the disjoint set
    ds2 = ds.copy()
    assert ds == ds2
    assert ds is not ds2
    ds2.add(5)
    assert ds != ds2
