"""
Unit tests for src/data/storage.py
"""

import pytest
import os
import json
import pickle
import tempfile
import pandas as pd
import pyarrow as pa

from src.data.storage import DataStorage


@pytest.fixture
def storage():
    return DataStorage()


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


# ── save/load processed data ────────────────────────────────────────
class TestSaveLoadData:
    def test_save_pandas_df(self, storage, temp_dir):
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        storage.save_processed_data({'mydata': df}, temp_dir)
        assert os.path.exists(os.path.join(temp_dir, 'mydata.parquet'))

    def test_save_pyarrow_table(self, storage, temp_dir):
        table = pa.table({'x': [10, 20], 'y': [30, 40]})
        storage.save_processed_data({'mytable': table}, temp_dir)
        assert os.path.exists(os.path.join(temp_dir, 'mytable.parquet'))

    def test_round_trip_pandas(self, storage, temp_dir):
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4.0, 5.0, 6.0]})
        storage.save_processed_data({'test': df}, temp_dir)
        loaded = storage.load_processed_data(temp_dir)
        assert 'test' in loaded
        pd.testing.assert_frame_equal(loaded['test'], df)

    def test_skip_unknown_type(self, storage, temp_dir, capsys):
        storage.save_processed_data({'bad': "not a dataframe"}, temp_dir)
        captured = capsys.readouterr()
        assert 'Warning' in captured.out


# ── save/load mappings ──────────────────────────────────────────────
class TestSaveLoadMappings:
    def test_save_dict_as_json(self, storage, temp_dir):
        mappings = {'drug_map': {'A': 0, 'B': 1}}
        storage.save_mappings(mappings, temp_dir)
        assert os.path.exists(os.path.join(temp_dir, 'drug_map.json'))

    def test_save_list_as_json(self, storage, temp_dir):
        mappings = {'my_list': ['a', 'b', 'c']}
        storage.save_mappings(mappings, temp_dir)
        assert os.path.exists(os.path.join(temp_dir, 'my_list.json'))

    def test_save_other_as_pickle(self, storage, temp_dir):
        mappings = {'my_set': frozenset([1, 2, 3])}
        storage.save_mappings(mappings, temp_dir)
        assert os.path.exists(os.path.join(temp_dir, 'my_set.pkl'))

    def test_round_trip_mappings(self, storage, temp_dir):
        mappings = {
            'dict_mapping': {'key1': 0, 'key2': 1},
            'list_mapping': ['a', 'b'],
        }
        storage.save_mappings(mappings, temp_dir)
        loaded = storage.load_mappings(temp_dir)
        assert loaded['dict_mapping'] == mappings['dict_mapping']
        assert loaded['list_mapping'] == mappings['list_mapping']

    def test_load_missing_dir_raises(self, storage):
        with pytest.raises(FileNotFoundError):
            storage.load_mappings('/nonexistent/path')
