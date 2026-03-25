"""
Unit tests for src/config.py (expanded)

Covers the Config class init, env overrides, _parse_env_value, and all get_* methods.
"""

import pytest
import os
from unittest.mock import patch
from src.config import Config, create_custom_config, get_config


class TestConfig:
    def test_default_init(self):
        config = Config()
        assert config.seed == 42
        assert config.model_choice is not None
        assert config.loss_function is not None

    def test_has_model_config(self):
        config = Config()
        assert isinstance(config.model_config, dict)
        assert 'hidden_channels' in config.model_config
        assert 'num_epochs' in config.model_config
        assert 'lr' in config.model_config

    def test_has_network_config(self):
        config = Config()
        assert isinstance(config.network_config, dict)
        assert 'use_ppi_network' in config.network_config

    def test_paths_are_set(self):
        config = Config()
        assert hasattr(config, 'paths')
        assert isinstance(config.paths, dict)
        assert 'molecule' in config.paths
        assert 'results' in config.paths

    def test_seed_default(self):
        config = Config()
        assert config.seed == 42

    def test_model_config_copy(self):
        config = Config()
        mc = config.get_model_config()
        mc['hidden_channels'] = 99999
        assert config.model_config['hidden_channels'] != 99999

    def test_explainer_config(self):
        config = Config()
        ec = config.get_explainer_config()
        assert isinstance(ec, dict)
        assert 'epochs' in ec

    def test_negative_sampling_config(self):
        config = Config()
        nsc = config.get_negative_sampling_config()
        assert isinstance(nsc, dict)
        assert 'strategy' in nsc
        assert 'pos_neg_ratio' in nsc

    def test_loss_config(self):
        config = Config()
        lc = config.get_loss_config()
        assert isinstance(lc, dict)
        assert 'loss_function' in lc
        assert 'params' in lc

    def test_all_paths(self):
        config = Config()
        paths = config.get_all_paths()
        assert isinstance(paths, dict)
        # Should be a copy
        paths['molecule'] = '/fake'
        assert config.paths['molecule'] != '/fake'

    def test_update_paths(self):
        config = Config()
        original = config.paths['molecule']
        config.update_paths(molecule='/new/path')
        assert config.paths['molecule'] == '/new/path'
        config.paths['molecule'] = original  # restore

    def test_has_training_version(self):
        config = Config()
        assert hasattr(config, 'training_version')

    def test_has_loss_params(self):
        config = Config()
        assert hasattr(config, 'loss_params')
        assert isinstance(config.loss_params, dict)

    def test_has_decoder_type(self):
        config = Config()
        assert 'decoder_type' in config.model_config

    def test_has_use_heuristics(self):
        config = Config()
        assert 'use_heuristics' in config.model_config


class TestParseEnvValue:
    def test_true(self):
        assert Config._parse_env_value("true") is True

    def test_false(self):
        assert Config._parse_env_value("false") is False

    def test_yes(self):
        assert Config._parse_env_value("yes") is True

    def test_no(self):
        assert Config._parse_env_value("no") is False

    def test_one(self):
        assert Config._parse_env_value("1") in (True, 1)

    def test_zero(self):
        assert Config._parse_env_value("0") in (False, 0)

    def test_int(self):
        assert Config._parse_env_value("42") == 42

    def test_float(self):
        assert Config._parse_env_value("3.14") == pytest.approx(3.14)

    def test_string(self):
        assert Config._parse_env_value("hello") == "hello"


class TestConfigEnvOverrides:
    def test_env_override_flat(self):
        with patch.dict(os.environ, {"GNN_OVERRIDE_loss_function": "focal"}):
            config = Config()
            assert config.loss_function == "focal"

    def test_env_override_nested(self):
        with patch.dict(os.environ, {"GNN_OVERRIDE_model_config__hidden_channels": "256"}):
            config = Config()
            assert config.model_config['hidden_channels'] == 256

    def test_env_override_bool(self):
        with patch.dict(os.environ, {"GNN_OVERRIDE_model_config__use_heuristics": "false"}):
            config = Config()
            assert config.model_config['use_heuristics'] is False


class TestCreateCustomConfig:
    def test_basic(self):
        config = create_custom_config(seed=99)
        assert config.seed == 99

    def test_model_config_override(self):
        config = create_custom_config(model_config={'hidden_channels': 512})
        assert config.model_config['hidden_channels'] == 512

    def test_paths_override(self):
        config = create_custom_config(paths={'molecule': '/custom'})
        assert config.paths['molecule'] == '/custom'

    def test_loss_params_override(self):
        config = create_custom_config(loss_params={'alpha': 0.9})
        assert config.loss_params['alpha'] == 0.9


class TestGetConfig:
    def test_returns_config(self):
        config = get_config()
        assert isinstance(config, Config)
