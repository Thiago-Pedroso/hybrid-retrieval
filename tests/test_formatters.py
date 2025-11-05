"""
Tests for output formatters.
"""

import pytest
import pandas as pd
from pathlib import Path
import tempfile
import json
from src.eval.formatters import (
    CSVFormatter,
    JSONFormatter,
    JSONLFormatter,
    get_formatter,
    FORMATTERS_REGISTRY,
)


class TestCSVFormatter:
    def test_format_dataframe(self):
        """Test formatting a DataFrame."""
        formatter = CSVFormatter()
        df = pd.DataFrame({"k": [1, 3, 5], "nDCG": [0.5, 0.7, 0.8]})
        
        result = formatter.format(df)
        assert "k,nDCG" in result
        assert "1,0.5" in result
    
    def test_format_dict(self):
        """Test formatting a dictionary."""
        formatter = CSVFormatter()
        data = {"k": 1, "nDCG": 0.5}
        
        result = formatter.format(data)
        assert "k" in result
        assert "nDCG" in result
    
    def test_save_to_file(self):
        """Test saving CSV to file."""
        formatter = CSVFormatter()
        df = pd.DataFrame({"k": [1, 3], "nDCG": [0.5, 0.7]})
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.csv"
            formatter.format(df, output_path=str(output_path))
            
            assert output_path.exists()
            loaded = pd.read_csv(output_path)
            assert len(loaded) == 2
            assert "k" in loaded.columns
            assert "nDCG" in loaded.columns
    
    def test_format_name(self):
        """Test format name property."""
        formatter = CSVFormatter()
        assert formatter.format_name == "csv"


class TestJSONFormatter:
    def test_format_dataframe(self):
        """Test formatting a DataFrame as JSON."""
        formatter = JSONFormatter()
        df = pd.DataFrame({"k": [1, 3], "nDCG": [0.5, 0.7]})
        
        result = formatter.format(df)
        parsed = json.loads(result)
        assert isinstance(parsed, list)
        assert len(parsed) == 2
        assert parsed[0]["k"] == 1
    
    def test_format_dict(self):
        """Test formatting a dictionary as JSON."""
        formatter = JSONFormatter()
        data = {"k": 1, "nDCG": 0.5}
        
        result = formatter.format(data)
        parsed = json.loads(result)
        assert parsed["k"] == 1
    
    def test_save_to_file(self):
        """Test saving JSON to file."""
        formatter = JSONFormatter()
        df = pd.DataFrame({"k": [1, 3], "nDCG": [0.5, 0.7]})
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.json"
            formatter.format(df, output_path=str(output_path))
            
            assert output_path.exists()
            with open(output_path) as f:
                loaded = json.load(f)
            assert len(loaded) == 2
    
    def test_format_name(self):
        """Test format name property."""
        formatter = JSONFormatter()
        assert formatter.format_name == "json"


class TestJSONLFormatter:
    def test_format_dataframe(self):
        """Test formatting a DataFrame as JSONL."""
        formatter = JSONLFormatter()
        df = pd.DataFrame({"k": [1, 3], "nDCG": [0.5, 0.7]})
        
        result = formatter.format(df)
        lines = result.strip().split("\n")
        assert len(lines) == 2
        parsed = json.loads(lines[0])
        assert parsed["k"] == 1
    
    def test_format_list(self):
        """Test formatting a list of dicts as JSONL."""
        formatter = JSONLFormatter()
        data = [{"k": 1, "nDCG": 0.5}, {"k": 3, "nDCG": 0.7}]
        
        result = formatter.format(data)
        lines = result.strip().split("\n")
        assert len(lines) == 2
    
    def test_format_single_dict(self):
        """Test formatting a single dict as JSONL."""
        formatter = JSONLFormatter()
        data = {"k": 1, "nDCG": 0.5}
        
        result = formatter.format(data)
        lines = result.strip().split("\n")
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["k"] == 1
    
    def test_save_to_file(self):
        """Test saving JSONL to file."""
        formatter = JSONLFormatter()
        df = pd.DataFrame({"k": [1, 3], "nDCG": [0.5, 0.7]})
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.jsonl"
            formatter.format(df, output_path=str(output_path))
            
            assert output_path.exists()
            with open(output_path) as f:
                lines = f.readlines()
            assert len(lines) == 2
    
    def test_format_name(self):
        """Test format name property."""
        formatter = JSONLFormatter()
        assert formatter.format_name == "jsonl"


class TestFormatterRegistry:
    def test_get_formatter_csv(self):
        """Test getting CSV formatter."""
        formatter = get_formatter("csv")
        assert isinstance(formatter, CSVFormatter)
    
    def test_get_formatter_json(self):
        """Test getting JSON formatter."""
        formatter = get_formatter("json")
        assert isinstance(formatter, JSONFormatter)
    
    def test_get_formatter_jsonl(self):
        """Test getting JSONL formatter."""
        formatter = get_formatter("jsonl")
        assert isinstance(formatter, JSONLFormatter)
    
    def test_get_formatter_case_insensitive(self):
        """Test getting formatter case-insensitively."""
        formatter1 = get_formatter("CSV")
        formatter2 = get_formatter("csv")
        assert isinstance(formatter1, CSVFormatter)
        assert isinstance(formatter2, CSVFormatter)
    
    def test_get_formatter_invalid(self):
        """Test error for invalid formatter name."""
        with pytest.raises(ValueError, match="Unknown format"):
            get_formatter("invalid_format")
    
    def test_all_formatters_in_registry(self):
        """Test that all formatters are in registry."""
        expected_formats = {"csv", "json", "jsonl"}
        assert set(FORMATTERS_REGISTRY.keys()) == expected_formats

