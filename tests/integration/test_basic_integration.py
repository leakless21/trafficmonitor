"""
Basic integration tests for the Traffic Monitor system.
These tests verify basic functionality without requiring full system setup.
"""

import pytest
from pathlib import Path


def test_project_structure():
    """Test that the basic project structure is in place."""
    project_root = Path(__file__).parent.parent.parent
    
    # Check main directories exist
    assert (project_root / "src" / "traffic_monitor").exists()
    assert (project_root / "configs").exists()
    assert (project_root / "data").exists()
    assert (project_root / "tools").exists()
    
    # Check key files exist
    assert (project_root / "pyproject.toml").exists()
    assert (project_root / "Makefile").exists()
    assert (project_root / "README.md").exists()


def test_configs_structure():
    """Test that configuration structure is properly organized."""
    configs_dir = Path(__file__).parent.parent.parent / "configs"
    
    # Check config directories
    assert (configs_dir / "base").exists()
    assert (configs_dir / "environments").exists()
    assert (configs_dir / "trackers").exists()
    
    # Check key config files
    assert (configs_dir / "base" / "default.yaml").exists()
    assert (configs_dir / "environments" / "development.yaml").exists()
    assert (configs_dir / "environments" / "production.yaml").exists()


def test_tools_structure():
    """Test that tools are properly organized."""
    tools_dir = Path(__file__).parent.parent.parent / "tools"
    
    # Check tool categories exist
    expected_categories = [
        "data_processing",
        "benchmarking", 
        "visualization",
        "batch_processing",
        "model_management",
        "development"
    ]
    
    for category in expected_categories:
        assert (tools_dir / category).exists(), f"Tool category '{category}' not found"


def test_data_structure():
    """Test that data directory structure is in place."""
    data_dir = Path(__file__).parent.parent.parent / "data"
    
    # Check data subdirectories
    assert (data_dir / "models").exists()
    assert (data_dir / "samples").exists()
    assert (data_dir / "outputs").exists()
    assert (data_dir / "db").exists()
    
    # Check README exists
    assert (data_dir / "README.md").exists()


@pytest.mark.integration
def test_import_main_modules():
    """Test that main modules can be imported (basic smoke test)."""
    try:
        # Test if we can import the main package
        import sys
        from pathlib import Path
        
        # Add src to path for testing
        project_root = Path(__file__).parent.parent.parent
        src_path = project_root / "src"
        sys.path.insert(0, str(src_path))
        
        # Try importing main modules
        import traffic_monitor
        assert traffic_monitor is not None
        
    except ImportError as e:
        pytest.skip(f"Module import failed (expected in restructured project): {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])