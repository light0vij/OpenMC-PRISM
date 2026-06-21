"""
environments/__init__.py
---------------------------------
Exposes all domain-specific components so the Notebook can import cleanly:

"""

from .env        import evaluate_for_ga, evaluate_for_bo, clean_openmc_files, get_model_params
from .mapping    import apply_symmetry_to_grid, ENR_LOW, ENR_HIGH
from .heuristics import check_corner_violations, count_adjacent_high_rods
from .bwr_vis    import plot_enr_grid, validate_and_plot_hifi
from .bwr_report import export_design_report

__all__ = [
    # Physics wrappers
    "evaluate_for_ga",
    "evaluate_for_bo",
    "clean_openmc_files",
    "get_model_params",
    
    # Geometry helpers
    "apply_symmetry_to_grid",
    "ENR_LOW",
    "ENR_HIGH",
    
    # Heuristics (pure counting)
    "check_corner_violations",
    "count_adjacent_high_rods",
    
    # Visualisation
    "plot_enr_grid",
    "validate_and_plot_hifi",
    
    # Reports
    "export_design_report",
]