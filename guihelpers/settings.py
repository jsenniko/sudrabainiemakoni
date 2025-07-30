"""
Application Settings Management

This module provides cross-platform settings storage for sudrabainiemakoni
applications using platformdirs for directory detection and hierarchical
settings search. Settings are stored as typed objects rather than dictionaries,
providing better type safety and direct access.

Settings Search Priority:
1. ./settings/settings.json (relative to program)
2. ~/SudrabainieMakoni/settings.json (user home, visible)
3. Platform-specific config dir/SudrabainieMakoni/settings.json (OS standard)

Requirements:
    pip install platformdirs

Author: Generated for sudrabainiemakoni project
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from platformdirs import user_config_dir
from sudrabainiemakoni.cloudimage_camera import CameraCalibrationParams


@dataclass
class SettingsMetadata:
    """Metadata about the settings file"""
    settings_version: str = "1.0"
    created_by: str = "sudrabainiemakoni"
    last_modified: Optional[str] = None
    settings_file_location: Optional[str] = None


def get_settings_search_paths(app_name: str = "SudrabainieMakoni", 
                             settings_filename: str = "settings.json") -> List[Path]:
    """
    Get settings file paths in hierarchical search priority order.
    
    Args:
        app_name: Application name for directory naming
        settings_filename: Name of the settings file
        
    Returns:
        List of Path objects in priority order (first found wins)
    """
    return [
        # 1. Local to program (highest priority)
        Path("./settings") / settings_filename,
        
        # 2. User home directory (visible, easy to find)
        Path.home() / app_name / settings_filename,
        
        # 3. Platform-specific config directory (OS standard)
        Path(user_config_dir(app_name)) / settings_filename
    ]


class AppSettings:
    """
    Cross-platform application settings manager with object-oriented storage.
    
    Settings are stored as typed objects (e.g., CameraCalibrationParams) rather
    than dictionaries, providing better type safety and direct access.
    """
    
    def __init__(self, app_name: str = "SudrabainieMakoni", 
                 settings_filename: str = "settings.json"):
        """
        Initialize settings manager.
        
        Args:
            app_name: Name of the application
            settings_filename: Name of the settings file
        """
        self.app_name = app_name
        self.settings_filename = settings_filename
        
        # Initialize settings objects with defaults
        self.camera_calibration = CameraCalibrationParams()
        self.metadata = SettingsMetadata(created_by=app_name)
        
        # Last used directory for projects and pictures
        self.last_directory: Optional[str] = None
        
        # Get search paths
        self.search_paths = get_settings_search_paths(app_name, settings_filename)
        
        # Find existing settings file or determine where to save
        self.active_settings_file = self._find_settings_file()
        
        # Update metadata with file location
        self.metadata.settings_file_location = str(self.active_settings_file)
        
        # Load settings from file
        self.load_from_file()
    
    def _find_settings_file(self) -> Path:
        """
        Find existing settings file in search path order.
        
        Returns:
            Path to existing settings file, or preferred save location
        """
        # Look for existing settings file
        for path in self.search_paths:
            if path.exists():
                return path
        
        # No existing file found, use the user home location for saving
        # (skip local ./settings to avoid permission issues)
        return self.search_paths[2]  # Platform-specific config dir is preferred for saving
    
    def _ensure_settings_dir(self, settings_path: Path):
        """
        Create settings directory if it doesn't exist.
        
        Args:
            settings_path: Path to settings file
        """
        try:
            settings_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            print(f"Warning: Could not create settings directory {settings_path.parent}: {e}")
    
    def _to_dict(self) -> Dict[str, Any]:
        """
        Convert settings objects to dictionary for JSON serialization.
        
        Returns:
            Dictionary representation of all settings
        """
        return {
            "camera_calibration": self.camera_calibration.to_dict(),
            "metadata": asdict(self.metadata),
            "last_directory": self.last_directory
        }
    
    def _from_dict(self, data: Dict[str, Any]):
        """
        Load settings objects from dictionary (JSON deserialization).
        
        Args:
            data: Dictionary with settings data
        """
        # Load camera calibration parameters
        if "camera_calibration" in data:
            try:
                self.camera_calibration = CameraCalibrationParams.from_dict(data["camera_calibration"])
            except Exception as e:
                print(f"Warning: Could not load camera calibration settings: {e}")
                self.camera_calibration = CameraCalibrationParams()  # Use defaults
        
        # Load metadata
        if "metadata" in data:
            try:
                metadata_dict = data["metadata"]
                self.metadata = SettingsMetadata(
                    settings_version=metadata_dict.get("settings_version", "1.0"),
                    created_by=metadata_dict.get("created_by", self.app_name),
                    last_modified=metadata_dict.get("last_modified"),
                    settings_file_location=metadata_dict.get("settings_file_location", str(self.active_settings_file))
                )
            except Exception as e:
                print(f"Warning: Could not load metadata: {e}")
                # Keep the default metadata initialized in __init__
        
        # Load last directory
        if "last_directory" in data:
            self.last_directory = data["last_directory"]
    
    def load_from_file(self):
        """Load settings from the active settings file"""
        if not self.active_settings_file.exists():
            return  # Keep defaults
        
        try:
            with open(self.active_settings_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self._from_dict(data)
            
        except (json.JSONDecodeError, IOError, OSError) as e:
            print(f"Warning: Could not load settings from {self.active_settings_file}: {e}")
            print("Using default settings.")
    
    def save_to_file(self):
        """Save current settings to the active settings file"""
        try:
            # Ensure directory exists
            self._ensure_settings_dir(self.active_settings_file)
            
            # Update metadata
            import datetime
            self.metadata.last_modified = datetime.datetime.now().isoformat()
            self.metadata.settings_file_location = str(self.active_settings_file)
            
            # Convert to dictionary and write to file
            data = self._to_dict()
            with open(self.active_settings_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except (IOError, OSError) as e:
            print(f"Error: Could not save settings to {self.active_settings_file}: {e}")
    
    def get_settings_file_path(self) -> Path:
        """
        Get the path to the active settings file.
        
        Returns:
            Path object pointing to the active settings file
        """
        return self.active_settings_file
    
    def get_search_paths_info(self) -> List[Dict[str, Any]]:
        """
        Get information about all search paths.
        
        Returns:
            List of dictionaries with path info
        """
        info = []
        for i, path in enumerate(self.search_paths, 1):
            info.append({
                "priority": i,
                "path": str(path),
                "exists": path.exists(),
                "is_active": path == self.active_settings_file,
                "description": {
                    1: "Local to program (portable)",
                    2: "User home (visible, easy to find)",
                    3: "Platform-specific config (OS standard)"
                }[i]
            })
        return info
    
    def reset_to_defaults(self):
        """Reset all settings to default values"""
        self.camera_calibration = CameraCalibrationParams()
        self.metadata = SettingsMetadata(created_by=self.app_name)
        self.metadata.settings_file_location = str(self.active_settings_file)
        self.save_to_file()
    
    def get_all_settings_dict(self) -> Dict[str, Any]:
        """
        Get all settings as dictionary (for debugging/export).
        
        Returns:
            Complete settings dictionary
        """
        return self._to_dict()


# Global settings instance for convenience
app_settings = AppSettings()


# Convenience functions for direct object access
def get_camera_calibration() -> CameraCalibrationParams:
    """Get camera calibration parameters object"""
    return app_settings.camera_calibration


def save_camera_calibration():
    """Save current camera calibration parameters to file"""
    app_settings.save_to_file()


def get_settings_location() -> str:
    """Get human-readable description of settings file location"""
    return str(app_settings.get_settings_file_path())


def get_settings_search_info() -> List[Dict[str, Any]]:
    """Get information about settings search paths"""
    return app_settings.get_search_paths_info()


# Example usage and testing
if __name__ == "__main__":
    from sudrabainiemakoni.cloudimage_camera import DistortionOrder
    
    print("SudrabainieMakoni Settings Module")
    print("=================================")
    
    # Show settings search paths
    print("\nSettings Search Paths:")
    for info in get_settings_search_info():
        status = "✓ ACTIVE" if info["is_active"] else ("✓ exists" if info["exists"] else "✗ missing")
        print(f"{info['priority']}. {info['path']} - {info['description']} [{status}]")
    
    print(f"\nActive settings file: {get_settings_location()}")
    
    # Show current camera calibration object
    camera_params = get_camera_calibration()
    print(f"Current camera calibration object: {camera_params}")
    print(f"Distortion description: {camera_params.get_distortion_description()}")
    print(f"Projection description: {camera_params.get_projection_description()}")
    
    # Test modifying parameters directly
    print(f"\nModifying parameters directly...")
    camera_params.distortion = DistortionOrder.SECOND_ORDER
    camera_params.centers = False
    camera_params.separate_x_y = False
    camera_params.projectiontype = "equirectangular"
    
    print(f"Modified parameters: {camera_params}")
    
    # Save the changes
    save_camera_calibration()
    print("Parameters saved to file")
    
    # Verify they persist by creating a new settings instance
    test_settings = AppSettings()
    loaded_params = test_settings.camera_calibration
    print(f"Loaded from new instance: {loaded_params}")
    
    # Reset to defaults
    print("\nResetting to defaults...")
    app_settings.reset_to_defaults()
    
    final_params = get_camera_calibration()
    print(f"After reset: {final_params}")