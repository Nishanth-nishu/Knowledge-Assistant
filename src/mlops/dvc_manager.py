"""
DVC Data Versioning Manager.

Wrapper around DVC for programmatic data versioning.
"""

import logging
import subprocess
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class DVCManager:
    """Wrapper for DVC commands."""
    
    def __init__(self, repo_path: str = "."):
        """
        Initialize DVC manager.
        
        Args:
            repo_path: Path to git/dvc repo root
        """
        self.repo_path = Path(repo_path).resolve()
    
    def _run_command(self, cmd: list[str]) -> bool:
        """Run a DVC command."""
        try:
            logger.info(f"Running DVC command: {' '.join(cmd)}")
            result = subprocess.run(
                ["dvc"] + cmd,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True,
            )
            logger.debug(result.stdout)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"DVC command failed: {e.stderr}")
            return False
        except FileNotFoundError:
            logger.error("DVC executable not found. Is DVC installed?")
            return False
    
    def init(self) -> bool:
        """Initialize DVC in the repo."""
        if (self.repo_path / ".dvc").exists():
            logger.info("DVC already initialized.")
            return True
        return self._run_command(["init"])
    
    def add(self, path: str) -> bool:
        """
        Add a file or directory to DVC.
        
        Args:
            path: Path relative to repo root
        """
        return self._run_command(["add", path])
    
    def push(self, target: str | None = None) -> bool:
        """
        Push data to remote storage.
        
        Args:
            target: Optional specific target to push
        """
        cmd = ["push"]
        if target:
            cmd.append(target)
        return self._run_command(cmd)
    
    def pull(self, target: str | None = None) -> bool:
        """
        Pull data from remote storage.
        
        Args:
            target: Optional specific target to pull
        """
        cmd = ["pull"]
        if target:
            cmd.append(target)
        return self._run_command(cmd)
    
    def status(self) -> bool:
        """Show DVC status."""
        return self._run_command(["status"])


__all__ = ["DVCManager"]
