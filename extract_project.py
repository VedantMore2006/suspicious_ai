#!/usr/bin/env python3
"""
Extract all Python files with their content and project structure into a single text file.
"""

import os
import sys
from pathlib import Path


def get_project_structure(root_dir, prefix="", max_depth=10, current_depth=0, ignore_dirs=None):
    """Generate a text representation of the project structure."""
    if ignore_dirs is None:
        ignore_dirs = {".git", "__pycache__", ".pytest_cache", "*.egg-info", ".venv", "venv", "env"}
    
    if current_depth >= max_depth:
        return ""
    
    structure = ""
    try:
        items = sorted(os.listdir(root_dir))
    except PermissionError:
        return ""
    
    dirs = []
    files = []
    
    for item in items:
        # Skip hidden files/dirs and common ignore patterns
        if item.startswith("."):
            continue
        if item in ignore_dirs or any(item.endswith(x.replace("*", "")) for x in ignore_dirs if "*" in x):
            continue
        
        item_path = os.path.join(root_dir, item)
        if os.path.isdir(item_path):
            dirs.append(item)
        else:
            files.append(item)
    
    # Add files first
    for i, file in enumerate(files):
        is_last_file = (i == len(files) - 1) and len(dirs) == 0
        structure += f"{prefix}{'└── ' if is_last_file else '├── '}{file}\n"
    
    # Add directories
    for i, dir_name in enumerate(dirs):
        is_last = i == len(dirs) - 1
        structure += f"{prefix}{'└── ' if is_last else '├── '}{dir_name}/\n"
        
        dir_path = os.path.join(root_dir, dir_name)
        extension = "    " if is_last else "│   "
        structure += get_project_structure(dir_path, prefix + extension, max_depth, current_depth + 1, ignore_dirs)
    
    return structure


def extract_python_files(root_dir, output_file):
    """Extract all Python files and their content."""
    py_files = []
    
    # Collect all Python files
    for root, dirs, files in os.walk(root_dir):
        # Skip certain directories
        dirs[:] = [d for d in dirs if d not in {".git", "__pycache__", ".pytest_cache", ".venv", "venv", "env"}]
        
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                py_files.append(file_path)
    
    # Sort files for consistent output
    py_files.sort()
    
    # Write to output file
    with open(output_file, "w", encoding="utf-8") as out:
        # Header
        out.write("=" * 80 + "\n")
        out.write("PROJECT EXTRACTION REPORT\n")
        out.write("=" * 80 + "\n\n")
        
        # Project Structure
        out.write("PROJECT STRUCTURE:\n")
        out.write("-" * 80 + "\n\n")
        out.write(f"{os.path.basename(root_dir)}/\n")
        out.write(get_project_structure(root_dir))
        out.write("\n\n")
        
        # File Contents
        out.write("=" * 80 + "\n")
        out.write("FILE CONTENTS\n")
        out.write("=" * 80 + "\n\n")
        
        for file_path in py_files:
            # Calculate relative path
            rel_path = os.path.relpath(file_path, root_dir)
            
            # Write file header
            out.write(f"\n{'=' * 80}\n")
            out.write(f"FILE: {rel_path}\n")
            out.write(f"{'=' * 80}\n\n")
            
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    out.write(content)
            except Exception as e:
                out.write(f"[ERROR] Could not read file: {str(e)}\n")
            
            out.write("\n")
        
        # Summary
        out.write("\n" + "=" * 80 + "\n")
        out.write(f"SUMMARY: {len(py_files)} Python files extracted\n")
        out.write("=" * 80 + "\n")
    
    print(f"✓ Extraction complete!")
    print(f"✓ Total Python files: {len(py_files)}")
    print(f"✓ Output saved to: {output_file}")
    print(f"✓ File size: {os.path.getsize(output_file) / 1024:.2f} KB")


if __name__ == "__main__":
    # Get project root (current directory)
    project_root = os.getcwd()
    
    # Output file
    output_filename = "project_extraction.txt"
    output_path = os.path.join(project_root, output_filename)
    
    print(f"Extracting Python files from: {project_root}")
    print(f"Output file: {output_path}\n")
    
    extract_python_files(project_root, output_path)
