# app.py - Root launcher for CineMatch AI
import sys
import os

# Ensure Src directory is in python module search path
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Switch working context if needed and run main
main_path = os.path.join(SRC_DIR, "main.py")
with open(main_path, "r", encoding="utf-8") as f:
    code = compile(f.read(), main_path, "exec")
    exec(code, globals())
