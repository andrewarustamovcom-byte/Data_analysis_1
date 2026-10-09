"""Выполняет ноутбук текущим Python; ядро и рабочие файлы остаются в проекте."""

import json
from pathlib import Path
import sys
import tempfile

import nbformat
from nbclient import NotebookClient
from jupyter_client import AsyncKernelManager
from jupyter_client.kernelspec import KernelSpecManager


ROOT = Path(__file__).resolve().parents[1]
kernel_root = ROOT / ".jupyter" / "kernels"
kernel_dir = kernel_root / "marketplace-analysis"
kernel_dir.mkdir(parents=True, exist_ok=True)
spec = {"argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "Python (.venv)", "language": "python"}
(kernel_dir / "kernel.json").write_text(json.dumps(spec), encoding="utf-8")
manager = AsyncKernelManager(kernel_name="marketplace-analysis", kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_root)]))
manager.connection_file = str(Path(tempfile.gettempdir()) / "marketplace-analysis-kernel.json")
notebook = nbformat.read(ROOT / "analysis.ipynb", as_version=4)
client = NotebookClient(notebook, km=manager, timeout=600, resources={"metadata": {"path": str(ROOT)}})
client.execute()
nbformat.write(notebook, ROOT / "analysis.ipynb")
print("Ноутбук выполнен и сохранён:", ROOT / "analysis.ipynb")
