#!/usr/bin/env python3
"""Execute documented Python examples without network / 离线执行文档 Python 示例。"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    count = 0
    for relative in ("README.md", "README_zh.md", "docs/en/API.md", "docs/zh/API.md"):
        blocks = re.findall(r"```python\n(.*?)```", (ROOT / relative).read_text(), re.DOTALL)
        for block in blocks:
            result = subprocess.run([sys.executable, "-c", block], cwd=ROOT, check=False)
            if result.returncode:
                print(f"Failed example / 示例失败: {relative}")
                return result.returncode
            count += 1
    if count < 4:
        raise ValueError("expected at least four examples / 应至少存在四个示例")
    print(f"Documented examples passed / 文档示例通过: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
