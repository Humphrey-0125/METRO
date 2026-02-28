#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from metrics import main


if __name__ == "__main__":
    if "--dataset" not in sys.argv:
        sys.argv.extend(["--dataset", "esc"])
    main()
