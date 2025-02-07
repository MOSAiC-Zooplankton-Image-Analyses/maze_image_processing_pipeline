"""Convert editable installs in the current environment and record the respective Git hashes into a requirements.txt."""

import importlib.metadata
import subprocess

remotes = {
    "morphocut": "https://github.com/morphocut/morphocut.git",
    "polytaxo": "https://github.com/moi90/polytaxo.git",
    "pyecotaxa": "https://github.com/moi90/pyecotaxa.git",
}

distributions = [
    dist
    for dist in importlib.metadata.Distribution.discover()
    if "site-packages" not in str(dist.locate_file("."))
]

with open("requirements.txt", "w") as f:
    for name, url in remotes.items():
        dist = next(
            (
                dist
                for dist in distributions
                if dist.metadata and dist.metadata["Name"] == name
            ),
        )
        dist_path = dist.locate_file(".")
        print(dist.metadata["Name"], dist_path, url)

        result = subprocess.run(
            ["git", "-C", str(dist_path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        git_hash = result.stdout.strip()
        print(f"{dist.metadata['Name']} @ {git_hash}")

        f.write(f"{name} @ git+{url}@{git_hash}\n")
