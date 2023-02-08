import re
from jinja2 import Environment, FileSystemLoader, select_autoescape
import os.path
import pandas as pd
import glob

env = Environment(
    loader=FileSystemLoader(os.path.dirname(__file__)),
    autoescape=select_autoescape(),
    trim_blocks=True,
    lstrip_blocks=True,
)
template = env.get_template("duplicates.html.j2")

dup_path = "/home/guest/sischr001/LOKI-Pipeline/data/output/LOKI_PS122-2-18-73_01"

frame_id_pat = re.compile("(\d{8} \d{6}  \d{3})  ")


def get_frame_id(object_id):
    match = frame_id_pat.match(object_id)
    if match is None:
        raise ValueError(f"No match: {object_id}")

    return match[0]


def load_images(path):
    print("Loading images...")

    pat = os.path.join(path, "**/*.jpg")

    df = pd.DataFrame(
        {"img_fn": (os.path.relpath(x, path) for x in glob.iglob(pat, recursive=True))}
    )
    df["object_id"] = df["img_fn"].map(
        lambda x: os.path.splitext(os.path.basename(x))[0]
    )
    df["frame_id"] = df["object_id"].map(get_frame_id)

    dupset_id = df["img_fn"].map(lambda x: os.path.dirname(x))

    df["dupset_id"] = dupset_id.where(dupset_id.str.len() > 0, df["object_id"])

    return df


df = load_images(os.path.join(dup_path, "duplicates"))
print(df)


def gen_rows(df):
    columns = []
    for frame_id, group in df.groupby("frame_id"):

        row = {}

        for item in group.itertuples():
            try:
                idx = columns.index(item.dupset_id)
            except ValueError:
                try:
                    idx = columns.index(None)
                except ValueError:
                    columns.append(item.dupset_id)
                    idx = len(columns) - 1
                else:
                    columns[idx] = item.dupset_id

            row[idx] = dict(object_id=item.object_id, img_fn=item.img_fn, color="blue")

        # Convert row to list
        row = [row.get(i, None) for i in range(max(row.keys()) + 1)]

        yield row


print("Writing output...")
template.stream(project_id="LOKI_PS122-2-18-73_01", rows=gen_rows(df)).dump(
    os.path.join(dup_path, "index.html")
)
