"""Build the wealthy-study comparisons dataframe from the raw Place Pulse 2.0 dump.

Joins votes.tsv (pairwise comparisons) with locations.tsv (lat/lon) and the
images/ directory (filenames encode the location id as their 3rd underscore
field), then saves a pickle consumable by data.preprocess_csv. A pickle is
used instead of CSV because left_image_location/right_image_location are
(lat, lon) tuples that would need re-parsing if round-tripped through CSV.
"""
import argparse
import os
import os.path as osp

import pandas as pd

WEALTHY_STUDY_ID = "50f62cb7a84ea7c5fdd2e458"


def build_location_id_to_path(images_dir):
    mapping = {}
    for name in os.listdir(images_dir):
        location_id = name.split("_")[2]
        mapping[location_id] = osp.join(images_dir, name)
    return mapping


def main(args):
    locations = pd.read_csv(osp.join(args.pp_dir, "locations.tsv"), sep="\t")
    locations = locations.set_index("_id")

    votes = pd.read_csv(osp.join(args.pp_dir, "votes.tsv"), sep="\t")
    votes = votes[votes["study_id"] == WEALTHY_STUDY_ID]

    id_to_path = build_location_id_to_path(osp.join(args.pp_dir, "images"))

    known_ids = set(id_to_path) & set(locations.index)
    votes = votes[votes["left"].isin(known_ids) & votes["right"].isin(known_ids)]

    rows = {
        "left_image_path": votes["left"].map(id_to_path),
        "right_image_path": votes["right"].map(id_to_path),
        "left_image_location": votes["left"].map(
            lambda i: (locations.at[i, "loc.0"], locations.at[i, "loc.1"])
        ),
        "right_image_location": votes["right"].map(
            lambda i: (locations.at[i, "loc.0"], locations.at[i, "loc.1"])
        ),
        "vote_timestamp": votes["timestamp"],
        "choice": votes["choice"],
    }
    result = pd.DataFrame(rows)
    result.to_pickle(args.out_path)
    print(f"Wrote {len(result)} wealthy-study votes to {args.out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pp_dir", default="data/place-pulse-2.0")
    parser.add_argument("--out_path", default="data/wealthy_votes.pkl")
    main(parser.parse_args())
