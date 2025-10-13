# imports
import os
import sys
import numpy as np
import numpy.typing as npt
import json
import csv
from typing import List


from signaturizer3d import CCSpace, Signaturizer

root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(root)


# current file directory
checkpoints_dir = os.path.join(root, "..", "..", "checkpoints")

# parse arguments
input_file = sys.argv[1]
output_file = sys.argv[2]

DATASETS = [i+j for i in "A" for j in "12345"]

def get_model_path(ds):
    return os.path.join(checkpoints_dir, "{}_split3.pt".format(ds))


def predict(smiles_list):

    # For each space, get N signatures (one per molecule)
    results = {}
    for ds in DATASETS:
        path = get_model_path(ds)
        signaturizer = Signaturizer(space=ds, local_weights_path= path)
        signatures = signaturizer.infer_from_smiles(smiles_list)
        results[ds]=signatures

    # For each space, store the N signatures (one per molecule)
    output = [[] for _ in range(len(smiles_list))]
    for ds in DATASETS:
        for r in range(len(smiles_list)):
            output[r].extend(results[ds][r])

    # to numpy array
    output = np.array(output)
    return output

with open(input_file, "r") as f:
    smiles = []
    reader = csv.reader(f)
    next(reader)
    for r in reader:
        smiles += [r[0]]

output = predict(smiles)

header = [f"{ds.lower()}_{r:03d}" for ds in DATASETS for r in range(128)]


# write output in a .csv file
with open(output_file, "w") as f:
    writer = csv.writer(f)
    writer.writerow(header)  # header
    for o in output:
        writer.writerow(o)