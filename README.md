# GNNs-for-RPI

This is the repository for the manuscript [De novo prediction of RNA–protein interactions with graph neural networks][1] ([bioarxiv link][2]).

If you have any questions or feedback, please contact Viplove Arora varora@sissa.it.

## Packages
```
python: 3.6.8
pytorch: 1.8.1
torch-geometric: 2.0.3
scikit-learn: 0.24.2
```

## Instructions
For transductive link prediction, use `RPI_gcn.py` to make predictions on the partitioned datasets stored in the `Data` folder. Use `--help` command to check the optional arguments.

For transductive link prediction, use `gcn_drop.py` to make predictions on the protein selected using the `--test_prot` argument. Use `--help` command to check the other optional arguments.

Run `gcn_transfer.py` for the transfer learning setting.

[1]: https://rnajournal.cshlp.org/content/28/11/1469.short
[2]: https://www.biorxiv.org/content/10.1101/2021.09.28.462100v3.abstract
