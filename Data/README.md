# Data

Each dataset contains:
- 4 data files (`node.dat`, `node_seq.dat`, `link.dat`, `all_links.dat`)
- 1 evaluation file (`link.dat.test`)
- 1 negative mask file (`neg_mask.mtx`)

### node.dat and node_seq.dat
- In each line, there are 4 elements (`node_id`, `node_name`, `node_type`, `node_attributes`) separated by `\t`.
- `node_type` is `1` for proteins and `0` for RNA.
- In `node_attributes`, attributes are separated by comma (`,`). In `node_seq.dat`, RNAseq values are added to the features of the nodes.

### link.dat
- In each line, there are 4 elements (`node_id`, `node_id`, `link_type`, `link_weight`) separated by `\t`.
- All links are directed. Each node is connected by at least one link.

### link.dat.test
- In each line, there are 3 elements (`node_id`, `node_id`, `link_status`) separated by `\t`.
- For `link_status`, `1` indicates a positive link and `0 `indicates a negative link.
- Number of positive links = Number of negative links

### negative_mask.mtx
- `scipy` sparse matrix containing the highly-likely negative links.
