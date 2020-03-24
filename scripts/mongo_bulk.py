from pymongo import MongoClient, ASCENDING, ReplaceOne, UpdateOne
from pymongo.errors import BulkWriteError, WriteError

import numpy as np
from tqdm import tqdm


def bulk_write_nodes(host, db_name, nodes_collection_name, data):
    client = MongoClient(host)
    db = client[db_name]
    nodes = db[nodes_collection_name]

    doc_list = []

    keys, values = [list(x) for x in zip(*[(k, v) for k, v in data.items()])]

    for row in tqdm(zip(*values)):

        node = {k: v for k, v in zip(keys, row)}
        doc_list.append(node)

    print(doc_list[0:10])

    try:
        nodes.insert_many(doc_list)
    except BulkWriteError as bwe:
        print(bwe.details)
        raise bwe


def bulk_write_edges(
    host, db_name, edges_collection_name, endpoint_names, edges, directed
):
    client = MongoClient(host)
    db = client[db_name]
    edge_collection = db[edges_collection_name]

    doc_list = []

    u_name, v_name = endpoint_names

    for u, v in tqdm(edges):
        if not directed:
            u, v = min(u, v), max(u, v)

        doc = {u_name: int(np.int64(u)), v_name: int(np.int64(v))}
        doc_list.append(doc)

    edge_collection.insert_many(doc_list)


def write_metadata(host, db_name, meta_collection_name, directed, offset, shape):
    pass
