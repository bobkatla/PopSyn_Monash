"""Create a tree structure to store the paired pools"""

from PopSynthesis.Methods.IPSF.const import (
    HH_TAG,
    HH_ATTS,
    PP_ATTS,
    NOT_INCLUDED_IN_BN_LEARN,
)
import pandas as pd

pp_atts = [x for x in PP_ATTS if x not in NOT_INCLUDED_IN_BN_LEARN]
hh_atts = [x for x in HH_ATTS if x not in NOT_INCLUDED_IN_BN_LEARN]


class PoolsTreeNode:
    def __init__(self, name: str, pool: pd.DataFrame, parent=None):
        self.parent = parent
        self.name = name
        self.children = {}
        self.pool = pool

    def find_node(self, name: str, checked_nodes):
        checked_nodes.append(self.name)
        if self.name == name:
            return self
        for child in [x for x in self.children.values() if x not in checked_nodes]:
            result = child.find_node(name, checked_nodes)
            if result is not None:
                return result
        if self.parent is not None and self.parent.name not in checked_nodes:
            return self.parent.find_node(name, checked_nodes)
        else:
            return None

    def add_node(self, paired_pool_name: str, pool: pd.DataFrame):
        root_rela, sample_rela = paired_pool_name.split("-")
        parent_node = self.find_node(root_rela, [])
        if parent_node is None:
            raise ValueError(f"Node {root_rela} not found to add {sample_rela}")
        if sample_rela in parent_node.children:
            print(f"WARNING: {sample_rela} already exists in {root_rela}")
        else:
            parent_node.children[sample_rela] = PoolsTreeNode(
                name=sample_rela, pool=pool, parent=parent_node
            )
        print(f"Added {sample_rela} to {root_rela}")

    def update_pool(self):
        # Traverse to root
        root_node = self
        while root_node.parent is not None:
            root_node = root_node.parent
        # Update pool

    def to_dict(self):
        result = {}
        if self.parent:
            key = f"{self.parent}_{self.name}"
        else:
            key = self.name
        result[key] = self.pool
        for child in self.children.values():
            result.update(child.to_dict())
        return result

    def __repr__(self):
        return f"TreeNode({self.name} from {self.parent}, {list(self.children.keys())})"


def build_tree(pool_refs):
    root = PoolsTreeNode("HH_TAG")
    for paired_pool in pool_refs:
        root_rela, sample_rela = paired_pool.split("-")
        parent_node = root.add_child(root_rela)
        parent_node.add_child(sample_rela)
    return root


# Example usage
pool_refs = {
    "A-B": "some_value",
    "A-C": "some_value",
    "B-D": "some_value",
    "C-E": "some_value",
}

tree = build_tree(pool_refs)
print(tree)
