"""Process-local cuRobo compatibility patches owned by visual manipulation.

cuRobo is third-party code. Keep version-specific workarounds here so upgrading
or replacing that dependency requires no source edits outside this repository.

This module is the sole compatibility boundary for cuRobo's Python-side APIs:
call an installer during local planner import, before any planner is allocated.
Remove a shim only after its reproducer passes with the target cuRobo and
NetworkX versions; do not copy its logic into the third-party checkout.
"""

import networkx as nx

from curobo._src.graph_planner.search.path_finder_networkx import NetworkXPathFinder


def install_networkx_edge_buffer_compat() -> None:
    """Patch PRM edge-buffer flushing for Python 3.12 / NetworkX 3.5.

    Trigger: after several repeated solves, cuRobo may retain ``nx.Graph`` in
    ``NetworkXPathFinder.edge_list`` instead of its normal ``(u, v, weight)``
    list. cuRobo then passes that graph to ``add_weighted_edges_from`` and fails
    with ``TypeError: 'Graph' object is not iterable`` during a later retract.

    Contract: preserve cuRobo's normal triple-list path exactly. Only convert a
    retained undirected NetworkX graph to a materialized weighted-edge list, then
    delegate to its original method. The class-level patch is process-local and
    guarded so repeated imports never stack wrappers.
    """
    if getattr(NetworkXPathFinder, "_legged_env_edge_buffer_compat", False):
        return

    # Delegate after normalization. Reimplementing cuRobo's method here would
    # silently diverge when a future cuRobo release changes node/edge flushing.
    original_update_graph = NetworkXPathFinder.update_graph

    def update_graph(self):
        if isinstance(self.edge_list, nx.Graph):
            # Materialize before cuRobo clears its buffer after merging. NetworkX's
            # EdgeDataView yields exactly the (u, v, weight) triples it expects.
            self.edge_list = list(self.edge_list.edges.data("weight"))
        return original_update_graph(self)

    NetworkXPathFinder.update_graph = update_graph
    NetworkXPathFinder._legged_env_edge_buffer_compat = True
