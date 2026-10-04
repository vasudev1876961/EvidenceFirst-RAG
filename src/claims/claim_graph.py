"""Evidence Knowledge Graph built using NetworkX for multi-hop scientific literature visualization."""

from __future__ import annotations
import networkx as nx
from typing import List, Dict, Any, Optional
from src.models import (
    FactualClaim,
    EvidenceGraphData,
    GraphNode,
    GraphEdge,
    SynthesisReport
)


class EvidenceGraphBuilder:
    """Constructs and serializes the Research Evidence Knowledge Graph."""

    def __init__(self):
        self.graph = nx.DiGraph()

    def build_graph(self, query: str, claims: List[FactualClaim]) -> EvidenceGraphData:
        """Build directed knowledge graph: Query -> Claims -> Papers -> Datasets."""
        self.graph.clear()
        nodes: List[GraphNode] = []
        edges: List[GraphEdge] = []

        # 1. Add Query node
        query_node_id = "node_query"
        self.graph.add_node(query_node_id, label="Research Query", type="query", full_text=query)
        nodes.append(GraphNode(id=query_node_id, label="Research Query", node_type="query", attributes={"query": query}))

        seen_papers = set()
        seen_datasets = set()

        for claim in claims:
            # 2. Add Claim node
            claim_node_id = f"node_{claim.claim_id}"
            status = claim.evidence_status
            self.graph.add_node(
                claim_node_id,
                label=claim.claim_id,
                type="claim",
                statement=claim.statement,
                status=status,
                confidence=claim.confidence_score,
            )
            nodes.append(
                GraphNode(
                    id=claim_node_id,
                    label=f"{claim.claim_id}: {claim.subquery_source or 'Claim'}",
                    node_type="claim",
                    attributes={"statement": claim.statement, "status": status, "confidence": claim.confidence_score},
                )
            )

            # Edge from Query -> Claim
            self.graph.add_edge(query_node_id, claim_node_id, relationship="SUBQUERY")
            edges.append(GraphEdge(source=query_node_id, target=claim_node_id, relationship="SUBQUERY", weight=1.0))

            # 3. Add Supporting Evidence & Papers
            for ev in claim.supporting_evidence:
                p_node_id = f"paper_{ev.paper_id}"
                if p_node_id not in seen_papers:
                    self.graph.add_node(p_node_id, label=ev.paper_id, type="paper", title=ev.paper_title, year=ev.year)
                    nodes.append(
                        GraphNode(
                            id=p_node_id,
                            label=f"[{ev.paper_id}] ({ev.year})",
                            node_type="paper",
                            attributes={"title": ev.paper_title, "year": ev.year},
                        )
                    )
                    seen_papers.add(p_node_id)

                self.graph.add_edge(claim_node_id, p_node_id, relationship="SUPPORTS", score=ev.composite_score)
                edges.append(GraphEdge(source=claim_node_id, target=p_node_id, relationship="SUPPORTS", weight=ev.composite_score))

                # Dataset attribution
                ds = ev.extracted_attributes.get("dataset")
                if ds and ds not in seen_datasets:
                    ds_node_id = f"dataset_{ds}"
                    self.graph.add_node(ds_node_id, label=ds, type="dataset")
                    nodes.append(GraphNode(id=ds_node_id, label=f"Dataset: {ds}", node_type="dataset", attributes={"name": ds}))
                    seen_datasets.add(ds)
                    self.graph.add_edge(p_node_id, ds_node_id, relationship="USES_DATASET")
                    edges.append(GraphEdge(source=p_node_id, target=ds_node_id, relationship="USES_DATASET", weight=1.0))

            # 4. Add Contradicting Evidence & Papers
            for ev in claim.contradicting_evidence:
                p_node_id = f"paper_{ev.paper_id}"
                if p_node_id not in seen_papers:
                    self.graph.add_node(p_node_id, label=ev.paper_id, type="paper", title=ev.paper_title, year=ev.year)
                    nodes.append(
                        GraphNode(
                            id=p_node_id,
                            label=f"[{ev.paper_id}] ({ev.year})",
                            node_type="paper",
                            attributes={"title": ev.paper_title, "year": ev.year},
                        )
                    )
                    seen_papers.add(p_node_id)

                self.graph.add_edge(claim_node_id, p_node_id, relationship="CONTRADICTS", score=ev.contradiction_prob)
                edges.append(GraphEdge(source=claim_node_id, target=p_node_id, relationship="CONTRADICTS", weight=ev.contradiction_prob))

                # Dataset attribution
                ds = ev.extracted_attributes.get("dataset")
                if ds and ds not in seen_datasets:
                    ds_node_id = f"dataset_{ds}"
                    self.graph.add_node(ds_node_id, label=ds, type="dataset")
                    nodes.append(GraphNode(id=ds_node_id, label=f"Dataset: {ds}", node_type="dataset", attributes={"name": ds}))
                    seen_datasets.add(ds)
                    self.graph.add_edge(p_node_id, ds_node_id, relationship="USES_DATASET")
                    edges.append(GraphEdge(source=p_node_id, target=ds_node_id, relationship="USES_DATASET", weight=1.0))

        return EvidenceGraphData(nodes=nodes, edges=edges)

    def to_plotly_figure(self, graph_data: Optional[EvidenceGraphData] = None):
        """Generate interactive Plotly visualization for Streamlit dashboard."""
        try:
            import plotly.graph_objects as go
        except ImportError:
            return None

        if len(self.graph.nodes) == 0 and graph_data:
            # Reconstruct from graph_data if graph is empty
            for n in graph_data.nodes:
                self.graph.add_node(n.id, label=n.label, type=n.node_type)
            for e in graph_data.edges:
                self.graph.add_edge(e.source, e.target, relationship=e.relationship)

        if len(self.graph.nodes) == 0:
            fig = go.Figure()
            fig.update_layout(title="No evidence graph data available")
            return fig

        pos = nx.spring_layout(self.graph, seed=42, k=1.2)

        # Edges
        edge_x = []
        edge_y = []
        edge_colors = []
        for edge in self.graph.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1.5, color="#888"),
            hoverinfo="none",
            mode="lines"
        )

        # Nodes
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        node_size = []

        COLOR_MAP = {
            "query": "#7C3AED",       # Purple
            "claim": "#2563EB",       # Blue
            "paper": "#059669",       # Emerald
            "dataset": "#D97706",     # Amber
            "method": "#DB2777",      # Pink
        }

        for node, data in self.graph.nodes(data=True):
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            ntype = data.get("type", "claim")
            node_text.append(f"{data.get('label', node)}<br>Type: {ntype}")
            node_color.append(COLOR_MAP.get(ntype, "#64748B"))
            node_size.append(30 if ntype == "query" else (22 if ntype == "claim" else 16))

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode="markers+text",
            hoverinfo="text",
            text=[self.graph.nodes[n].get("label", n) for n in self.graph.nodes],
            textposition="top center",
            hovertext=node_text,
            marker=dict(
                showscale=False,
                color=node_color,
                size=node_size,
                line_width=2,
                line_color="#1E293B"
            )
        )

        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=dict(text="Scientific Evidence & Contradiction Graph", font=dict(size=16, color="#E2E8F0")),
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=40),
                paper_bgcolor="rgba(15, 23, 42, 0.8)",
                plot_bgcolor="rgba(15, 23, 42, 0.8)",
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
        )
        return fig
