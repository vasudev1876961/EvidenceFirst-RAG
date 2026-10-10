"""Interactive Visualizations for Evidence Knowledge Graphs and Verification Flow (Sankey & Network)."""

from __future__ import annotations
import json
import logging
from typing import Dict, Any, List
from src.models import SynthesisReport

logger = logging.getLogger(__name__)


class EvidenceGraphVisualizer:
    """Generates publication-ready interactive HTML visualizers:

    1. Multi-Stage Sankey Diagram (Query -> Papers -> Claims -> Verification Consensus)
    2. Interactive Vis.js Physics Network Graph (Green = Support, Red = Contradiction)
    """

    @staticmethod
    def build_sankey_data(report: SynthesisReport) -> Dict[str, Any]:
        """Construct Sankey nodes and directed weighted flows for literature verification."""
        labels: List[str] = []
        colors: List[str] = []
        label_to_index: Dict[str, int] = {}

        def get_or_add_node(name: str, color: str = "#3B82F6") -> int:
            if name not in label_to_index:
                idx = len(labels)
                labels.append(name)
                colors.append(color)
                label_to_index[name] = idx
                return idx
            return label_to_index[name]

        # Stage 1: Root Query
        query_node = get_or_add_node(f"Query: {report.query[:35]}...", "#8B5CF6")

        sources: List[int] = []
        targets: List[int] = []
        values: List[float] = []
        link_colors: List[str] = []

        # Color mapping for verification status
        status_colors = {
            "Strong support": "#10B981",     # Emerald green
            "Moderate support": "#3B82F6",   # Blue
            "Conflicting": "#EF4444",        # Red
            "Weak support": "#F59E0B",       # Amber
            "Unsupported": "#6B7280",        # Gray
        }

        # Track papers and claims
        for claim in report.claims:
            c_label = f"[{claim.claim_id}] {claim.statement[:32]}..."
            c_node = get_or_add_node(c_label, "#EC4899")

            # Link Query -> Claim
            sources.append(query_node)
            targets.append(c_node)
            values.append(1.0)
            link_colors.append("rgba(139, 92, 246, 0.4)")

            # Supporting Papers
            for ev in claim.supporting_evidence[:2]:
                p_label = f"Paper: {ev.paper_id} ({ev.year})"
                p_node = get_or_add_node(p_label, "#10B981")
                sources.append(c_node)
                targets.append(p_node)
                values.append(max(0.4, ev.composite_score))
                link_colors.append("rgba(16, 185, 129, 0.4)")

            # Contradicting Papers
            for ev in claim.contradicting_evidence[:2]:
                p_label = f"Paper: {ev.paper_id} ({ev.year})"
                p_node = get_or_add_node(p_label, "#EF4444")
                sources.append(c_node)
                targets.append(p_node)
                values.append(max(0.4, ev.composite_score))
                link_colors.append("rgba(239, 68, 68, 0.4)")

            # Link Claim -> Verification Consensus Status
            stat_name = f"Status: {claim.evidence_status}"
            stat_color = status_colors.get(claim.evidence_status, "#6B7280")
            stat_node = get_or_add_node(stat_name, stat_color)

            sources.append(c_node)
            targets.append(stat_node)
            values.append(1.2)
            link_colors.append(f"{stat_color}66")

        return {
            "nodes": labels,
            "node_colors": colors,
            "sources": sources,
            "targets": targets,
            "values": values,
            "link_colors": link_colors,
        }

    @classmethod
    def generate_sankey_html(cls, report: SynthesisReport, title: str = "EvidenceFirst-RAG: Verification Flow") -> str:
        """Generate standalone interactive HTML with Plotly.js for the Sankey flow."""
        data = cls.build_sankey_data(report)
        sankey_json = json.dumps(data)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>{title}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{
      margin: 0;
      padding: 24px;
      background-color: #0F172A;
      color: #F8FAFC;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }}
    .header {{
      margin-bottom: 20px;
      border-bottom: 1px solid #334155;
      padding-bottom: 12px;
    }}
    h1 {{
      font-size: 24px;
      color: #38BDF8;
      margin: 0 0 6px 0;
    }}
    p {{
      color: #94A3B8;
      margin: 0;
      font-size: 14px;
    }}
    #sankey-div {{
      width: 100%;
      height: 650px;
      border-radius: 12px;
      background: #1E293B;
      border: 1px solid #334155;
      box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    }}
  </style>
</head>
<body>
  <div class="header">
    <h1>🔬 EvidenceFirst-RAG: Citation & Verification Sankey Flow</h1>
    <p>Traceability flow: Research Query ➔ Extracted Claims ➔ Supporting / Contradicting Papers ➔ Final Verification Consensus</p>
  </div>
  <div id="sankey-div"></div>
  <script>
    const data = {sankey_json};
    const plotData = [{{
      type: "sankey",
      orientation: "h",
      node: {{
        pad: 18,
        thickness: 24,
        line: {{ color: "#0F172A", width: 1.5 }},
        label: data.nodes,
        color: data.node_colors
      }},
      link: {{
        source: data.sources,
        target: data.targets,
        value: data.values,
        color: data.link_colors
      }}
    }}];
    const layout = {{
      paper_bgcolor: "#1E293B",
      plot_bgcolor: "#1E293B",
      font: {{ color: "#F8FAFC", size: 12, family: "sans-serif" }},
      margin: {{ l: 20, r: 20, t: 25, b: 25 }}
    }};
    Plotly.newPlot("sankey-div", plotData, layout, {{ responsive: true, displayModeBar: false }});
  </script>
</body>
</html>
"""
        return html

    @classmethod
    def generate_network_html(cls, report: SynthesisReport, title: str = "EvidenceFirst-RAG: Interactive Knowledge Graph") -> str:
        """Generate standalone interactive HTML network graph using Vis.js Physics Engine."""
        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []

        seen_nodes = set()

        def add_node(n_id: str, label: str, group: str, color: str):
            if n_id not in seen_nodes:
                seen_nodes.add(n_id)
                nodes.append({
                    "id": n_id,
                    "label": label,
                    "group": group,
                    "color": {"background": color, "border": "#FFFFFF", "highlight": {"background": "#F59E0B", "border": "#FFFFFF"}},
                    "font": {"color": "#FFFFFF", "size": 13, "face": "sans-serif"},
                    "shape": "box" if group in ["claim", "query"] else "ellipse",
                })

        # Query node
        q_id = "query_root"
        add_node(q_id, f"Query:\n{report.query[:45]}...", "query", "#8B5CF6")

        for claim in report.claims:
            c_id = claim.claim_id
            stat_color = "#10B981" if "Strong" in claim.evidence_status else ("#EF4444" if "Conflict" in claim.evidence_status else "#3B82F6")
            add_node(c_id, f"[{claim.claim_id}]\n{claim.statement[:40]}...", "claim", stat_color)
            edges.append({
                "from": q_id,
                "to": c_id,
                "label": "CLAIM",
                "color": {"color": "#8B5CF6", "opacity": 0.5},
                "arrows": "to",
            })

            # Supporting evidence
            for ev in claim.supporting_evidence:
                p_id = f"paper_{ev.paper_id}"
                add_node(p_id, f"📄 {ev.paper_id}\n({ev.year})", "paper", "#059669")
                edges.append({
                    "from": p_id,
                    "to": c_id,
                    "label": f"SUPPORTS ({ev.composite_score:.2f})",
                    "color": {"color": "#10B981", "highlight": "#34D399"},
                    "width": 2.5,
                    "arrows": "to",
                })

            # Contradicting evidence
            for ev in claim.contradicting_evidence:
                p_id = f"paper_{ev.paper_id}"
                add_node(p_id, f"📄 {ev.paper_id}\n({ev.year})", "paper", "#DC2626")
                edges.append({
                    "from": p_id,
                    "to": c_id,
                    "label": f"CONTRADICTS ({ev.composite_score:.2f})",
                    "color": {"color": "#EF4444", "highlight": "#F87171"},
                    "dashes": True,
                    "width": 2.5,
                    "arrows": "to",
                })

        nodes_json = json.dumps(nodes)
        edges_json = json.dumps(edges)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>{title}</title>
  <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <style>
    body {{
      margin: 0;
      padding: 24px;
      background-color: #0B1120;
      color: #F8FAFC;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }}
    .header {{
      margin-bottom: 16px;
      border-bottom: 1px solid #1E293B;
      padding-bottom: 12px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    h1 {{
      font-size: 22px;
      color: #38BDF8;
      margin: 0 0 6px 0;
    }}
    p {{
      color: #94A3B8;
      margin: 0;
      font-size: 13px;
    }}
    .legend {{
      display: flex;
      gap: 16px;
      font-size: 13px;
    }}
    .legend-item {{
      display: flex;
      align-items: center;
      gap: 6px;
    }}
    .legend-color {{
      width: 14px;
      height: 14px;
      border-radius: 3px;
    }}
    #network-container {{
      width: 100%;
      height: 700px;
      border-radius: 12px;
      background: #111827;
      border: 1px solid #1E293B;
      box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
    }}
  </style>
</head>
<body>
  <div class="header">
    <div>
      <h1>🔬 Multi-Hop Scientific Evidence & Contradiction Graph</h1>
      <p>Interactive knowledge graph showing directional entailment (Green) and contradiction (Red dashed) flows.</p>
    </div>
    <div class="legend">
      <div class="legend-item"><div class="legend-color" style="background: #10B981;"></div> Entailment / Support</div>
      <div class="legend-item"><div class="legend-color" style="background: #EF4444;"></div> Contradiction / Conflict</div>
      <div class="legend-item"><div class="legend-color" style="background: #8B5CF6;"></div> Query Root</div>
    </div>
  </div>
  <div id="network-container"></div>
  <script>
    const nodes = new vis.DataSet({nodes_json});
    const edges = new vis.DataSet({edges_json});
    const container = document.getElementById("network-container");
    const data = {{ nodes: nodes, edges: edges }};
    const options = {{
      physics: {{
        stabilization: true,
        barnesHut: {{
          gravitationalConstant: -3500,
          centralGravity: 0.35,
          springLength: 120,
          springConstant: 0.04
        }}
      }},
      interaction: {{
        hover: true,
        tooltipDelay: 150,
        navigationButtons: true,
        keyboard: true
      }}
    }};
    const network = new vis.Network(container, data, options);
  </script>
</body>
</html>
"""
        return html
