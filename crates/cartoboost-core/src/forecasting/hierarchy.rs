use std::collections::BTreeMap;

use crate::{CartoBoostError, Result};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HierarchyNode {
    pub id: String,
    pub parent: Option<String>,
}

impl HierarchyNode {
    pub fn new(id: impl Into<String>, parent: Option<impl Into<String>>) -> Self {
        Self {
            id: id.into(),
            parent: parent.map(Into::into),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SparseHierarchyRow {
    pub node_index: usize,
    pub bottom_weights: Vec<(usize, f64)>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HierarchySpec {
    nodes: Vec<HierarchyNode>,
    node_index: BTreeMap<String, usize>,
    children: Vec<Vec<usize>>,
    bottom_indices: Vec<usize>,
    bottom_position_by_node: BTreeMap<usize, usize>,
    sparse_rows: Vec<SparseHierarchyRow>,
}

impl HierarchySpec {
    pub fn new(nodes: Vec<HierarchyNode>) -> Result<Self> {
        if nodes.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "hierarchy must contain at least one node".to_string(),
            ));
        }

        let mut node_index = BTreeMap::new();
        for (idx, node) in nodes.iter().enumerate() {
            if node.id.trim().is_empty() {
                return Err(CartoBoostError::InvalidInput(
                    "hierarchy node ids must be non-empty".to_string(),
                ));
            }
            if node_index.insert(node.id.clone(), idx).is_some() {
                return Err(CartoBoostError::InvalidInput(format!(
                    "duplicate hierarchy node id '{}'",
                    node.id
                )));
            }
        }

        let mut children = vec![Vec::new(); nodes.len()];
        for (idx, node) in nodes.iter().enumerate() {
            if let Some(parent) = &node.parent {
                let parent_idx = *node_index.get(parent).ok_or_else(|| {
                    CartoBoostError::InvalidInput(format!(
                        "hierarchy parent '{}' for node '{}' is missing",
                        parent, node.id
                    ))
                })?;
                if parent_idx == idx {
                    return Err(CartoBoostError::InvalidInput(format!(
                        "hierarchy node '{}' cannot be its own parent",
                        node.id
                    )));
                }
                children[parent_idx].push(idx);
            }
        }

        let roots = nodes.iter().filter(|node| node.parent.is_none()).count();
        if roots == 0 {
            return Err(CartoBoostError::InvalidInput(
                "hierarchy must contain at least one root".to_string(),
            ));
        }

        let mut visiting = vec![false; nodes.len()];
        let mut visited = vec![false; nodes.len()];
        for idx in 0..nodes.len() {
            Self::visit_acyclic(idx, &children, &mut visiting, &mut visited)?;
        }

        let bottom_indices = children
            .iter()
            .enumerate()
            .filter_map(|(idx, node_children)| node_children.is_empty().then_some(idx))
            .collect::<Vec<_>>();
        let bottom_position_by_node = bottom_indices
            .iter()
            .enumerate()
            .map(|(bottom_pos, node_idx)| (*node_idx, bottom_pos))
            .collect::<BTreeMap<_, _>>();

        let mut sparse_rows = Vec::with_capacity(nodes.len());
        for node_idx in 0..nodes.len() {
            let mut weights = Vec::new();
            Self::collect_bottom_weights(
                node_idx,
                &children,
                &bottom_position_by_node,
                &mut weights,
            );
            sparse_rows.push(SparseHierarchyRow {
                node_index: node_idx,
                bottom_weights: weights,
            });
        }

        Ok(Self {
            nodes,
            node_index,
            children,
            bottom_indices,
            bottom_position_by_node,
            sparse_rows,
        })
    }

    pub fn from_edges(edges: Vec<(&str, Option<&str>)>) -> Result<Self> {
        Self::new(
            edges
                .into_iter()
                .map(|(id, parent)| HierarchyNode::new(id, parent))
                .collect(),
        )
    }

    pub fn nodes(&self) -> &[HierarchyNode] {
        &self.nodes
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn bottom_count(&self) -> usize {
        self.bottom_indices.len()
    }

    pub fn bottom_node_ids(&self) -> Vec<&str> {
        self.bottom_indices
            .iter()
            .map(|idx| self.nodes[*idx].id.as_str())
            .collect()
    }

    pub fn sparse_rows(&self) -> &[SparseHierarchyRow] {
        &self.sparse_rows
    }

    pub fn node_index(&self, id: &str) -> Option<usize> {
        self.node_index.get(id).copied()
    }

    pub fn bottom_position_for_node(&self, node_idx: usize) -> Option<usize> {
        self.bottom_position_by_node.get(&node_idx).copied()
    }

    pub fn level_indices(&self, depth: usize) -> Vec<usize> {
        (0..self.nodes.len())
            .filter(|idx| self.depth(*idx) == depth)
            .collect()
    }

    pub fn depth(&self, node_idx: usize) -> usize {
        let mut depth = 0;
        let mut current = node_idx;
        while let Some(parent) = &self.nodes[current].parent {
            current = self.node_index[parent];
            depth += 1;
        }
        depth
    }

    pub fn descendants_bottom_positions(&self, node_idx: usize) -> Vec<usize> {
        self.sparse_rows[node_idx]
            .bottom_weights
            .iter()
            .map(|(idx, _)| *idx)
            .collect()
    }

    pub fn aggregate_bottom_values(&self, bottom_values: &[f64]) -> Result<Vec<f64>> {
        if bottom_values.len() != self.bottom_count() {
            return Err(CartoBoostError::InvalidInput(format!(
                "expected {} bottom values, got {}",
                self.bottom_count(),
                bottom_values.len()
            )));
        }
        if !bottom_values.iter().all(|value| value.is_finite()) {
            return Err(CartoBoostError::InvalidInput(
                "bottom values must be finite".to_string(),
            ));
        }
        Ok(self
            .sparse_rows
            .iter()
            .map(|row| {
                row.bottom_weights
                    .iter()
                    .map(|(bottom_idx, weight)| bottom_values[*bottom_idx] * weight)
                    .sum()
            })
            .collect())
    }

    pub fn is_coherent(&self, node_values: &[f64], tolerance: f64) -> Result<bool> {
        if node_values.len() != self.node_count() {
            return Err(CartoBoostError::InvalidInput(format!(
                "expected {} node values, got {}",
                self.node_count(),
                node_values.len()
            )));
        }
        if !node_values.iter().all(|value| value.is_finite()) {
            return Err(CartoBoostError::InvalidInput(
                "node values must be finite".to_string(),
            ));
        }
        let bottom = self
            .bottom_indices
            .iter()
            .map(|node_idx| node_values[*node_idx])
            .collect::<Vec<_>>();
        let aggregated = self.aggregate_bottom_values(&bottom)?;
        Ok(aggregated
            .iter()
            .zip(node_values)
            .all(|(expected, actual)| (expected - actual).abs() <= tolerance))
    }

    fn visit_acyclic(
        node_idx: usize,
        children: &[Vec<usize>],
        visiting: &mut [bool],
        visited: &mut [bool],
    ) -> Result<()> {
        if visited[node_idx] {
            return Ok(());
        }
        if visiting[node_idx] {
            return Err(CartoBoostError::InvalidInput(
                "hierarchy must be acyclic".to_string(),
            ));
        }
        visiting[node_idx] = true;
        for child in &children[node_idx] {
            Self::visit_acyclic(*child, children, visiting, visited)?;
        }
        visiting[node_idx] = false;
        visited[node_idx] = true;
        Ok(())
    }

    fn collect_bottom_weights(
        node_idx: usize,
        children: &[Vec<usize>],
        bottom_position_by_node: &BTreeMap<usize, usize>,
        weights: &mut Vec<(usize, f64)>,
    ) {
        if let Some(bottom_pos) = bottom_position_by_node.get(&node_idx) {
            weights.push((*bottom_pos, 1.0));
            return;
        }
        for child in &children[node_idx] {
            Self::collect_bottom_weights(*child, children, bottom_position_by_node, weights);
        }
        weights.sort_by_key(|(bottom_idx, _)| *bottom_idx);
    }
}
