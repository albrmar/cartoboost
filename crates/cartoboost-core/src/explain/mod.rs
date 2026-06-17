use crate::tree::{Node, Tree};

pub fn dump_tree(tree: &Tree) -> String {
    fn walk(node: &Node, depth: usize, out: &mut String) {
        let indent = "  ".repeat(depth);
        match node {
            Node::Leaf { value, .. } => out.push_str(&format!("{indent}leaf value={value:.6}\n")),
            Node::LinearLeaf { model, .. } => out.push_str(&format!(
                "{indent}linear_leaf intercept={:.6} coefficients={:?}\n",
                model.intercept, model.coefficients
            )),
            Node::Branch {
                split,
                left,
                right,
                gain,
                ..
            } => {
                out.push_str(&format!("{indent}{split:?} gain={gain:.6}\n"));
                walk(left, depth + 1, out);
                walk(right, depth + 1, out);
            }
        }
    }
    let mut out = String::new();
    walk(&tree.root, 0, &mut out);
    out
}
