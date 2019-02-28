use crate::record::Record;
use itertools::Itertools;
use std::collections::HashMap;

#[derive(Debug, Default)]
struct BranchNode {
    paths: HashMap<usize, usize>,
    feature: usize,
    majority_class: usize,
}

#[derive(Debug, Default)]
struct LeafNode {
    class: usize,
}

#[derive(Debug)]
enum Node {
    Branch(BranchNode),
    Leaf(LeafNode),
}

#[derive(Debug, Default)]
pub struct DecisionTree {
    nodes: Vec<Node>,
}

fn calculate_information(distribution: &[u8], total_size: f32) -> f32 {
    distribution
        .iter()
        .filter(|val| **val > 0)
        .map(|val| *val as f32)
        .map(|val| val / total_size)
        .map(|frac| frac * frac.log2())
        .fold(0., |acc, v| acc + v)
}

fn build_distribution(sub_set: &[&Record], total_size: usize) -> Vec<u8> {
    let mut distribution = Vec::with_capacity(total_size);
    for _ in 0..total_size {
        distribution.push(0);
    }
    for record in sub_set.iter() {
        distribution[record.class] += 1;
    }
    distribution
}

impl DecisionTree {
    pub fn train(&mut self, data: &[Record], class_tags: &Vec<Vec<String>>) {
        let mut stack: Vec<(Vec<usize>, Vec<&Record>, Option<(usize, usize)>)> =
            Vec::with_capacity((2.0_f32).powf(class_tags.len() as f32) as usize);
        loop {
            let (mut used_features, sub_set, parent_info) = stack
                .pop()
                .unwrap_or_else(|| (Vec::new(), data.iter().collect(), None));
            if let Some((index, feature)) = parent_info {
                let node_index = self.nodes.len();
                if let Node::Branch(ref mut parent) = &mut self.nodes[index] {
                    parent.paths.insert(feature, node_index);
                }
            }
            let distribution = build_distribution(&sub_set, class_tags.last().unwrap().len());
            let class_count = distribution.iter().filter(|count| **count > 0).count();
            if class_count == 1 {
                self.nodes.push(Node::Leaf(LeafNode {
                    class: sub_set[0].class,
                }));
            } else if used_features.len() < class_tags.len() - 1 {
                let (feature, _info) = class_tags
                    .split_last()
                    .unwrap()
                    .1
                    .iter()
                    .enumerate()
                    .filter(|(index, _)| {
                        used_features
                            .iter()
                            .all(|feature_index| feature_index != index)
                    })
                    .map(|(feature, classes)| {
                        (
                            feature,
                            (0..classes.len())
                                .into_iter()
                                .map(|feature_value| {
                                    let potential_sub_set: Vec<&Record> = sub_set
                                        .iter()
                                        .cloned()
                                        .filter(|record| record.features[feature] == feature_value)
                                        .collect();
                                    let distribution = build_distribution(
                                        potential_sub_set.as_slice(),
                                        classes.last().expect("no classes").len(),
                                    );
                                    calculate_information(&distribution, sub_set.len() as f32)
                                        * (potential_sub_set.len() as f32 / sub_set.len() as f32)
                                        * -1.
                                })
                                .fold(0., |acc, v| acc + v),
                        )
                    })
                    .min_by(|x, y| x.1.partial_cmp(&y.1).unwrap())
                    .unwrap();
                used_features.push(feature);
                let mut child_sub_sets = sub_set
                    .iter()
                    .map(|record| (record.features[feature], *record))
                    .into_group_map();
                let (majority_class, _) = distribution.iter().enumerate().max_by(|x, y| x.1.cmp(y.1)).unwrap();
                self.nodes.push(Node::Branch(BranchNode {
                    feature,
                    paths: HashMap::new(),
                    majority_class,
                }));
                for (feature, child_subset) in child_sub_sets.drain() {
                    if !child_subset.is_empty() {
                        stack.push((
                            used_features.clone(),
                            child_subset,
                            Some((self.nodes.len() - 1, feature)),
                        ));
                    }
                }
            } else {
                println!("undecided: used {:?}:  {:?}", used_features, distribution);
            }
            if stack.is_empty() {
                break;
            }
        }
    }
    pub fn predict(&self, record: &Record) -> usize {
        let mut node_index = 0;
        loop {
            let node = &self.nodes[node_index];
            if let Node::Branch(ref branch) = node {
                if let Some(index) = branch.paths.get(&record.features[branch.feature]) {
                    node_index = *index;
                } else {
                    return branch.majority_class;
                }
            }
            if let Node::Leaf(ref leaf) = node {
                return leaf.class;
            }
        }
    }
}
