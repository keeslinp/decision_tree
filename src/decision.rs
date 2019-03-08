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

fn calculate_information(distribution: &[u32], total_size: f32) -> f32 {
    distribution
        .iter()
        .filter(|val| **val > 0)
        .map(|val| *val as f32)
        .map(|val| val / total_size)
        .map(|frac| frac * frac.log2())
        .fold(0., |acc, v| acc + v)
}

fn build_distribution(sub_set: &[&Record], total_size: usize) -> Vec<u32> {
    let mut distribution = Vec::with_capacity(total_size);
    for _ in 0..total_size {
        distribution.push(0);
    }
    for record in sub_set.iter() {
        distribution[record.class] += 1;
    }
    distribution
}

use crate::arff::Class;
impl DecisionTree {
    pub fn train(&mut self, data: &[Record], class_tags: &Vec<Class>) {
        // Forget previous training
        self.nodes = Vec::new();
        let mut stack: Vec<(Vec<usize>, Vec<&Record>, Option<(usize, usize)>)> =
            Vec::with_capacity((2.0_f32).powf(class_tags.len() as f32) as usize);
        let training_count = (data.len() as f32 * 1.) as usize;
        let (training, validation) = data.split_at(training_count);
        let mut previous_accuracy = 0.;
        loop {
            let (mut used_features, sub_set, parent_info) = stack
                .pop()
                .unwrap_or_else(|| (Vec::new(), training.iter().collect(), None));
            let output_count = if let Some(Class::Nominal(class)) = class_tags.last() {
                class.len()
            } else {
                unreachable!();
            };
            let distribution = build_distribution(&sub_set, output_count);
            let class_count = distribution.iter().filter(|count| **count > 0).count();
            // avoid making dead connections
            if class_count == 1 || used_features.len() < class_tags.len() - 1 {
                if let Some((index, feature)) = parent_info {
                    let node_index = self.nodes.len();
                    if let Node::Branch(ref mut parent) = &mut self.nodes[index] {
                        parent.paths.insert(feature, node_index);
                    }
                }
            }
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
                        let count = match classes {
                            Class::Nominal(list) => list.len(),
                            Class::Continuous(max) => *max,
                        };
                        (
                            feature,
                            (0..count)
                                .into_iter()
                                .map(|feature_value| {
                                    let potential_sub_set: Vec<&Record> = sub_set
                                        .iter()
                                        .cloned()
                                        .filter(|record| record.features[feature] == feature_value)
                                        .collect();
                                    let distribution = build_distribution(
                                        potential_sub_set.as_slice(),
                                        output_count,
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
            let test_accuracy = self.test_set(training);
            // dbg!(test_accuracy);
            let validation_accuracy = self.test_set(validation);
            let _stagnant = validation_accuracy - previous_accuracy < 0.001;
            if stack.is_empty() {
                break;
            }
            previous_accuracy = validation_accuracy;
        }
        // dbg!(&self.nodes);
    }
    fn test_set(&self, records: &[Record]) -> f32 {
        records.iter().map(|record| {
            if self.predict(record) == record.class { 1 } else { 0 }
        }).fold(0, |acc, x| acc + x) as f32 / records.len() as f32
    }

    pub fn predict(&self, record: &Record) -> usize {
        let mut node_index = 0;
        loop {
            let node = &self.nodes[node_index];
            if let Node::Branch(ref branch) = node {
                if let Some(index) = branch.paths.get(&record.features[branch.feature]) {
                    node_index = *index;
                } else {
                    // println!("falling back to majority on Record: {:?}", record);
                    return branch.majority_class;
                }
            }
            if let Node::Leaf(ref leaf) = node {
                return leaf.class;
            }
        }
    }

    pub fn print_tree(&self, depth: usize, class_tags: &Vec<Class>, labels: &Vec<String>) {
        let mut current_depth = 0;
        let mut stack: Vec<((usize, usize), usize)> = vec![((0, 0), 0)]; 
        loop {
            let mut next_stack = Vec::new();
            for ((previous_choice, previous_feature), node_index) in stack.drain(..) {
                let node = &self.nodes[node_index];
                let previous_value = match &class_tags[previous_feature] {
                    Class::Continuous(_) => format!("{}", previous_choice),
                    Class::Nominal(list) => list[previous_choice].clone(),
                };
                match node {
                    Node::Branch(branch) => {
                        print!("b({}: {}, {}) | ", labels[previous_feature], previous_value, labels[branch.feature]);
                        for (k, v) in branch.paths.iter() {
                            next_stack.push(((*k, branch.feature), *v));
                        }
                    },
                    Node::Leaf(leaf) => {
                        let class = match &class_tags[class_tags.len() - 1] {
                            Class::Nominal(list) => list[leaf.class].clone(),
                            Class::Continuous(_) => format!("{}", leaf.class),
                        };
                        print!("l({}: {}, {}) | ", labels[previous_feature], previous_value, class);
                    },
                }
            }
            println!("");
            current_depth += 1;
            if next_stack.is_empty() || current_depth >= depth {
                break;
            }
            stack = next_stack;
        };
    }

    pub fn prune(&mut self, validation_set: &[Record]) {
        let mut counter = 0;
        loop {
            counter += 1;
            let current_accuracy = self.test_set(validation_set);
            let worst_node = (0..self.nodes.len()).map(|skipped_node| {
                let (accuracy, saved_paths) = if let Node::Branch(ref mut branch) = self.nodes[skipped_node] {
                    let saved_paths = std::mem::replace(&mut branch.paths, HashMap::new());
                    if saved_paths.is_empty() {
                        (0., None)
                    } else {
                        (self.test_set(validation_set), Some(saved_paths))
                    }
                } else {
                    (0., None)
                };
                if let Node::Branch(ref mut branch) = self.nodes[skipped_node] {
                    if let Some(paths) = saved_paths {
                        branch.paths = paths;
                    }
                }
                accuracy
            }).enumerate().max_by(|x, y| x.1.partial_cmp(&y.1).unwrap()).unwrap();
            // dbg!((worst_node, current_accuracy));
            if worst_node.1 >= current_accuracy {
                if let Node::Branch(ref mut branch) = self.nodes[worst_node.0] {
                    branch.paths = HashMap::new();
                } else {
                    unreachable!()
                }
            } else {
                break;
            }
        }
        dbg!(counter);
    }

    fn count_children(&self, node_index: usize) -> usize {
        match &self.nodes[node_index] {
            Node::Branch(branch) => {
                branch.paths.values().map(|node| {
                    1 + self.count_children(*node)
                }).fold(0, |acc, val| acc + val)
            },
            Node::Leaf(_) => {
                1
            },
        }
    }

    pub fn count_live_nodes(&self) -> usize {
        self.count_children(0)
    }

    fn max_children(&self, node_index: usize) -> usize {
        match &self.nodes[node_index] {
            Node::Branch(branch) => {
                branch.paths.values().map(|node| {
                    self.max_children(*node)
                }).max().unwrap_or(0) + 1
            },
            Node::Leaf(_) => {
                1
            },
        }
    }

    pub fn max_depth(&self) -> usize {
        self.max_children(0)
    }

    pub fn count_pruned_nodes(&self) -> usize {
        self.nodes.len() - self.count_children(0)
    }
}
