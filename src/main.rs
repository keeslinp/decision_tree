mod arff;
mod decision;
mod evaluate;
mod record;

use clap::{App, Arg};
use evaluate::evaluate;
use std::fs;

fn main() {
    let matches = App::new("decision tree")
        .version("1.0")
        .author("Pearce Keesling")
        .arg(
            Arg::with_name("file")
                .short("f")
                .long("file")
                .required(true)
                .takes_value(true),
        )
        .arg(
            Arg::with_name("validation")
                .short("v")
                .required(true)
                .number_of_values(2)
                .takes_value(true),
        )
        .arg(
            Arg::with_name("learner")
                .long("learner")
                .short("l")
                .takes_value(true),
        )
        .arg(Arg::with_name("prune").long("prune").short("p"))
        .get_matches();
    let file = matches.value_of("file").unwrap();
    let should_prune = matches.is_present("prune");

    use self::arff::Arff;
    let mut data =
        Arff::parse(&fs::read_to_string(file).expect("file not found")).expect("parse failed");
    // data.normalize();
    data.shuffle();
    // dbg!(&data.records);
    let mut learner = decision::DecisionTree::default();
    let mut validation_values = matches.values_of("validation").unwrap();
    match validation_values.next() {
        Some("random") => {
            if let Some(training_size) = validation_values
                .next()
                .and_then(|val| val.parse::<f64>().ok())
            {
                let training_count = (data.records.len() as f64 * (training_size / 100.0)) as usize;
                let (train, test) = data.records.split_at(training_count);
                learner.train(train, &data.classes);
                let accuracy = evaluate(test, &learner);
                println!("test accuracy: {}", accuracy);
            }
        }
        Some("training") => {
            learner.train(data.records.as_slice(), &data.classes);
            learner.print_tree(10, &data.classes, &data.labels);
            // dbg!(learner);
        }
        Some("cross") => {
            if let Some(fold_count) = validation_values
                .next()
                .and_then(|val| val.parse::<usize>().ok())
            {
                let chunk_size = (data.records.len() as f64 / fold_count as f64).ceil() as usize;
                let results: Vec<(f64, usize, usize)> = (0..data.records.len())
                    .step_by(chunk_size)
                    .map(|chunk_start| {
                        let mut training_data = Vec::with_capacity(data.records.len() - chunk_size);
                        if chunk_start > 0 {
                            training_data.extend_from_slice(&data.records[0..chunk_start]);
                        }
                        if chunk_start + chunk_size < data.records.len() {
                            training_data.extend_from_slice(
                                &data.records[chunk_start + chunk_size..data.records.len()],
                            );
                        }
                        if should_prune {
                            let training_count = (training_data.len() as f32 * 0.7) as usize;
                            let (training, validation) = training_data.split_at(training_count);
                            learner.train(training, &data.classes);
                            learner.prune(validation);
                        } else {
                            learner.train(training_data.as_slice(), &data.classes);
                        }
                        let end_index = if chunk_start + chunk_size > data.records.len() {
                            data.records.len()
                        } else {
                            chunk_start + chunk_size
                        };
                        (evaluate(&data.records[chunk_start..end_index], &learner), learner.count_live_nodes(), learner.max_depth())
                        // (evaluate(&data.records[chunk_start..end_index], &learner), 0, 0)
                    })
                    .collect();
                let average_accuracy =
                    results.iter().fold(0., |acc, x| acc + x.0) / fold_count as f64;
                dbg!(average_accuracy);
                let average_node_count =
                    results.iter().fold(0, |acc, x| acc + x.1) / fold_count;
                dbg!(average_node_count);
                let max_depth =
                    results.iter().fold(0, |acc, x| acc + x.2) / fold_count;
                dbg!(max_depth);
            }
        }
        _ => panic!("unknown validation type"),
    }
}
