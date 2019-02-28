mod arff;
mod decision;
mod evaluate;
mod record;

use clap::{ App, Arg };
use std::fs;
use evaluate::evaluate;

fn main() {
    let matches = App::new("decision tree")
        .version("1.0")
        .author("Pearce Keesling")
        .arg(Arg::with_name("file")
             .short("f")
             .long("file")
             .required(true)
             .takes_value(true))
        .arg(Arg::with_name("validation")
             .short("v")
             .required(true)
             .number_of_values(2)
             .takes_value(true))
        .arg(Arg::with_name("output-count")
             .long("output-count")
             .takes_value(true))
        .arg(Arg::with_name("learning-rate")
             .long("learning-rate")
             .takes_value(true))
        .arg(Arg::with_name("momentum")
             .long("momentum")
             .takes_value(true))
        .arg(Arg::with_name("learner")
             .long("learner")
             .short("l")
             .takes_value(true))
        .get_matches();
    let file = matches.value_of("file").unwrap();

    use self::arff::Arff;
    let mut data = Arff::parse(&fs::read_to_string(file).expect("file not found")).expect("parse failed");
    // data.normalize();
    data.shuffle();
    // dbg!(&data.records);
    let mut learner = decision::DecisionTree::default();
    let mut validation_values = matches.values_of("validation").unwrap();
    match validation_values.next() {
        Some("random") => {
            if let Some(training_size) = validation_values.next().and_then(|val| val.parse::<f64>().ok()) {
                let training_count = (data.records.len() as f64 * (training_size / 100.0)) as usize;
                let (train, test) = data.records.split_at(training_count);
                learner.train(train, &data.classes);
                let accuracy = evaluate(test, &learner);
                println!("test accuracy: {}", accuracy);
            }
        },
        Some("training") => {
                learner.train(data.records.as_slice(), &data.classes);
                // dbg!(learner);
        },
        _ => {}
    }

}
