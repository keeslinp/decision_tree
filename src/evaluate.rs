use crate::record::Record;
use crate::decision::DecisionTree;

pub fn evaluate(validation_data: &[Record], learner: &DecisionTree) -> f64 {
    let mut correct = 0;
    for record in validation_data.iter() {
        let predicted = learner.predict(record);
        let value = record.class;
        let hit = value == predicted;
        if hit {
            correct += 1;
        }
    }
    let accuracy = correct as f64 / validation_data.len() as f64;
    accuracy
}


